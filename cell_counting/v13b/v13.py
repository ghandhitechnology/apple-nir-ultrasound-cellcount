# v13 counter: v11 segmentation followed by a learned call for every object as no cell, clump, or
# one, two or three cells. It trains the gradient-boosting classifier on reviewer-adjudicated points
# with apple-grouped cross-validation, and applies a saved model to a run. Run it as: python v13.py
# train <run> <gt csv> <model> or python v13.py apply <run> <model> <out_dir>.

import json, random, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import joblib
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\cellcount_v11_full")
FULL = Path(__import__("os").environ["CC_FULL"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import feats2
from score_points import sq_box

USE_EMBED = "--embed" in sys.argv
ALL_IN_FRAME = "--all-in-frame" in sys.argv
FEATS = feats2.FEATS2 + ([f"e{i}" for i in range(48)] if USE_EMBED else [])
CLASSES = ["no", "clump", "1", "2", "3"]
VALUE = np.array([0, 0, 1, 2, 3], float)
P, GX, GY, W, H = 274, 2, -16, 3840, 2160


def load_feats(run):
    F = feats2.all_features2(run)
    e = Path(run) / "embed.npz"
    if USE_EMBED and e.exists():
        z = np.load(e)
        E = z["E"].astype(np.float32)
        from sklearn.decomposition import PCA

        pcs = PCA(48, random_state=0).fit_transform(E)
        Ed = pd.DataFrame(pcs, columns=[f"e{i}" for i in range(48)])
        Ed["image"] = z["image"]
        Ed["id"] = z["id"]
        F = F.merge(Ed, on=["image", "id"], how="left")
    return F


def targets(run, gt_files):
    G = pd.concat([pd.read_csv(f) for f in gt_files], ignore_index=True)
    F = load_feats(run).set_index(["image", "id"])
    rows, sq = [], []
    for img, g in G.groupby("image"):
        z = np.load(Path(run) / "labels" / f"{img}.npz")
        lab = z["labels"]
        win = z["window"]
        o = pd.read_csv(Path(run) / "objects" / f"{img}.csv")
        tree = cKDTree(o[["x", "y"]].to_numpy())
        xi = np.clip(np.round(g.x).astype(int) - win[0], 0, lab.shape[1] - 1)
        yi = np.clip(np.round(g.y).astype(int) - win[1], 0, lab.shape[0] - 1)
        oid = lab[yi, xi].astype(int)
        dd, j = tree.query(g[["x", "y"]].to_numpy())
        g = g.assign(obj=np.where(oid > 0, oid, np.where(dd <= 6, o.id.to_numpy()[j], 0)))
        for s, h in g.groupby("square"):
            x0, y0, x1, y1 = sq_box(s)
            ins = o[
                (o.x >= x0) & (o.x < x1) & (o.y >= y0) & (o.y < y1) & (o.counted | ALL_IN_FRAME)
            ]
            cw = h[h.type == "cell"].groupby("obj").w.sum()
            kw = h[h.type == "clump"].groupby("obj").w.sum()
            for _, q in ins.iterrows():
                n, k = float(cw.get(q.id, 0)), float(kw.get(q.id, 0))
                if n >= 2.5:
                    c, w = "3", 1.0
                elif n >= 1.5:
                    c, w = "2", 1.0
                elif n >= 0.75:
                    c, w = "1", 1.0 if n >= 1 else 0.6
                elif k >= 0.5:
                    c, w = "clump", 1.0 if k >= 1 else 0.6
                elif n >= 0.5:
                    c, w = "1", 0.4
                else:
                    c, w = "no", 1.0
                rows.append(dict(image=img, square=s, id=int(q.id), cls=c, w=w))
            sq.append(dict(image=img, square=s, gtc=float(h[h.type == "cell"].w.sum())))
    T = pd.DataFrame(rows).drop_duplicates(["image", "id"]).join(F[FEATS], on=["image", "id"])
    return T, pd.DataFrame(sq)


def fit(X, y, w):
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, l2_regularization=1.0, class_weight=None, random_state=0
    ).fit(X, y, sample_weight=w)


def train(run, gts, out):
    T, S = targets(run, gts)
    X = T[FEATS].to_numpy(float)
    y = T.cls.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    w = T.w.to_numpy()
    grp = T.image.str[:2].to_numpy()
    P_ = np.zeros((len(y), len(CLASSES)))
    for tr, te in GroupKFold(10).split(X, y, grp):
        m = fit(X[tr], y[tr], w[tr])
        P_[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    best = None
    for alpha in np.arange(0.8, 3.01, 0.05):
        Q = P_.copy()
        Q[:, 2:] *= alpha
        cells = VALUE[Q.argmax(1)]
        s_ = S.join(
            T.assign(c=cells).groupby(["image", "square"]).c.sum().rename("pred"),
            on=["image", "square"],
        ).fillna({"pred": 0})
        ratio = s_.pred.sum() / s_.gtc.sum()
        mae = (s_.pred - s_.gtc).abs().mean()
        if best is None or abs(ratio - 1) + 0.02 * mae < abs(best[1] - 1) + 0.02 * best[2]:
            best = (alpha, ratio, mae)
    ALPHA = float(best[0])
    print("alpha", round(ALPHA, 2), "cv ratio", round(best[1], 3))
    Q = P_.copy()
    Q[:, 2:] *= ALPHA
    T["call"] = np.array(CLASSES)[Q.argmax(1)]
    T["cells"] = VALUE[Q.argmax(1)]
    T["ev"] = P_ @ VALUE
    acc = float((T.call == T.cls).mean())
    res = {
        "n_objects": len(T),
        "acc_call": round(acc, 3),
        "confusion": pd.crosstab(T.cls, T.call).to_dict(),
    }
    for col in ("cells", "ev"):
        s = S.join(
            T.groupby(["image", "square"])[col].sum().rename("pred"), on=["image", "square"]
        ).fillna({"pred": 0})
        im = s.groupby("image")[["gtc", "pred"]].sum()
        ie = (im.pred - im.gtc) / im.gtc.clip(lower=1)
        res[col] = dict(
            sum_ratio=round(s.pred.sum() / s.gtc.sum(), 3),
            sq_mae=round((s.pred - s.gtc).abs().mean(), 2),
            img_abs_rel=round(ie.abs().mean(), 3),
            img_within10=round(float((ie.abs() <= 0.1).mean()), 3),
        )
    print(json.dumps(res, indent=1, default=str))
    m = fit(X, y, w)
    joblib.dump(dict(model=m, feats=FEATS, classes=CLASSES, run=str(run), alpha=ALPHA), out)


def square_box(r, c):
    x0 = GX + P * (c - 1)
    y0 = GY + P * (r - 1)
    return max(x0, 0), max(y0, 0), min(x0 + P, W), min(y0 + P, H)


def apply(run, model, vdir, rnd=None, per_image=3):
    M = joblib.load(model)
    vdir = Path(vdir)
    for d in ("objects", "overlays"):
        (vdir / d).mkdir(parents=True, exist_ok=True)
    F = load_feats(run)
    rows, samples = [], {}
    squares = [(r, c) for r in range(1, 9) for c in range(1, 15)]
    if rnd:
        (vdir / f"review_s{rnd}").mkdir(exist_ok=True)
    for img, f in F.groupby("image"):
        o = pd.read_csv(Path(run) / "objects" / f"{img}.csv").merge(
            f[["id"] + M["feats"]], on="id", how="left"
        )
        pr = np.zeros((len(o), len(CLASSES)))
        pr[:, M["model"].classes_] = M["model"].predict_proba(o[M["feats"]].to_numpy(float))
        pr[:, 2:] *= M.get("alpha", 1.0)
        cnt = o.counted | ALL_IN_FRAME
        o["call"] = np.where(cnt, np.array(CLASSES)[pr.argmax(1)], "edge_or_outside")
        o["cells"] = np.where(cnt, VALUE[pr.argmax(1)], 0)
        for i, c in enumerate(CLASSES):
            o[f"p_{c}"] = pr[:, i].round(3)
        o.drop(columns=M["feats"]).to_csv(vdir / "objects" / f"{img}.csv", index=False)
        z = np.load(Path(run) / "labels" / f"{img}.npz")
        lab = z["labels"]
        win = z["window"]
        keep = np.zeros(int(lab.max()) + 1, bool)
        keep[o.id[o.cells > 0].to_numpy()] = True
        cov = float(keep[lab].sum()) / (W * H)
        rows.append(
            dict(
                image=img,
                count=int(o.cells.sum()),
                objects_counted=int((o.cells > 0).sum()),
                clumps=int((o.call == "clump").sum()),
                rejected=int((o.call == "no").sum()),
                merged_objects=int((o.cells >= 2).sum()),
                coverage=round(cov, 5),
            )
        )
        im = cv2.imread(str(FULL / f"{img}.jpg"))
        col = np.zeros((int(lab.max()) + 1, 3), np.uint8)
        for i, c in zip(o.id, o.call):
            col[i] = {
                "1": (60, 200, 60),
                "2": (255, 120, 0),
                "3": (255, 120, 0),
                "clump": (0, 140, 255),
            }.get(c, (150, 150, 150))
        edge = (lab != cv2.erode(lab.astype(np.float32), np.ones((3, 3))).astype(lab.dtype)) & (
            lab > 0
        )
        ov = im.copy()
        sub = ov[win[1] : win[3], win[0] : win[2]]
        sub[edge] = col[lab[edge]]
        for _, q in o[o.cells >= 2].iterrows():
            cv2.putText(
                ov,
                str(int(q.cells)),
                (int(q.x) + 8, int(q.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 120, 0),
                2,
            )
        for r in range(9):
            y = GY + P * r
            if 0 <= y < H:
                cv2.line(ov, (0, y), (W, y), (200, 155, 75), 1)
        for c in range(15):
            cv2.line(ov, (GX + P * c, 0), (GX + P * c, H), (200, 155, 75), 1)
        cv2.imwrite(str(vdir / "overlays" / f"{img}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if rnd:
            rng = random.Random(f"{img}-{vdir.name}-s{rnd}")
            pick = sorted(rng.sample(squares, per_image))
            samples[img] = [f"r{r}c{c}" for r, c in pick]
            for r, c in pick:
                x0, y0, x1, y1 = square_box(r, c)
                m_ = 20
                cx0, cy0, cx1, cy1 = (
                    max(x0 - m_, 0),
                    max(y0 - m_, 0),
                    min(x1 + m_, W),
                    min(y1 + m_, H),
                )
                for tag, src in (("raw", im), ("ovl", ov)):
                    crop = cv2.resize(
                        src[cy0:cy1, cx0:cx1], None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4
                    )
                    cv2.rectangle(
                        crop,
                        ((x0 - cx0) * 3, (y0 - cy0) * 3),
                        ((x1 - cx0) * 3 - 1, (y1 - cy0) * 3 - 1),
                        (0, 220, 255),
                        2,
                    )
                    if tag == "ovl":
                        for _, q in o[
                            (o.x >= cx0) & (o.x < cx1) & (o.y >= cy0) & (o.y < cy1)
                        ].iterrows():
                            p_ = (int((q.x - cx0) * 3) + 6, int((q.y - cy0) * 3) - 6)
                            cv2.putText(
                                crop,
                                str(q.id),
                                p_,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                crop,
                                str(q.id),
                                p_,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    cv2.imwrite(str(vdir / f"review_s{rnd}" / f"{img}_r{r}c{c}_{tag}.png"), crop)
                sq = o[(o.x >= x0 - 15) & (o.x < x1 + 15) & (o.y >= y0 - 15) & (o.y < y1 + 15)]
                sq[["id", "x", "y", "area_px", "call", "cells"]].round(1).to_csv(
                    vdir / f"review_s{rnd}" / f"{img}_r{r}c{c}_objects.csv", index=False
                )
        print(img, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(vdir / "summary.csv", index=False)
    if rnd:
        (vdir / f"review_s{rnd}" / "samples.json").write_text(json.dumps(samples, indent=1))
    print("total", int(pd.DataFrame(rows)["count"].sum()))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
    else:
        a = sys.argv
        rnd = int(a[a.index("--round") + 1]) if "--round" in a else None
        k = int(a[a.index("--per-image") + 1]) if "--per-image" in a else 3
        apply(a[2], a[3], a[4], rnd, k)
