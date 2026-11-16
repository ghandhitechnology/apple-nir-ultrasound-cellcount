# Retrains the two learned split and merge models of the v11 counter from reviewer point
# annotations. It reports apple-grouped cross-validated AUCs and writes the models as small numpy
# files. Only training needs scikit-learn; counting does not.

import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cellcount import core

WIN0 = (core.WIN[0], core.WIN[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reviewers", default="astra,opus")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    bench = Path(a.bench)
    rd = bench / a.run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bench / "bench"))
    from score import load_gt

    split = json.loads((bench / "gt/split.json").read_text())["tune"]
    squares = json.loads((bench / "gt/squares.json").read_text())["squares"]
    gts = [load_gt(r) for r in a.reviewers.split(",")]
    rows = {"split": [], "merge": []}
    for im in split:
        lab = np.load(rd / f"{im}.npz")["labels"].astype(np.int32)[::2, ::2]
        img = cv2.imread(str(bench / "images" / f"{im}.jpg"), cv2.IMREAD_REDUCED_COLOR_2)
        img = img[
            WIN0[1] // 2 : WIN0[1] // 2 + lab.shape[0], WIN0[0] // 2 : WIN0[0] // 2 + lab.shape[1]
        ]
        dark = core.darkness_half(img)
        st = core._stats(lab, dark)
        cands, _ = core.split_cands(lab, dark, st)
        P, PF = core.pair_feats(lab, dark, st)
        band = np.zeros(lab.shape, bool)
        for sq in squares[im]:
            b = [
                int((sq["band"][0] - WIN0[0]) / 2),
                int((sq["band"][1] - WIN0[1]) / 2),
                int((sq["band"][2] - WIN0[0]) / 2),
                int((sq["band"][3] - WIN0[1]) / 2),
            ]
            band[max(0, b[1]) : b[3], max(0, b[0]) : b[2]] = True
        cy = np.clip(st["cy"].round().astype(int), 0, lab.shape[0] - 1)
        cx = np.clip(st["cx"].round().astype(int), 0, lab.shape[1] - 1)
        inb = band[cy, cx] & (st["cnt"] > 0)
        for G in gts:
            own = {}
            for sq in squares[im]:
                for p in G.get((im, sq["square"]), []):
                    lx, ly = int(round((p[0] - WIN0[0]) / 2)), int(round((p[1] - WIN0[1]) / 2))
                    if 0 <= ly < lab.shape[0] and 0 <= lx < lab.shape[1] and lab[ly, lx] > 0:
                        own.setdefault(int(lab[ly, lx]), []).append((lx, ly))
            for o, s, part, f in cands:
                if not inb[o]:
                    continue
                pts = own.get(o, [])
                if len(pts) >= 2:
                    which = {int(part[y - s[0].start, x - s[1].start]) for x, y in pts}
                    rows["split"].append((im, f, 1 if {1, 2} <= which else 0))
                elif len(pts) == 1:
                    rows["split"].append((im, f, 0))
            for (p1, p2), f in zip(P, PF):
                if not (inb[p1] and inb[p2]):
                    continue
                n1, n2 = len(own.get(int(p1), [])), len(own.get(int(p2), []))
                if n1 >= 1 and n2 >= 1:
                    rows["merge"].append((im, f, 0))
                elif (n1 >= 1) != (n2 >= 1):
                    rows["merge"].append((im, f, 1))
    apples = sorted({i.split("-")[0] for i in split})
    folds = [apples[k::5] for k in range(5)]
    for name, R in rows.items():
        X = np.array([r[1] for r in R])
        y = np.array([r[2] for r in R])
        ims = np.array([r[0] for r in R])
        aucs = []
        for fa in folds:
            te = np.array([i.split("-")[0] in fa for i in ims])
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
                (X[tr] - mu) / sd, y[tr]
            )
            if te.any() and len(set(y[te])) > 1:
                aucs.append(roc_auc_score(y[te], clf.decision_function((X[te] - mu) / sd)))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
            (X - mu) / sd, y
        )
        np.savez(
            out / f"{name}_all.npz",
            mu=mu.astype(np.float32),
            sd=sd.astype(np.float32),
            w=clf.coef_[0].astype(np.float32),
            b=np.float32(clf.intercept_[0]),
        )
        print(
            f'{name}: {len(y)} samples, {int(y.sum())} positive, apple-grouped CV AUC {np.mean(aucs):.3f} (folds {", ".join(f"{v:.2f}" for v in aucs)})'
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
