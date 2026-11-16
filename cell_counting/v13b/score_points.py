# Turns reviewer points into training targets and per-square scores for a run. Each reviewer cell or
# clump point is assigned to the object that contains it, and each counting square gets reviewer
# counts, counter counts and a point-level F1.

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(r"D:\cellcount_v11_full")
P, GX, GY, W, H = 274, 2, -16, 3840, 2160


def sq_box(name):
    r, c = name[1:].split("c")
    r, c = int(r), int(c)
    x0 = GX + P * (c - 1)
    y0 = GY + P * (r - 1)
    return max(x0, 0), max(y0, 0), min(x0 + P, W), min(y0 + P, H)


def load_points(rnd):
    rows = []
    bad = []
    for rv in ("sol", "opus"):
        for f in (ROOT / "reviews" / f"s{rnd}" / rv).glob("batch_*/points/*.csv"):
            stem = f.stem
            img, sq = stem.rsplit("_", 1)
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if d.empty:
                rows.append(
                    dict(reviewer=rv, image=img, square=sq, x=np.nan, y=np.nan, type="none")
                )
                continue
            d.columns = [c.strip().lower() for c in d.columns]
            for _, q in d.iterrows():
                t = str(q.get("type", "cell")).strip().lower()
                try:
                    x, y = float(q["x"]), float(q["y"])
                except (TypeError, ValueError):
                    bad.append(str(f))
                    continue
                rows.append(
                    dict(
                        reviewer=rv,
                        image=img,
                        square=sq,
                        x=x,
                        y=y,
                        type="clump" if "clump" in t else "cell",
                    )
                )
    if bad:
        print("skipped malformed rows in", sorted(set(bad)))
    return pd.DataFrame(rows)


def assign(run, pts):
    out = []
    for img, g in pts.groupby("image"):
        z = np.load(Path(run) / "labels" / f"{img}.npz")
        lab = z["labels"]
        win = z["window"]
        o = pd.read_csv(Path(run) / "objects" / f"{img}.csv")
        tree = cKDTree(o[["x", "y"]].to_numpy())
        g = g.copy()
        ok = g.x.notna()
        xi = np.clip(np.round(g.x[ok]).astype(int) - win[0], 0, lab.shape[1] - 1)
        yi = np.clip(np.round(g.y[ok]).astype(int) - win[1], 0, lab.shape[0] - 1)
        oid = lab[yi, xi].astype(int)
        d, j = tree.query(g.loc[ok, ["x", "y"]].to_numpy())
        near = np.where(d <= 6, o.id.to_numpy()[j], 0)
        g["obj"] = 0
        g.loc[ok, "obj"] = np.where(oid > 0, oid, near)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def object_targets(run, A):
    rows = []
    for (rv, img, sq), g in A.groupby(["reviewer", "image", "square"]):
        o = pd.read_csv(Path(run) / "objects" / f"{img}.csv")
        x0, y0, x1, y1 = sq_box(sq)
        inside = o[(o.x >= x0) & (o.x < x1) & (o.y >= y0) & (o.y < y1) & o.counted]
        cells = g[g.type == "cell"].groupby("obj").size()
        clumps = g[g.type == "clump"].groupby("obj").size()
        for _, q in inside.iterrows():
            rows.append(
                dict(
                    reviewer=rv,
                    image=img,
                    square=sq,
                    id=int(q.id),
                    n_cells=int(cells.get(q.id, 0)),
                    n_clump=int(clumps.get(q.id, 0)),
                )
            )
    return pd.DataFrame(rows)


def square_scores(A, contrib_by_img, run):
    rows = []
    for (rv, img, sq), g in A.groupby(["reviewer", "image", "square"]):
        o = pd.read_csv(Path(run) / "objects" / f"{img}.csv")
        x0, y0, x1, y1 = sq_box(sq)
        inside = o[(o.x >= x0) & (o.x < x1) & (o.y >= y0) & (o.y < y1) & o.counted]
        c = contrib_by_img[img].reindex(inside.id).fillna(0)
        rows.append(
            dict(
                reviewer=rv,
                image=img,
                square=sq,
                reviewer_cells=int((g.type == "cell").sum()),
                counter_cells=float(c.sum()),
            )
        )
    S = pd.DataFrame(rows)
    S["err"] = S.counter_cells - S.reviewer_cells
    return S


if __name__ == "__main__":
    run = sys.argv[1]
    rnd = int(sys.argv[2])
    pts = load_points(rnd)
    print("points", len(pts), pts.groupby(["reviewer", "type"]).size().to_dict())
    A = assign(run, pts)
    print(
        "cell points with no counter object:",
        int(((A.type == "cell") & (A.obj == 0)).sum()),
        "of",
        int((A.type == "cell").sum()),
    )
    T = object_targets(run, A)
    T.to_csv(ROOT / "learn" / f"targets_s{rnd}.csv", index=False)
    A.to_csv(ROOT / "learn" / f"points_s{rnd}.csv", index=False)
    print(T.groupby("reviewer").n_cells.value_counts().unstack().fillna(0))
    if len(sys.argv) > 3:
        V = Path(sys.argv[3])
        contrib = {}
        for img in A.image.unique():
            ob = pd.read_csv(V / "objects" / f"{img}.csv")
            contrib[img] = pd.Series((ob.status == "counted").astype(float).to_numpy(), index=ob.id)
        S = square_scores(A, contrib, run)
        for rv, s in S.groupby("reviewer"):
            print(
                rv,
                "squares",
                len(s),
                "sum reviewer",
                s.reviewer_cells.sum(),
                "sum counter",
                s.counter_cells.sum(),
                "MAE/sq",
                round(s.err.abs().mean(), 2),
                "rel abs err/sq",
                round((s.err.abs() / s.reviewer_cells.clip(lower=1)).mean(), 3),
            )
