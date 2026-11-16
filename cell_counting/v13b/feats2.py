# Computes the shape, darkness, core, skeleton, neighbour-contact and stain-colour features of every
# object in a run. The features are cached per run in features2.csv.

from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import skeletonize

FULL = Path(__import__("os").environ["CC_FULL"])
BASE = [
    "area",
    "perimeter",
    "solidity",
    "eccentricity",
    "major",
    "minor",
    "extent",
    "convex_deficit",
]
FEATS2 = BASE + [
    "dark_mean",
    "dark_max",
    "dark_p90",
    "core50_n",
    "core65_n",
    "core80_n",
    "core50_frac",
    "skel_len",
    "skel_branch",
    "skel_end",
    "violet_mean",
    "violet_core",
    "green_frac",
    "nbr_n",
    "group_n",
    "group_area",
    "log_area",
    "core50_maxsep",
]


def maps(img):
    f = img.astype(np.float32)
    g = f.mean(2)
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((31, 31), np.uint8)), (0, 0), 25)
    dark = cv2.GaussianBlur(np.clip(bg - g, 0, None), (0, 0), 1.2)
    b, gg, r = f[..., 0], f[..., 1], f[..., 2]
    violet = cv2.GaussianBlur((r + b) / 2 - gg, (0, 0), 1.0)
    greenish = cv2.GaussianBlur(gg - (r + b) / 2, (0, 0), 1.0)
    return dark, violet, greenish


def features2(run, img):
    z = np.load(Path(run) / "labels" / f"{img}.npz")
    lab = z["labels"]
    win = z["window"]
    im = cv2.imread(str(FULL / f"{img}.jpg"))[win[1] : win[3], win[0] : win[2]]
    D, V, G = maps(im)

    dil = ndi.grey_dilation(lab, size=(7, 7))
    pairs = set()
    m = (dil != lab) & (lab > 0) & (dil > 0)
    for a_, b_ in zip(lab[m], dil[m]):
        if a_ != b_:
            pairs.add((min(a_, b_), max(a_, b_)))
    nbr = {}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for a_, b_ in pairs:
        nbr[a_] = nbr.get(a_, 0) + 1
        nbr[b_] = nbr.get(b_, 0) + 1
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    rows = []
    props = regionprops(lab)
    area_of = {p.label: p.area for p in props}
    groups = {}
    for p in props:
        groups.setdefault(find(p.label), []).append(p.label)
    gsize = {l: len(v) for v in groups.values() for l in v}
    garea = {l: sum(area_of[k] for k in v) for v in groups.values() for l in v}
    for p in props:
        sl = p.slice
        mk = p.image
        d = D[sl][mk]
        dm = float(d.max()) if d.size else 0.0
        row = dict(
            id=p.label,
            cx=p.centroid[1] + win[0],
            cy=p.centroid[0] + win[1],
            area=p.area,
            perimeter=p.perimeter,
            solidity=p.solidity,
            eccentricity=p.eccentricity,
            major=p.major_axis_length,
            minor=p.minor_axis_length,
            extent=p.extent,
            convex_deficit=p.convex_area - p.area,
            dark_mean=float(d.mean()),
            dark_max=dm,
            dark_p90=float(np.percentile(d, 90)),
        )
        Dl = np.where(mk, D[sl], 0)
        for t in (50, 65, 80):
            cm = Dl > dm * t / 100
            lbl, n = ndi.label(cm)
            if n:
                sz = ndi.sum(cm, lbl, range(1, n + 1))
                n = int((sz >= 6).sum())
            row[f"core{t}_n"] = n
            if t == 50:
                row["core50_frac"] = float(cm.sum() / max(p.area, 1))
                if n >= 2:
                    cs = ndi.center_of_mass(cm, lbl, [i + 1 for i in np.where(sz >= 6)[0]])
                    cs = np.array(cs)
                    row["core50_maxsep"] = float(
                        np.max(np.hypot(*(cs[:, None] - cs[None]).transpose(2, 0, 1)))
                    )
                else:
                    row["core50_maxsep"] = 0.0
        sk = skeletonize(mk)
        nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") * sk
        row.update(
            skel_len=int(sk.sum()), skel_branch=int((nb >= 4).sum()), skel_end=int((nb == 2).sum())
        )
        v = V[sl][mk]
        gr = G[sl][mk]
        row.update(
            violet_mean=float(v.mean()),
            violet_core=float(np.percentile(v, 90)),
            green_frac=float((gr > 4).mean()),
            nbr_n=nbr.get(p.label, 0),
            group_n=gsize[p.label],
            group_area=garea[p.label],
            log_area=float(np.log(p.area)),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def all_features2(run):
    cache = Path(run) / "features2.csv"
    if cache.exists():
        return pd.read_csv(cache)
    out = []
    for f in sorted((Path(run) / "labels").glob("*.npz")):
        F = features2(run, f.stem)
        F["image"] = f.stem
        out.append(F)
    F = pd.concat(out, ignore_index=True)
    F.to_csv(cache, index=False)
    return F


if __name__ == "__main__":
    import sys
    from joblib import Parallel, delayed

    run = sys.argv[1]
    imgs = sorted(f.stem for f in (Path(run) / "labels").glob("*.npz"))
    parts = Parallel(n_jobs=12)(delayed(features2)(run, i) for i in imgs)
    for i, F in zip(imgs, parts):
        F["image"] = i
    pd.concat(parts, ignore_index=True).to_csv(Path(run) / "features2.csv", index=False)
    print("done", sum(len(p) for p in parts))
