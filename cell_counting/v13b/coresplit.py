# Splits objects whose dark cores are separated by a pale gap, on top of a v11 run. Basins are
# merged back when the darkness at their shared boundary stays close to the weaker peak. Run it as:
# python coresplit.py <in_run> <out_run> [ratio] [h].

import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from joblib import Parallel, delayed

FULL = Path(__import__("os").environ["CC_FULL"])


def darkness(im):
    g = im.astype(np.float32).mean(2)
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((31, 31), np.uint8)), (0, 0), 25)
    return cv2.GaussianBlur(np.clip(bg - g, 0, None), (0, 0), 1.5)


def split_image(src, dst, img, ratio, h):
    z = np.load(src / "labels" / f"{img}.npz")
    lab = z["labels"]
    win = z["window"]
    o = pd.read_csv(src / "objects" / f"{img}.csv").set_index("id")
    im = cv2.imread(str(FULL / f"{img}.jpg"))[win[1] : win[3], win[0] : win[2]]
    D = darkness(im)
    out = np.zeros_like(lab)
    nid = 0
    rows = []
    for p in regionprops(lab):
        sl = p.slice
        mk = p.image
        parent = o.loc[p.label] if p.label in o.index else None
        d = np.where(mk, D[sl], 0)
        pieces = None
        if p.area >= 120:
            mx = h_maxima(d, h) & mk
            mlab, n = ndi.label(mx)
            if n >= 2:
                ws = watershed(-d, mlab, mask=mk)
                peaks = ndi.maximum(d, ws, range(1, n + 1))

                par = list(range(n + 1))

                def f(a):
                    while par[a] != a:
                        a = par[a]
                    return a

                for a in range(1, n + 1):
                    ma = ws == a
                    ring = (
                        cv2.dilate(ma.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                        & ~ma
                        & mk
                    )
                    for b in np.unique(ws[ring]):
                        if b <= a:
                            continue
                        bd = ring & (ws == b)
                        sad = np.percentile(d[bd], 80) if bd.any() else 0
                        if sad >= ratio * min(peaks[a - 1], peaks[b - 1]):
                            ra, rb = f(a), f(b)
                            if ra != rb:
                                par[ra] = rb
                roots = np.array([f(a) for a in range(n + 1)])
                pieces = roots[ws]
                pieces[~mk] = 0

                ids, cnt = np.unique(pieces[pieces > 0], return_counts=True)
                for i, c in zip(ids, cnt):
                    if c < 40 and len(ids) > 1:
                        m_ = pieces == i
                        ring = (
                            cv2.dilate(m_.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                            & ~m_
                        )
                        nb = pieces[ring]
                        nb = nb[(nb > 0) & (nb != i)]
                        if nb.size:
                            pieces[m_] = np.bincount(nb).argmax()
                if len(np.unique(pieces[pieces > 0])) < 2:
                    pieces = None
        if pieces is None:
            nid += 1
            out[sl][mk] = nid
            rows.append(
                dict(
                    id=nid,
                    parent=p.label,
                    x=p.centroid[1] + win[0],
                    y=p.centroid[0] + win[1],
                    area_px=p.area,
                    counted=bool(parent.counted) if parent is not None else False,
                    n_split=1,
                )
            )
        else:
            us = np.unique(pieces[pieces > 0])
            for u in us:
                m_ = pieces == u
                nid += 1
                out[sl][m_] = nid
                yy, xx = np.nonzero(m_)
                rows.append(
                    dict(
                        id=nid,
                        parent=p.label,
                        x=xx.mean() + sl[1].start + win[0],
                        y=yy.mean() + sl[0].start + win[1],
                        area_px=int(m_.sum()),
                        counted=bool(parent.counted) if parent is not None else False,
                        n_split=len(us),
                    )
                )
    np.savez_compressed(dst / "labels" / f"{img}.npz", labels=out, window=win)
    pd.DataFrame(rows).to_csv(dst / "objects" / f"{img}.csv", index=False)
    return img, len(o), nid


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    h = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    for d in ("labels", "objects"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    imgs = sorted(f.stem for f in (src / "labels").glob("*.npz"))
    res = Parallel(n_jobs=12)(delayed(split_image)(src, dst, i, ratio, h) for i in imgs)
    print("objects before", sum(r[1] for r in res), "after", sum(r[2] for r in res))
