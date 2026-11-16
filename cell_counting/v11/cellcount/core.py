# Segmentation core of the v11 cell counter for stained apple-flesh photos. It estimates the
# background, scores each pixel for stain, splits touching bodies by watershed, rejects faint rings,
# haze, defocused discs and optical ghosts, and adds faint bodies. Two small learned models then
# split masks that hold two bodies and merge pieces of one body.

import atexit
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

SQUARE = (1372, 532, 2468, 1628)
MARGIN = 64
WIN = (SQUARE[0] - MARGIN, SQUARE[1] - MARGIN, SQUARE[2] + MARGIN, SQUARE[3] + MARGIN)


DEFAULT = dict(
    grow=6.0,
    seed=9.0,
    waist=0.9,
    hmax_h=3.0,
    contrast=6.0,
    min_area=8,
    ghost=True,
    ring_core=0.18,
    disc_minor=35.0,
    disc_ecc=0.85,
    disc_edge=0.075,
    disc_tex=0.75,
    haze_edge=0.35,
    haze_sd=1.5,
    faint_T=3.0,
    faint_s1=2.0,
    faint_s2=6.0,
    faint_gap=20.0,
    faint_r=5.0,
    pp=True,
    pp_tm=0.9,
    pp_ts=0.7,
    half_decode=True,
    scale=1,
    bg_scale=16,
    hmax_ds=2,
    hmax_q=True,
    lowres_ref=True,
    seed_sigma=3.0,
    pad=64,
    workers=8,
    grid_rows=2,
    grid_cols=4,
    overlap=48,
    threads=8,
)
CFG = dict(DEFAULT)
TIMES = {}
_TP = None


def configure(cfg=None):
    CFG.clear()
    CFG.update(DEFAULT)
    if cfg:
        unknown = set(cfg) - set(DEFAULT)
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        CFG.update(cfg)
    if CFG["half_decode"]:
        CFG["scale"] = 2
    if use_pool():
        _start_pool()
    cv2.setNumThreads(int(CFG["threads"]))


def use_pool():
    return int(CFG["workers"]) > 1 and hasattr(os, "fork")


def _tp():
    global _TP
    if _TP is None:
        _TP = ThreadPoolExecutor(3)
    return _TP


def _tick(name, t):
    TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t
    return time.perf_counter()


def _percentile_filter(a, pct, size):
    r = size // 2
    p = np.pad(a, r, mode="symmetric")
    win = sliding_window_view(p, (size, size)).reshape(a.shape[0], a.shape[1], size * size)
    k = int(float(size * size) * pct / 100.0)
    return np.partition(win, k, axis=-1)[..., k]


def background(im, box, ps=1):
    fp = int(CFG["bg_scale"])
    win = {8: 41, 16: 21, 32: 11}[fp]
    f = fp // ps
    h, w = im.shape[:2]
    sw, sh = math.ceil(w / f), math.ceil(h / f)
    x0, y0, x1, y1 = box
    g = int(math.ceil(3 * 35 / fp)) + 2
    ox0, oy0 = max(0, x0 // f - g), max(0, y0 // f - g)
    ox1, oy1 = min(sw, x1 // f + 2 + g), min(sh, y1 // f + 2 + g)
    hw = win // 2 + 1
    ix0, iy0, ix1, iy1 = max(0, ox0 - hw), max(0, oy0 - hw), min(sw, ox1 + hw), min(sh, oy1 + hw)
    if ix0 > 0 and iy0 > 0 and ix1 < sw and iy1 < sh:
        region = im[iy0 * f : iy1 * f, ix0 * f : ix1 * f]
        q = 4 if f >= 8 else 2
        sub = cv2.resize(
            region, (region.shape[1] // q, region.shape[0] // q), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        sub = cv2.resize(sub, (ix1 - ix0, iy1 - iy0), interpolation=cv2.INTER_AREA)
        env = np.dstack(
            list(
                _tp().map(
                    lambda c: _percentile_filter(np.ascontiguousarray(sub[..., c]), 70, win),
                    range(3),
                )
            )
        )
    else:
        full = cv2.resize(im.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        env = np.dstack(
            [_percentile_filter(np.ascontiguousarray(full[..., c]), 70, win) for c in range(3)]
        )[iy0:iy1, ix0:ix1]
    env = cv2.GaussianBlur(np.ascontiguousarray(env, np.float32), (0, 0), 35 / fp)
    xs = ((np.arange(x0, x1, dtype=np.float64) + 0.5) * (sw / w) - 0.5 - ix0).astype(np.float32)
    ys = ((np.arange(y0, y1, dtype=np.float64) + 0.5) * (sh / h) - 0.5 - iy0).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(env, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _blur_ds(a, sigma, f):
    h, w = a.shape[:2]
    s = cv2.resize(a, (math.ceil(w / f), math.ceil(h / f)), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), sigma / f)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def _fill_holes(mask):
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    ff = pad.copy()
    fm = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, fm, (0, 0), 1, flags=4)
    return ((pad > 0) | (ff == 0))[1:-1, 1:-1].astype(np.uint8)


def segment(tile, bg, contrast, s):
    t = time.perf_counter()
    lab = cv2.cvtColor(tile * (1 / 255.0), cv2.COLOR_BGR2LAB)
    hh, ww = bg.shape[:2]
    if CFG["lowres_ref"]:
        small = cv2.resize(bg, (math.ceil(ww / 8), math.ceil(hh / 8)), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(
            cv2.cvtColor(np.clip(small * (1 / 255.0), 0, 1), cv2.COLOR_BGR2LAB),
            (ww, hh),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        ref = cv2.cvtColor(np.clip(bg * (1 / 255.0), 0, 1), cv2.COLOR_BGR2LAB)
    residual = lab[:, :, 1:] - ref[:, :, 1:]
    fds = max(1, 4 // s)
    residual -= 0.85 * _blur_ds(residual, 25 / s, fds)
    chroma = cv2.magnitude(
        np.ascontiguousarray(residual[..., 0]), np.ascontiguousarray(residual[..., 1])
    )
    local_dark = contrast - 0.85 * _blur_ds(np.maximum(contrast, 0), 25 / s, fds)
    score = cv2.GaussianBlur(np.maximum(np.maximum(local_dark, 0), chroma * 1.8), (0, 0), 1.2 / s)
    t = _tick("score", t)
    support = _fill_holes((score >= CFG["grow"]).astype(np.uint8))
    k = 3 if s == 1 else 2
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    dist = cv2.distanceTransform(support, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    smooth = cv2.GaussianBlur(dist, (0, 0), 2 / s)
    seed_map = cv2.GaussianBlur(score, (0, 0), CFG["seed_sigma"] / s)
    t = _tick("support", t)
    d = int(CFG["hmax_ds"]) if s == 1 else 1
    sm = seed_map
    if d > 1:
        hs, ws = seed_map.shape
        sm = cv2.resize(
            seed_map, (math.ceil(ws / d), math.ceil(hs / d)), interpolation=cv2.INTER_AREA
        )
    if CFG["hmax_q"]:
        hm = h_maxima(
            np.round(np.clip(sm, 0, 16000) * 4).astype(np.uint16), int(round(CFG["hmax_h"] * 4))
        )
    else:
        hm = h_maxima(sm, CFG["hmax_h"])
    if d > 1:
        hm = (
            cv2.resize(
                hm.astype(np.uint8),
                (seed_map.shape[1], seed_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        hm &= seed_map >= cv2.dilate(seed_map, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) - 1e-4
    else:
        hm = hm > 0
    maxima = hm & (support > 0) & (dist >= 2 / s) & (score >= CFG["seed"])
    t = _tick("hmax", t)
    n, markers = cv2.connectedComponents(maxima.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    n -= 1
    nc, comps = cv2.connectedComponents(support, connectivity=4, ltype=cv2.CV_32S)
    nc -= 1
    has = np.zeros(nc + 1, bool)
    has[comps[markers > 0]] = True
    missing = ~has
    missing[0] = False
    if missing.any():
        ys, xs = np.nonzero(missing[comps])
        cid = comps[ys, xs]
        order = np.lexsort((smooth[ys, xs], cid))
        cid_s = cid[order]
        last = np.ones(len(order), bool)
        last[:-1] = cid_s[1:] != cid_s[:-1]
        pick = order[last]
        markers[ys[pick], xs[pick]] = n + np.arange(1, len(pick) + 1)
    t = _tick("markers", t)
    split = watershed(-smooth, markers, mask=support > 0)
    t = _tick("watershed", t)
    n = int(split.max())
    peaks = np.zeros(n + 1)
    m = split > 0
    np.maximum.at(peaks, split[m], dist[m])
    pairs = []
    for a, b, da, db in (
        (split[:-1], split[1:], dist[:-1], dist[1:]),
        (split[:, :-1], split[:, 1:], dist[:, :-1], dist[:, 1:]),
    ):
        hit = (a != b) & (a > 0) & (b > 0)
        u, v, dd = a[hit].astype(np.int64), b[hit].astype(np.int64), np.minimum(da[hit], db[hit])
        pairs.append(np.stack([np.minimum(u, v), np.maximum(u, v), dd], 1))
    parent = np.arange(n + 1)

    def owner(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    P = np.concatenate(pairs) if pairs else np.zeros((0, 3))
    if len(P):
        key = P[:, 0].astype(np.int64) * (n + 1) + P[:, 1].astype(np.int64)
        order = np.lexsort((-P[:, 2], key))
        key, P = key[order], P[order]
        first = np.ones(len(key), bool)
        first[1:] = key[1:] != key[:-1]
        S = P[first]
        for a, b, sdl in S[np.argsort(-S[:, 2], kind="stable")]:
            a, b = owner(int(a)), owner(int(b))
            if a != b and sdl >= CFG["waist"] * min(peaks[a], peaks[b]):
                parent[b] = a
                peaks[a] = max(peaks[a], peaks[b])
    lookup = np.array([owner(i) for i in range(n + 1)])
    _tick("merge", t)
    return lookup[split], score


def _group_pct(L, v, n, q):
    if not len(L):
        return np.full(n + 1, np.nan)
    v = v.astype(np.float64)
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo + 1.0
    sv = np.sort(L * span + (v - lo))
    cnt = np.bincount(L, minlength=n + 1)
    ends = np.cumsum(cnt)
    starts = ends - cnt
    out = np.full(n + 1, np.nan)
    ok = cnt > 0
    pos = starts[ok] + (cnt[ok] - 1) * (q / 100.0)
    a = np.floor(pos).astype(np.int64)
    b = np.minimum(a + 1, ends[ok] - 1)
    fr = pos - a
    La = np.nonzero(ok)[0].astype(np.float64)
    out[ok] = (sv[a] - La * span + lo) * (1 - fr) + (sv[b] - La * span + lo) * fr
    return out


def measure_and_filter(labels, field, score, s):
    t = time.perf_counter()
    lab = labels.astype(np.int32)
    n = int(lab.max())
    if n == 0:
        return lab, dict(ids=np.zeros(0, int), C=np.zeros(0), x=np.zeros(0), y=np.zeros(0))
    labf = lab.astype(np.float32)
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5) if s == 1 else (3, 3), np.uint8)
    e3 = cv2.erode(labf, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    d3 = cv2.dilate(labf, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    e5 = cv2.erode(labf, k5, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    d5 = cv2.dilate(labf, k5, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3) * (1 / (8 * s))
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3) * (1 / (8 * s))
    grad = cv2.magnitude(gx, gy)
    detail = field - cv2.GaussianBlur(field, (0, 0), 4 / s)
    ys, xs = np.nonzero(lab)
    L = lab[ys, xs].astype(np.int64)
    fv, sv, gv, dv = field[ys, xs], score[ys, xs], grad[ys, xs], detail[ys, xs]
    lv = labf[ys, xs]
    interior3 = (e3[ys, xs] == lv) & (d3[ys, xs] == lv)
    inner5 = (e5[ys, xs] == lv) & (d5[ys, xs] == lv)
    bnd = ~interior3
    area = np.bincount(L, minlength=n + 1).astype(np.float64) * (s * s)
    in_cnt = np.bincount(L[inner5], minlength=n + 1)
    C = _group_pct(L, fv, n, 90)
    S = _group_pct(L, sv, n, 90)
    edge = _group_pct(L[bnd], gv[bnd], n, 75)
    core = np.where(in_cnt > 0, _group_pct(L[inner5], fv[inner5], n, 50), _group_pct(L, fv, n, 50))

    def gstd(Lg, v):
        c = np.bincount(Lg, minlength=n + 1).astype(np.float64)
        s1 = np.bincount(Lg, weights=v, minlength=n + 1)
        s2 = np.bincount(Lg, weights=v.astype(np.float64) ** 2, minlength=n + 1)
        mu = s1 / np.maximum(c, 1)
        return np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mu * mu, 0))

    sd = np.where(in_cnt > 0, gstd(L[inner5], fv[inner5]), gstd(L, fv))
    tex = gstd(L, dv)
    xf, yf = xs.astype(np.float64) * s, ys.astype(np.float64) * s
    cnt = np.maximum(np.bincount(L, minlength=n + 1).astype(np.float64), 1)
    mx = np.bincount(L, xf, n + 1) / cnt
    my = np.bincount(L, yf, n + 1) / cnt
    cxx = np.bincount(L, xf * xf, n + 1) / cnt - mx * mx
    cyy = np.bincount(L, yf * yf, n + 1) / cnt - my * my
    cxy = np.bincount(L, xf * yf, n + 1) / cnt - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    minor = 4 * np.sqrt(l2)
    ecc = np.sqrt(1 - l2 / np.maximum(l1, 1e-9))
    diam = np.sqrt(4 * area / math.pi)
    with np.errstate(invalid="ignore"):
        reject = area < CFG["min_area"]
        reject |= (diam > 12) & (core < np.maximum(2.0, CFG["ring_core"] * C)) & (C < 14) & (S < 18)
        reject |= (
            (minor > CFG["disc_minor"])
            & (ecc < CFG["disc_ecc"])
            & (edge / np.maximum(C, 1) < CFG["disc_edge"])
            & (tex < CFG["disc_tex"])
        )
        reject |= (edge < CFG["haze_edge"]) & (sd < CFG["haze_sd"])
        reject |= S < CFG["seed"]
        reject |= C < CFG["contrast"]
    reject[0] = True
    reject |= np.bincount(L, minlength=n + 1) == 0
    keep = np.where(reject, 0, np.arange(n + 1)).astype(np.int32)
    ids = np.nonzero(~reject)[0]
    _tick("measure", t)
    return keep[lab], dict(ids=ids, C=C[ids], x=mx[ids] / s, y=my[ids] / s)


def ghost_screen(lab, info, raw_hp, s):
    t = time.perf_counter()
    ids, C = info["ids"], info["C"]
    if not len(ids):
        return lab
    Cmap = dict(zip(ids.tolist(), C.tolist()))
    h, w = raw_hp.shape
    T = 15 // s
    sy = 23 // s
    sx0, sx1 = 30 // s, 81 // s
    off = 45 // s
    dy0 = 8 // s
    rad = 10 // s
    drop = []
    for oid, x, y in zip(ids.tolist(), info["x"], info["y"]):
        x, y = int(round(x)), int(round(y))
        if x < T + 1 or y < sy + 1 or x + sx1 >= w or y + sy + 1 >= h:
            continue
        tpl = raw_hp[y - T : y + T + 1, x - T : x + T + 1]
        if tpl.std() < 0.15:
            continue
        r = cv2.matchTemplate(
            raw_hp[y - sy : y + sy + 1, x + sx0 : x + sx1], tpl, cv2.TM_CCOEFF_NORMED
        )
        _, sc, _, loc = cv2.minMaxLoc(r)
        if sc < 0.75:
            continue
        px, py = x + off + loc[0], y + loc[1] - dy0
        win = lab[max(0, py - rad) : py + rad + 1, max(0, px - rad) : px + rad + 1]
        if any(
            o > 0 and o != oid and Cmap.get(int(o), 0) >= 1.7 * Cmap[oid] for o in np.unique(win)
        ):
            drop.append(oid)
    if drop:
        lut = np.arange(int(lab.max()) + 1, dtype=np.int32)
        lut[drop] = 0
        lab = lut[lab]
    _tick("ghost", t)
    return lab


def add_faint(lab, contrast, s):
    T = float(CFG["faint_T"])
    if T <= 0:
        return lab
    t = time.perf_counter()
    dog = cv2.GaussianBlur(contrast, (0, 0), CFG["faint_s1"] / s) - cv2.GaussianBlur(
        contrast, (0, 0), CFG["faint_s2"] / s
    )
    k = max(3, int(round(2 * CFG["faint_s2"] / s)) | 1)
    peak = (dog >= T) & (dog >= cv2.dilate(dog, np.ones((k, k), np.uint8)) - 1e-6)
    free = cv2.distanceTransform((lab == 0).astype(np.uint8), cv2.DIST_L2, 3) > CFG["faint_gap"] / s
    ys, xs = np.nonzero(peak & free)
    if len(ys):
        nxt = int(lab.max()) + 1
        r = max(1, int(round(CFG["faint_r"] / s)))
        lab = lab.astype(np.int32).copy()
        for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
            cv2.circle(lab, (x, y), r, nxt + i, -1)
    _tick("faint", t)
    return lab


GAP = 2
SPLIT_MIN = 150
_PP = {}


def darkness_half(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.resize(
        cv2.resize(g, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA),
        (g.shape[1], g.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    bg = cv2.GaussianBlur(bg, (0, 0), 4)
    return cv2.GaussianBlur(bg - g, (0, 0), 0.75)


def _stats(lab, dark):
    n = int(lab.max())
    ys, xs = np.nonzero(lab)
    L = lab[ys, xs].astype(np.int64)
    cnt = np.bincount(L, minlength=n + 1).astype(np.float64)
    c = np.maximum(cnt, 1)
    s1 = np.bincount(L, dark[ys, xs], n + 1)
    s2 = np.bincount(L, dark[ys, xs].astype(np.float64) ** 2, n + 1)
    mu = s1 / c
    sd = np.sqrt(np.maximum(s2 / c - mu * mu, 0))
    cx = np.bincount(L, xs, n + 1) / c
    cy = np.bincount(L, ys, n + 1) / c
    mx = np.full(n + 1, -1e9)
    np.maximum.at(mx, L, dark[ys, xs])
    return dict(n=n, cnt=cnt, mu=mu, sd=sd, cx=cx, cy=cy, mx=mx)


def pair_feats(lab, dark, st):
    labf = lab.astype(np.float32)
    k = np.ones((2 * GAP + 1, 2 * GAP + 1), np.uint8)
    dil = cv2.dilate(labf, k)
    big = np.where(lab > 0, labf, 1e9).astype(np.float32)
    ero = cv2.erode(big, k)
    ero[ero >= 1e9] = 0
    m = (dil > 0) & (ero > 0) & (dil != ero)
    a = np.minimum(dil[m], ero[m]).astype(np.int64)
    b = np.maximum(dil[m], ero[m]).astype(np.int64)
    if not len(a):
        return np.zeros((0, 2), int), np.zeros((0, 9), np.float32)
    n = st["n"] + 1
    key = a * n + b
    uk, inv, cnts = np.unique(key, return_inverse=True, return_counts=True)
    zd = np.bincount(inv, dark[m], len(uk)) / cnts
    zero_zone = np.bincount(inv, (lab[m] == 0).astype(np.float64), len(uk))
    pa, pb = uk // n, uk % n
    ca, cb = st["cnt"][pa], st["cnt"][pb]
    mn = np.minimum(ca, cb)
    da, db = st["mu"][pa], st["mu"][pb]
    cd = np.hypot(st["cx"][pa] - st["cx"][pb], st["cy"][pa] - st["cy"][pb])
    F = np.stack(
        [
            np.log1p(ca),
            np.log1p(cb),
            np.log(np.maximum(ca, cb) / np.maximum(1, mn)),
            cnts / np.sqrt(mn),
            cd / np.sqrt(ca + cb),
            (np.minimum(da, db) - zd) / np.maximum(1, np.minimum(da, db)),
            zd / 20.0,
            np.minimum(da, db) / 20.0,
            zero_zone / np.sqrt(mn),
        ],
        1,
    ).astype(np.float32)
    return np.stack([pa, pb], 1), F


def split_cands(lab, dark, st):
    ds = cv2.GaussianBlur(dark, (0, 0), 1.0)
    lm = (ds >= cv2.dilate(ds, np.ones((5, 5), np.uint8)) - 1e-6) & (lab > 0)
    ys, xs = np.nonzero(lm)
    L = lab[ys, xs]
    out = []
    if not len(L):
        return out, ds
    big = st["cnt"][L] >= SPLIT_MIN
    ys, xs, L = ys[big], xs[big], L[big]
    order = np.lexsort((-ds[ys, xs], L))
    ys, xs, L = ys[order], xs[order], L[order]
    first = np.r_[True, L[1:] != L[:-1]]
    starts = np.nonzero(first)[0]
    ends = np.r_[starts[1:], len(L)]
    sl = ndi.find_objects(lab)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < 2:
            continue
        a = int(L[s0])
        s = sl[a - 1]
        if s is None:
            continue
        m = lab[s] == a
        mk = np.zeros(m.shape, np.int32)
        p1 = (ys[s0] - s[0].start, xs[s0] - s[1].start)
        p2 = (ys[s0 + 1] - s[0].start, xs[s0 + 1] - s[1].start)
        mk[p1] = 1
        mk[p2] = 2
        sub = ds[s]
        part = watershed(-sub, mk, mask=m)
        n1, n2 = int((part == 1).sum()), int((part == 2).sum())
        if min(n1, n2) < 10:
            continue
        bd = (part == 1) & (cv2.dilate((part == 2).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        sad = float(sub[bd].max()) if bd.any() else float(min(sub[p1], sub[p2]))
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        neck = float(dist[bd].max()) if bd.any() else 0.0
        pk = sorted([float(sub[p1]), float(sub[p2])])
        A = float(m.sum())
        sep = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        f = [
            math.log1p(A),
            math.log(max(n1, n2) / min(n1, n2)),
            (pk[0] - sad) / max(1.0, abs(pk[0])),
            pk[0] / 20.0,
            (pk[1] - pk[0]) / 20.0,
            neck / max(1.0, math.sqrt(min(n1, n2) / math.pi)),
            sep / math.sqrt(A),
            float(e0 - s0),
        ]
        out.append((a, s, part, np.array(f, np.float32)))
    return out, ds


def logit(m, X):
    return 1 / (1 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


def pp_apply(lab, dark, models, tm, ts):
    lab = lab.astype(np.int32).copy()
    if "split" in models and ts < 1:
        st = _stats(lab, dark)
        cands, _ = split_cands(lab, dark, st)
        nxt = int(lab.max()) + 1
        for a, s, part, f in cands:
            if logit(models["split"], f[None])[0] >= ts:
                sub = lab[s]
                sub[(part == 2) & (sub == a)] = nxt
                nxt += 1
    if "merge" in models and tm < 1:
        st = _stats(lab, dark)
        P, F = pair_feats(lab, dark, st)
        if len(P):
            p = logit(models["merge"], F)
            parent = np.arange(int(lab.max()) + 1)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in np.argsort(-p):
                if p[i] < tm:
                    break
                ra, rb = find(int(P[i, 0])), find(int(P[i, 1]))
                if ra != rb:
                    parent[rb] = ra
            lab = np.array([find(i) for i in range(len(parent))], np.int32)[lab]
    return lab


def _pp_models():
    if not _PP:
        d = Path(__file__).resolve().parent / "models"
        for n in ("split", "merge"):
            _PP[n] = {k: v for k, v in np.load(d / f"{n}_all.npz").items()}
    return _PP


def process_arrays(tile, bg, pre=False, dark=None):
    t = time.perf_counter()
    s = int(CFG["scale"])
    if s > 1 and not pre:
        hh, ww = tile.shape[:2]
        tile = cv2.resize(tile, (ww // s, hh // s), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(bg, (ww // s, hh // s), interpolation=cv2.INTER_AREA)
    third = np.full((1, 3), 1 / 3, np.float32)

    def mean3(a):
        r = cv2.transform(a, third)
        return r[..., 0] if r.ndim == 3 else r

    raw = mean3(tile)
    contrast = cv2.GaussianBlur(mean3(bg) - raw, (0, 0), 1.5 / s)
    _tick("contrast", t)
    labels, score = segment(tile, bg, contrast, s)
    lab, info = measure_and_filter(labels, contrast, score, s)
    if CFG["ghost"]:
        hp = raw - cv2.GaussianBlur(raw, (0, 0), 12 / s)
        lab = ghost_screen(lab, info, hp, s)
    lab = add_faint(lab.astype(np.int32), contrast, s)
    if CFG["pp"] and s == 2:
        t2 = time.perf_counter()
        lab = pp_apply(
            lab,
            darkness_half(tile) if dark is None else dark,
            _pp_models(),
            CFG["pp_tm"],
            CFG["pp_ts"],
        )
        _tick("pp", t2)
    if s > 1 and not pre:
        lab = cv2.resize(
            lab.astype(np.float32),
            (lab.shape[1] * s, lab.shape[0] * s),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    return lab


_SHM = {}
_OWNED = []
_POOLP = None
_GEOM = None


def _attach(name, shape, dtype):
    from multiprocessing import shared_memory

    if name not in _SHM:
        _SHM[name] = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, dtype, buffer=_SHM[name].buf)


def _worker(args):
    k, core, proc, box, names, shapes, win, pre = args
    im = _attach(names[0], shapes[0], np.uint8)
    bgf = _attach(names[1], shapes[1], np.float32)
    out = _attach(names[2], shapes[2], np.int32)
    tile = im[proc[1] : proc[3], proc[0] : proc[2]].astype(np.float32)
    bg = bgf[proc[1] - box[1] : proc[3] - box[1], proc[0] - box[0] : proc[2] - box[0]].copy()
    dk = None
    if len(names) > 3:
        dkf = _attach(names[3], shapes[1][:2], np.float32)
        dk = dkf[proc[1] - box[1] : proc[3] - box[1], proc[0] - box[0] : proc[2] - box[0]].copy()
    lab = process_arrays(tile, bg, pre, dk)
    ys, xs = np.nonzero(lab)
    if not len(ys):
        return 0
    L = lab[ys, xs]
    cnt = np.bincount(L).astype(np.float64)
    cx = np.bincount(L, xs, len(cnt)) / np.maximum(cnt, 1) + proc[0]
    cy = np.bincount(L, ys, len(cnt)) / np.maximum(cnt, 1) + proc[1]
    keep = (cnt > 0) & (cx >= core[0]) & (cx < core[2]) & (cy >= core[1]) & (cy < core[3])
    keep[0] = False
    sel = keep[L]
    gx, gy = xs[sel] + proc[0] - win[0], ys[sel] + proc[1] - win[1]
    ok = (gx >= 0) & (gx < shapes[2][1]) & (gy >= 0) & (gy < shapes[2][0])
    out[gy[ok], gx[ok]] = L[sel][ok] + (k << 20)
    return int(keep.sum())


def _tiles(box, win, al):
    R, Cn, ov = int(CFG["grid_rows"]), int(CFG["grid_cols"]), int(CFG["overlap"])
    xs = np.linspace(win[0], win[2], Cn + 1).round().astype(int)
    ys = np.linspace(win[1], win[3], R + 1).round().astype(int)
    out = []
    for r in range(R):
        for c in range(Cn):
            core = (
                int(xs[c]) if c > 0 else -(10**6),
                int(ys[r]) if r > 0 else -(10**6),
                int(xs[c + 1]) if c < Cn - 1 else 10**6,
                int(ys[r + 1]) if r < R - 1 else 10**6,
            )
            a0 = box[0] + ((max(box[0], int(xs[c]) - ov) - box[0]) // al) * al
            b0 = box[1] + ((max(box[1], int(ys[r]) - ov) - box[1]) // al) * al
            a1 = min(box[2], int(xs[c + 1]) + ov)
            b1 = min(box[3], int(ys[r + 1]) + ov)
            a1 = a0 + ((a1 - a0) // al) * al if a1 < box[2] else a1
            b1 = b0 + ((b1 - b0) // al) * al if b1 < box[3] else b1
            out.append((core, (int(a0), int(b0), int(a1), int(b1))))
    return out


def _start_pool():
    global _POOLP
    import multiprocessing as mp, sys, types

    if _POOLP is not None:
        return

    if (
        sys.modules.get(__name__) is None
        or getattr(sys.modules[__name__], "_worker", None) is not _worker
    ):
        m = types.ModuleType(__name__)
        m.__dict__.update(globals())
        sys.modules[__name__] = m
    _POOLP = mp.get_context("fork").Pool(
        int(CFG["workers"]), initializer=cv2.setNumThreads, initargs=(1,)
    )


def _free_shm():
    for s in _OWNED:
        try:
            s.close()
            s.unlink()
        except Exception:
            pass
    _OWNED.clear()
    _SHM.clear()


def _ensure_shm(im_shape, bg_shape, oshape):
    global _GEOM
    from multiprocessing import shared_memory

    if _GEOM == (im_shape, bg_shape, oshape):
        return
    _free_shm()
    for key, shape, dt in (
        ("im", im_shape, np.uint8),
        ("bg", bg_shape, np.float32),
        ("out", oshape, np.int32),
        ("dark", bg_shape[:2], np.float32),
    ):
        seg = shared_memory.SharedMemory(
            create=True, size=int(np.prod(shape)) * np.dtype(dt).itemsize
        )
        _SHM[key] = seg
        _OWNED.append(seg)
    _GEOM = (im_shape, bg_shape, oshape)


@atexit.register
def _shutdown():
    if _POOLP is not None:
        _POOLP.terminate()
    _free_shm()


def segment_area(image_path, area=None, margin=MARGIN):
    t = time.perf_counter()
    half = bool(CFG["half_decode"])
    ps = 2 if half else 1
    im = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_2 if half else cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"cannot read image {image_path}")
    t = _tick("read", t)
    H, W = im.shape[0] * ps, im.shape[1] * ps
    if area is None:
        area = (0, 0, W, H)
    window = (
        max(0, area[0] - margin),
        max(0, area[1] - margin),
        min(W, area[2] + margin),
        min(H, area[3] + margin),
    )
    window = tuple(v - v % ps for v in window)
    win = tuple(v // ps for v in window)
    P_ = int(CFG["pad"]) // ps
    P_ -= P_ % (16 // ps)
    box = (
        max(0, win[0] - P_),
        max(0, win[1] - P_),
        min(im.shape[1], win[2] + P_),
        min(im.shape[0], win[3] + P_),
    )
    bg = background(im, box, ps)
    t = _tick("background", t)
    oshape = (win[3] - win[1], win[2] - win[0])
    dark = darkness_half(im[box[1] : box[3], box[0] : box[2]]) if (CFG["pp"] and half) else None
    if not use_pool():
        lab = process_arrays(
            im[box[1] : box[3], box[0] : box[2]].astype(np.float32), bg, pre=half, dark=dark
        )
        res = lab[win[1] - box[1] : win[3] - box[1], win[0] - box[0] : win[2] - box[0]]
    else:
        _start_pool()
        _ensure_shm(im.shape, bg.shape, oshape)
        np.ndarray(im.shape, np.uint8, buffer=_SHM["im"].buf)[:] = im
        np.ndarray(bg.shape, np.float32, buffer=_SHM["bg"].buf)[:] = bg
        out = np.ndarray(oshape, np.int32, buffer=_SHM["out"].buf)
        out[:] = 0
        names = (_SHM["im"].name, _SHM["bg"].name, _SHM["out"].name)
        if dark is not None:
            np.ndarray(bg.shape[:2], np.float32, buffer=_SHM["dark"].buf)[:] = dark
            names = names + (_SHM["dark"].name,)
        jobs = [
            (k, core, proc, box, names, (im.shape, bg.shape, oshape), win, half)
            for k, (core, proc) in enumerate(_tiles(box, win, 16 // ps))
        ]
        _POOLP.map(_worker, jobs, chunksize=1)
        res = out.copy()
    if half:
        res = cv2.resize(
            res.astype(np.float32),
            (window[2] - window[0], window[3] - window[1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    _tick("total", t)
    return res, window


def segment_window(image_path):
    lab, window = segment_area(image_path, SQUARE, MARGIN)
    assert window == WIN, window
    return lab
