# Counts apple cells in one microscope photo: it finds the counting square, segments the dark cell
# bodies, drops out-of-focus or faint objects, and labels each cell intact or broken. It writes the
# object table, overlay images and a summary, and can apply a reviewer's corrections file. All
# exported coordinates refer to the original image.

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

cv2.setNumThreads(1)
CONTRAST_THRESHOLD = 6.0
MIN_AREA = 8

DEFAULT_CONFIG = dict(
    virtual_grid=True,
    virtual_side_px=1096,
    set3_detection=True,
    set3_grow=7.0,
    set3_seed=9.0,
    set3_waist=0.80,
    baseline_r4=False,
    group_triple_cutoff_removal=True,
    neutral_proposals=False,
    neutral_seed=4.0,
    neutral_grow=2.7,
    neutral_B=0.9,
    neutral_strong=20.0,
    core_percentile_contrast=True,
    core_percentile=90,
    line_median_301=True,
    merge_core_preservation=True,
    rejected_mask_core_recovery=True,
    line_band_closing=True,
    click_snapping_dedup=True,
    audit_robustness=True,
    ghost_screen=True,
    shared_body_gate=True,
    body_threshold=0.9,
    parent_boundary_preservation=True,
    support_merge_anchors=True,
)
CONFIG = DEFAULT_CONFIG.copy()


def configure(config=None):
    global CONFIG, BODY_THRESHOLD
    CONFIG = DEFAULT_CONFIG.copy()
    if config:
        CONFIG.update(config)
    unknown = set(CONFIG) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("Unknown config keys: " + str(unknown))
    if CONFIG["virtual_grid"] and CONFIG["baseline_r4"]:
        raise ValueError("virtual_grid cannot use the physical-grid r4 baseline")
    BODY_THRESHOLD = CONFIG["body_threshold"]


CLASSES = {"intact": (40, 220, 40), "broken": (255, 65, 65), "uncertain": (255, 220, 20)}


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def save_image(path, im):
    cv2.imencode(".png", im)[1].tofile(str(path))


def grid(im):
    if CONFIG["virtual_grid"]:
        h, w = im.shape[:2]
        side = float(CONFIG["virtual_side_px"])
        if side <= 0 or side > min(h, w):
            raise ValueError("Virtual square must fit the image")
        x, y = (w - side) / 2, (h - side) / 2
        X, Y = x + side, y + side
        box = [x, y, X, Y]
        corners = np.array([[x, y], [X, y], [X, Y], [x, Y]], np.float32)
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        diag = [
            dict(
                mode="virtual",
                selection_candidates=[
                    dict(
                        complete=True,
                        visible_fraction=1.0,
                        center_distance_px=0.0,
                        rectified_bbox=box,
                    )
                ],
            )
        ]
        return (
            im,
            M,
            M.copy(),
            (True, 1.0, 0.0, box, corners),
            side / 4,
            diag,
            [np.zeros(w), np.zeros(h)],
        )
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
    lines = cv2.HoughLinesP(
        cv2.Canny(small, 35, 90), 1, np.pi / 1800, 90, minLineLength=250, maxLineGap=25
    )
    hs = []
    vs = []
    if lines is not None:
        for x, y, X, Y in lines.reshape(-1, 4):
            if abs(X - x) > 250 and abs(Y - y) < 30:
                hs.append((Y - y) / (X - x))
            if abs(Y - y) > 250 and abs(X - x) < 30:
                vs.append((X - x) / (Y - y))
    if not hs or not vs:
        raise ValueError("No reliable orthogonal grid lines")
    a = float(np.median(vs))
    c = float(np.median(hs))
    M = np.array([[1, -a, 0], [-c, 1, 0]], np.float32)
    inv = cv2.invertAffineTransform(M)
    rect = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    dark = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((41, 41), np.uint8))
    axes = []
    profiles = []
    pitches = []
    diagnostics = []
    for axis in (0, 1):
        kernel = np.ones((181, 1) if axis == 0 else (1, 181), np.uint8)
        p = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel).mean(axis=axis)
        p = ndi.gaussian_filter1d(p, 1.3)
        profiles.append(p)

        ac = np.correlate(p - p.mean(), p - p.mean(), "full")[len(p) - 1 :]
        peaks, _ = find_peaks(ac[100:500], distance=25)
        if not len(peaks):
            raise ValueError("No grid periodicity")
        pitch = float(100 + peaks[np.argmax(ac[100 + peaks])])
        pitches.append(pitch)

        best = np.zeros(len(p))
        ds = np.zeros(len(p))
        for d in range(round(0.075 * pitch), round(0.13 * pitch) + 1):
            score = np.zeros(len(p))
            score[d:-d] = np.minimum(np.minimum(p[: -2 * d], p[d:-d]), p[2 * d :])
            use = score > best
            best[use] = score[use]
            ds[use] = d
        centers, _ = find_peaks(best, distance=int(3.5 * pitch), prominence=1)

        edge = (centers < 0.2 * pitch) | (centers > len(p) - 0.2 * pitch)
        observed_all = centers[
            (
                (best[centers] > 1)
                if CONFIG["group_triple_cutoff_removal"]
                else ((best[centers] > 0.4 * best.max()) | (edge & (best[centers] > 1)))
            )
        ].tolist()
        centers = centers[
            (
                (best[centers] > 1)
                if CONFIG["group_triple_cutoff_removal"]
                else (best[centers] > 0.4 * best.max())
            )
            & (centers > 0.2 * pitch)
            & (centers < len(p) - 0.2 * pitch)
        ]
        if not len(centers):
            raise ValueError("Triple-line boundaries not found")
        period = float(np.median(np.diff(centers))) if len(centers) > 1 else 4 * pitch
        pitch_override = None
        if abs(period / 4 - pitch) > 8:

            if len(centers) > 1 and 250 <= period / 4 <= 300:
                pitch_override = {"autocorr_pitch_px": pitch, "pitch_px": period / 4}
                pitch = period / 4
                pitches[-1] = pitch
            else:
                raise ValueError("Triple-line period disagrees with small-square pitch")
        anchor = float(centers[np.argmax(best[centers])])
        positions = []
        for n in range(-6, 7):
            target = anchor + n * period
            near = centers[abs(centers - target) < 20]
            positions.append(float(near[0]) if len(near) else target)
        axes.append(sorted(positions))
        diagnostics.append(
            {
                "observed_triple_centers": observed_all,
                "period_px": period,
                "triple_line_spacing_px": float(ds[round(anchor)]),
                **({"pitch_override": pitch_override} if pitch_override else {}),
            }
        )

    frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    choices = []
    for x, X in zip(axes[0][:-1], axes[0][1:]):
        for y, Y in zip(axes[1][:-1], axes[1][1:]):
            corners = cv2.transform(np.array([[[x, y], [X, y], [X, Y], [x, Y]]], np.float32), inv)[
                0
            ]
            area = cv2.contourArea(corners)
            visible, _ = cv2.intersectConvexConvex(corners, frame)
            fraction = float(np.clip(visible / area, 0, 1))
            if fraction < 0.1:
                continue

            observed = all(
                (
                    min(abs(np.array(diag["observed_triple_centers"]) - v)) < 20
                    or v <= 0.13 * pitch
                    or v >= (w if ai == 0 else h) - 0.13 * pitch
                )
                for ai, (diag, vals) in enumerate(zip(diagnostics, ([x, X], [y, Y])))
                for v in vals
            )
            if not observed:
                continue
            local = dark[
                max(0, int(y) + 45) : min(h, int(Y) - 45), max(0, int(x) + 45) : min(w, int(X) - 45)
            ]
            if min(local.shape) < 100:
                continue
            verified = []
            for axis, start, end, origin in [
                (0, x, X, max(0, int(x) + 45)),
                (1, y, Y, max(0, int(y) + 45)),
            ]:
                kernel = np.ones((121, 1) if axis == 0 else (1, 121), np.uint8)
                lp = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel).mean(axis=axis)
                verified.append(
                    all(
                        np.max(
                            lp[
                                max(0, int(start + i * (end - start) / 4 - origin) - 12) : min(
                                    len(lp), int(start + i * (end - start) / 4 - origin) + 13
                                )
                            ],
                            initial=0,
                        )
                        > max(1, 0.12 * profiles[axis].max())
                        for i in (1, 2, 3)
                    )
                )
            if not all(verified):
                continue
            complete = bool(
                np.all(corners[:, 0] >= 0)
                and np.all(corners[:, 0] <= w - 1)
                and np.all(corners[:, 1] >= 0)
                and np.all(corners[:, 1] <= h - 1)
            )
            dist = float(np.linalg.norm(corners.mean(axis=0) - [w / 2, h / 2]))
            choices.append((complete, fraction, dist, [x, y, X, Y], corners))
    if not choices:
        raise ValueError("No observed triple-bounded group with verified interior lines")

    near_complete = [q for q in choices if q[0] or q[1] >= 0.99]
    selected = (
        min(near_complete, key=lambda q: q[2])
        if near_complete
        else min(
            [q for q in choices if q[1] >= max(v[1] for v in choices) - 0.01], key=lambda q: q[2]
        )
    )
    diagnostics[0]["selection_candidates"] = [
        dict(complete=q[0], visible_fraction=q[1], center_distance_px=q[2], rectified_bbox=q[3])
        for q in sorted(choices, key=lambda q: -q[1])
        if q[1] >= selected[1] - 0.01
    ]
    return rect, M, inv, selected, float(np.mean(pitches)), diagnostics, profiles


def object_record(mask, offset, inv, source="auto", override=None):
    props = regionprops(mask.astype(np.uint8))[0]
    yy, xx = np.where(mask)
    pts = np.column_stack([xx + offset[0], yy + offset[1]]).astype(np.float32)
    original = cv2.transform(pts[None], inv)[0]
    contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [
        cv2.transform((ct.reshape(-1, 2) + offset).astype(np.float32)[None], inv)[0]
        .round()
        .astype(np.int32)
        for ct in contour
    ]
    area = float(props.area)
    diameter = math.sqrt(4 * area / math.pi)

    circularity = 4 * math.pi * area / max(props.perimeter, 1) ** 2
    cls = "broken" if props.solidity < 0.75 or circularity < 0.40 else "uncertain"
    flags = []
    if props.solidity < 0.8 and area > 300:
        flags.append("clump")
    if CONFIG["virtual_grid"] and cls == "uncertain":
        cls = "intact"
        flags.append("low_evidence;provisional_shape")
    if override:
        cls = override
    return dict(
        x=float(original[:, 0].mean()),
        y=float(original[:, 1].mean()),
        area_px=area,
        eq_diam_um_meta=diameter * 0.37,
        bbox_x_min=int(original[:, 0].min()),
        bbox_y_min=int(original[:, 1].min()),
        bbox_x_max=int(original[:, 0].max()),
        bbox_y_max=int(original[:, 1].max()),
        **{"class": cls},
        flags=";".join(flags),
        source=source,
        _pts=pts,
        _contours=cs,
        circularity=circularity,
        solidity=float(props.solidity),
    )


def local_contrast(im, x, y, r=10):
    x, y = int(round(x)), int(round(y))
    r = max(3, min(float(r), 25))
    h, w = im.shape[:2]
    x0 = max(0, x - 50)
    y0 = max(0, y - 50)
    tile = im[y0 : min(h, y + 51), x0 : min(w, x + 51)].astype(np.float32).mean(2)
    tile = cv2.GaussianBlur(tile, (0, 0), 1.5)
    yy, xx = np.indices(tile.shape)
    rr = np.hypot(xx + x0 - x, yy + y0 - y)
    core = tile[rr <= max(3, 0.6 * r)]
    ring = tile[(rr >= r + 8) & (rr <= r + 20)]
    return float(np.median(ring) - np.percentile(core, 30)) if len(core) and len(ring) else -999.0


def fixed_contrast(field, px, py):

    x, y = int(round(px)), int(round(py))
    h, w = field.shape
    a, b = max(0, x - 30), max(0, y - 30)
    c, d = min(w, x + 31), min(h, y + 31)
    patch = field[b:d, a:c]
    yy, xx = np.indices(patch.shape)
    rr = np.hypot(xx + a - x, yy + b - y)
    core = patch[rr <= 4.8]
    ring = patch[(rr >= 16) & (rr <= 28)]
    return float(np.percentile(core, 70) - np.median(ring)) if len(core) and len(ring) else -999.0


def boundary_reason(pts, box):

    x, y, X, Y = box
    inside = (
        (pts[:, 0] >= x - 2) & (pts[:, 0] <= X + 2) & (pts[:, 1] >= y - 2) & (pts[:, 1] <= Y + 2)
    )
    if not inside.any():
        return "out_of_group"
    if np.any(pts[:, 0] + 2 >= X) or np.any(pts[:, 1] + 2 >= Y):
        return "edge_rule"
    return None


def candidate_sheet(im, items, path):
    tw, th, ncol = 210, 246, 6
    canvas = np.full((max(1, (len(items) + ncol - 1) // ncol) * th, ncol * tw, 3), 245, np.uint8)
    for i, o in enumerate(items):
        x, y = round(o["x"]), round(o["y"])
        a, b = max(0, x - 32), max(0, y - 32)
        t = im[b : min(im.shape[0], y + 33), a : min(im.shape[1], x + 33)].copy()

        t = np.clip(128 + 2 * (t.astype(float) - cv2.GaussianBlur(t, (0, 0), 12)), 0, 255).astype(
            np.uint8
        )
        t = cv2.resize(t, (195, 195), interpolation=cv2.INTER_NEAREST)
        for ct in o["_contours"]:
            scaled = (ct - [a, b]) * [
                195 / max(1, min(im.shape[1], x + 33) - a),
                195 / max(1, min(im.shape[0], y + 33) - b),
            ]
            cv2.polylines(t, [scaled.round().astype(np.int32)], True, (0, 255, 0), 1)
        cv2.drawMarker(
            t,
            (
                round((x - a) / max(1, min(im.shape[1], x + 33) - a) * 195),
                round((y - b) / max(1, min(im.shape[0], y + 33) - b) * 195),
            ),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            8,
            1,
        )
        q, r = i // ncol * th, i % ncol * tw
        canvas[q : q + 195, r : r + 195] = t
        lines = [
            f"{o['id']} {x},{y}",
            f"C={o['contrast_dL']:.2f} A={o['area_px']:.0f}",
            o.get("reason", "COUNT")
            + " "
            + ("J" if o.get("junction_flag") else "")
            + ("S" if o.get("sliver_flag") else "")
            + (" CORE" if "core_recovered" in o.get("flags", "") else ""),
        ]
        for j, txt in enumerate(lines):
            cv2.putText(canvas, txt, (r + 2, q + 208 + 15 * j), 0, 0.36, (0, 0, 0), 1)
    save_image(path, canvas)


def support_contrast(field, pts, ox, oy):
    if not CONFIG["core_percentile_contrast"]:
        from r4_reference import support_contrast as legacy

        return legacy(field, pts, ox, oy)
    q = np.rint(pts - [ox, oy]).astype(int)
    valid = (
        (q[:, 0] >= 0) & (q[:, 0] < field.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < field.shape[0])
    )
    q = q[valid]
    if not len(q):
        return -999.0
    peak = q[np.argmax(field[q[:, 1], q[:, 0]])]

    if len(q) >= MIN_AREA:
        return float(np.percentile(field[q[:, 1], q[:, 0]], CONFIG["core_percentile"]))
    return fixed_contrast(field, *peak)


def body_response(field):
    response = np.zeros_like(field)
    for sigma in (2.0, 4.0):
        g = cv2.GaussianBlur(field, (0, 0), sigma)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) / 4
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) / 4
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4
        weak = -(dxx + dyy + np.sqrt((dxx - dyy) ** 2 + 4 * dxy * dxy)) / 2
        response = np.maximum(response, weak * sigma * sigma)
    return response


BODY_THRESHOLD = 0.9


def body_value(obj, response, ox, oy):
    q = np.rint(obj["_pts"] - [ox, oy]).astype(int)
    valid = (
        (q[:, 0] >= 0)
        & (q[:, 0] < response.shape[1])
        & (q[:, 1] >= 0)
        & (q[:, 1] < response.shape[0])
    )
    q = q[valid]
    return float(np.max(response[q[:, 1], q[:, 0]], initial=0))


def record_points(pts, inv, old):
    lo = pts.min(0).astype(int)
    hi = pts.max(0).astype(int)
    mask = np.zeros((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), np.uint8)
    q = (pts - lo).astype(int)
    mask[q[:, 1], q[:, 0]] = 1
    obj = object_record(mask, lo, inv, old.get("source", "auto"), old["class"])
    for k in ["id", "flags", "reason", "review_reason", "contrast_dL", "_line_only_fraction"]:
        if k in old:
            obj[k] = old[k]
    return obj


def consolidate(candidates, rejected, inv, field, ox, oy, box):
    audit = []

    while not (CONFIG["virtual_grid"] and CONFIG["set3_detection"]):
        pair = None
        for i, a in enumerate(candidates):
            tree = cKDTree(a["_pts"])
            for b in candidates[i + 1 :]:
                if np.max(np.abs(a["_pts"].mean(0) - b["_pts"].mean(0))) > 150:
                    continue
                gap = float(tree.query(b["_pts"])[0].min())
                if (
                    gap <= 4
                    and min(a["area_px"], b["area_px"])
                    <= max(300, 0.5 * max(a["area_px"], b["area_px"]))
                ) or (
                    gap <= 5
                    and max(a.get("_line_only_fraction", 0), b.get("_line_only_fraction", 0)) >= 0.8
                ):
                    pair = (a, b, gap)
                    break
            if pair:
                break
        if pair is None:
            break
        a, b, gap = pair
        if a["area_px"] < b["area_px"]:
            a, b = b, a
        points = np.unique(np.concatenate([a["_pts"], b["_pts"]]), axis=0)
        merged = record_points(points, inv, a)
        merged["contrast_dL"] = (
            max(a["contrast_dL"], b["contrast_dL"], support_contrast(field, points, ox, oy))
            if CONFIG["merge_core_preservation"]
            else support_contrast(field, points, ox, oy)
        )
        reason = boundary_reason(points, box)
        if reason is None and merged["contrast_dL"] < 6:
            reason = "below_contrast"
        if reason:
            merged["reason"] = reason
        merged["flags"] += ";merged:" + b["id"]
        candidates.remove(a)
        candidates.remove(b)
        (rejected if reason else candidates).append(merged)
        b["reason"] = "merged_duplicate"
        b["review_reason"] = "One particle; merged support into " + a["id"]
        rejected.append(b)
        audit.append(dict(kept=a["id"], suppressed=b["id"], gap_px=gap))
    near = []
    for i, a in enumerate(candidates):
        tree = cKDTree(a["_pts"])
        for b in candidates[i + 1 :]:
            if np.max(np.abs(a["_pts"].mean(0) - b["_pts"].mean(0))) > 150:
                continue
            gap = float(tree.query(b["_pts"])[0].min())
            if gap <= 10:
                near.append(dict(a=a["id"], b=b["id"], gap_px=gap))
    return audit, near


def virtual_background(image):
    h, w = image.shape[:2]
    small = cv2.resize(
        image.astype(np.float32), (math.ceil(w / 8), math.ceil(h / 8)), interpolation=cv2.INTER_AREA
    )
    envelope = ndi.percentile_filter(small, 70, size=(41, 41, 1), mode="reflect")
    envelope = cv2.GaussianBlur(envelope, (0, 0), 35 / 8)
    return cv2.resize(envelope, (w, h), interpolation=cv2.INTER_LINEAR)


def set3_segment(tile, bg, contrast):
    lab = cv2.cvtColor(tile / 255.0, cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(np.clip(bg / 255.0, 0, 1), cv2.COLOR_BGR2LAB)
    residual = lab[:, :, 1:] - ref[:, :, 1:]
    residual -= 0.85 * cv2.GaussianBlur(residual, (0, 0), 25)
    chroma = np.linalg.norm(residual, axis=2)

    local_dark = contrast - 0.85 * cv2.GaussianBlur(np.maximum(contrast, 0), (0, 0), 25)
    score = cv2.GaussianBlur(np.maximum(np.maximum(local_dark, 0), chroma * 1.8), (0, 0), 1.2)
    support = score >= CONFIG["set3_grow"]
    support = ndi.binary_fill_holes(support)
    support = ndi.binary_opening(support, structure=np.ones((3, 3)))
    dist = ndi.distance_transform_edt(support).astype(np.float32)
    smooth = cv2.GaussianBlur(dist, (0, 0), 2)
    seed_map = cv2.GaussianBlur(score, (0, 0), 3)
    maxima = h_maxima(seed_map, 3.0) & support & (dist >= 2) & (score >= CONFIG["set3_seed"])
    markers = label(maxima)

    components = label(support)
    for p in regionprops(components):
        m = components[p.slice] == p.label
        if not np.any(markers[p.slice][m]):
            q = np.where(m, smooth[p.slice], -1)
            v = np.unravel_index(q.argmax(), q.shape)
            markers[p.bbox[0] + v[0], p.bbox[1] + v[1]] = markers.max() + 1
    split = watershed(-smooth, markers, mask=support)

    n = int(split.max())
    parent = np.arange(n + 1)
    peaks = ndi.maximum(dist, split, index=np.arange(n + 1))

    def owner(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    saddles = {}
    for axis in (0, 1):
        a = split[:-1, :] if axis == 0 else split[:, :-1]
        b = split[1:, :] if axis == 0 else split[:, 1:]
        da = dist[:-1, :] if axis == 0 else dist[:, :-1]
        db = dist[1:, :] if axis == 0 else dist[:, 1:]
        hit = (a != b) & (a > 0) & (b > 0)
        for u, v, d in zip(a[hit], b[hit], np.minimum(da[hit], db[hit])):
            key = tuple(sorted((int(u), int(v))))
            saddles[key] = max(saddles.get(key, 0), float(d))
    for (a, b), s in sorted(saddles.items(), key=lambda q: -q[1]):
        a, b = owner(a), owner(b)
        if a != b and s >= CONFIG["set3_waist"] * min(peaks[a], peaks[b]):
            parent[b] = a
            peaks[a] = max(peaks[a], peaks[b])
    lookup = np.array([owner(i) for i in range(n + 1)])
    return lookup[split], score


def set3_measure(obj, field, score, ox, oy):
    q = np.rint(obj["_pts"] - [ox, oy]).astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 0] < field.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < field.shape[0])]
    if len(q) < MIN_AREA:
        return "small_area"
    x0, y0 = q.min(0)
    x1, y1 = q.max(0) + 1
    mask = np.zeros((y1 - y0 + 4, x1 - x0 + 4), np.uint8)
    mask[q[:, 1] - y0 + 2, q[:, 0] - x0 + 2] = 1
    p = regionprops(mask)[0]
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    boundary = (mask > 0) & ~(cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0)

    a, b = max(0, x0 - 16), max(0, y0 - 16)
    c, d = min(field.shape[1], x1 + 16), min(field.shape[0], y1 + 16)
    patch = field[b:d, a:c]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) / 8
    yy, xx = np.where(boundary)
    bx = np.clip(xx + x0 - 2, 0, field.shape[1] - 1)
    by = np.clip(yy + y0 - 2, 0, field.shape[0] - 1)
    edge = float(np.percentile(np.hypot(gx[by - b, bx - a], gy[by - b, bx - a]), 75))
    vals = field[q[:, 1], q[:, 0]]
    C = float(np.percentile(vals, 90))
    S = float(np.percentile(score[q[:, 1], q[:, 0]], 90))
    iy, ix = np.where(inner)
    interior = (
        field[
            np.clip(iy + y0 - 2, 0, field.shape[0] - 1), np.clip(ix + x0 - 2, 0, field.shape[1] - 1)
        ]
        if len(ix)
        else vals
    )
    sd = float(np.std(interior))
    core = float(np.median(interior))
    diam = math.sqrt(4 * len(q) / math.pi)
    detail = patch - cv2.GaussianBlur(patch, (0, 0), 4)
    texture = float(np.std(detail[q[:, 1] - b, q[:, 0] - a]))
    obj.update(
        stain_score=S,
        edge_gradient=edge,
        interior_std=sd,
        interior_variance=sd * sd,
        minor_axis_px=float(p.axis_minor_length),
        eccentricity=float(p.eccentricity),
        focus_diameter_px=diam,
        structure_std=texture,
    )

    if diam > 12 and core < max(2.0, 0.18 * C) and C < 14 and S < 18:
        return "faint_ring"
    if (
        p.axis_minor_length > 35
        and p.eccentricity < 0.85
        and edge / max(C, 1) < 0.075
        and texture < 0.75
    ):
        return "defocused_disc"
    if edge < 0.35 and sd < 1.5:
        return "haze"
    if S < CONFIG["set3_seed"]:
        return "weak_particle_signal"
    if C < CONTRAST_THRESHOLD:
        return "below_contrast"
    return None


def count(image_path, out, corrections=None, audit_center=None):
    if CONFIG["baseline_r4"]:
        from r4_reference import count as legacy

        return legacy(image_path, out, corrections, audit_center)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    im = read_image(image_path)
    if im is None:
        raise ValueError("Cannot read image")
    rect, M, inv, selected, pitch, diag, profiles = grid(im)
    if audit_center is not None:
        opts = diag[0]["selection_candidates"]
        q = min(
            opts,
            key=lambda q: np.linalg.norm(
                cv2.transform(
                    np.array(
                        [
                            [
                                [
                                    0.5 * (q["rectified_bbox"][0] + q["rectified_bbox"][2]),
                                    0.5 * (q["rectified_bbox"][1] + q["rectified_bbox"][3]),
                                ]
                            ]
                        ],
                        np.float32,
                    ),
                    inv,
                )[0, 0]
                - audit_center
            ),
        )
        box = q["rectified_bbox"]
        corners = cv2.transform(
            np.array(
                [[[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]],
                np.float32,
            ),
            inv,
        )[0]
        selected = (q["complete"], q["visible_fraction"], q["center_distance_px"], box, corners)
    complete, fraction, dist, box, corners = selected
    x, y, X, Y = box
    h, w = im.shape[:2]
    margin = 65
    ox = max(0, int(x) - margin)
    oy = max(0, int(y) - margin)
    ex = min(w, int(X) + margin)
    ey = min(h, int(Y) + margin)
    tile = rect[oy:ey, ox:ex].astype(np.float32)
    if CONFIG["virtual_grid"]:

        bg = virtual_background(rect)[oy:ey, ox:ex]
        darkness = bg.mean(2) - tile.mean(2)
        contrast_field = cv2.GaussianBlur(darkness, (0, 0), 1.5)
        gc = np.zeros(tile.shape[1], bool)
        gr = np.zeros(tile.shape[0], bool)
    else:

        grid_cols = profiles[0] > max(2, profiles[0].max() * 0.15)
        grid_rows = profiles[1] > max(2, profiles[1].max() * 0.15)
        excluded = (
            ndi.maximum_filter1d(grid_cols, size=35)[ox:ex][None, :]
            | ndi.maximum_filter1d(grid_rows, size=35)[oy:ey][:, None]
        )
        weight = (~excluded).astype(np.float32)
        den = cv2.GaussianBlur(weight, (0, 0), 35)
        bg = cv2.GaussianBlur(tile * weight[..., None], (0, 0), 35) / np.maximum(
            den[..., None], 0.001
        )
        darkness = bg.mean(2) - tile.mean(2)

        col = np.zeros_like(darkness)
        row = np.zeros_like(darkness)
        gc = ndi.maximum_filter1d(grid_cols, size=101)[ox:ex]
        gr = ndi.maximum_filter1d(grid_rows, size=101)[oy:ey]
        col[:, gc] = ndi.median_filter(
            darkness[:, gc], size=(301 if CONFIG["line_median_301"] else 101, 1), mode="nearest"
        )
        row[gr, :] = ndi.median_filter(
            darkness[gr, :], size=(1, 301 if CONFIG["line_median_301"] else 101), mode="nearest"
        )
        corrected = darkness - np.maximum(col, 0) - np.maximum(row, 0)

        squares = []
        for j in range(4):
            for i in range(4):
                a, b = int(x + i * (X - x) / 4) - ox, int(y + j * (Y - y) / 4) - oy
                c, d = int(x + (i + 1) * (X - x) / 4) - ox, int(y + (j + 1) * (Y - y) / 4) - oy
                if a >= 0 and b >= 0 and c <= corrected.shape[1] and d <= corrected.shape[0]:
                    squares.append(cv2.resize(corrected[b:d, a:c], (128, 128)))
        template = cv2.GaussianBlur(
            np.maximum(np.median(squares, axis=0), 0).astype(np.float32), (0, 0), 1
        )
        yy, xx = np.indices(corrected.shape, dtype=np.float32)
        mx = ((xx + ox - x) / ((X - x) / 4) % 1 * 128).astype(np.float32)
        my = ((yy + oy - y) / ((Y - y) / 4) % 1 * 128).astype(np.float32)
        baseline = cv2.remap(template, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        contrast_field = cv2.GaussianBlur(corrected - baseline, (0, 0), 1.5)

    chromatic_bg = cv2.GaussianBlur(tile, (0, 0), 35) if CONFIG["virtual_grid"] else bg
    od = np.log((chromatic_bg + 8) / (tile + 8))
    b, g, r = cv2.split(od)
    stain = r - 0.50 * (g + b)
    signal = np.maximum(np.maximum(stain, 0) * 255, np.maximum(r, 0) * 80)

    ridges = np.maximum(np.median(signal, axis=0)[None, :], np.median(signal, axis=1)[:, None])
    ridges = cv2.dilate(ridges, np.ones((21, 21), np.uint8))
    if CONFIG["virtual_grid"]:
        ridges = np.zeros_like(signal)
    clean = cv2.GaussianBlur(np.maximum(signal - 1.4 * ridges, 0), (3, 3), 0.65)
    seeds = clean > 5.5
    support = clean > 2.7
    support = ndi.binary_propagation(seeds, mask=support)
    support = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    support = ndi.binary_fill_holes(support)

    sharp_field = contrast_field - cv2.GaussianBlur(contrast_field, (0, 0), 8)
    yy0, xx0 = np.indices(contrast_field.shape)
    dx0 = np.min(abs(xx0[..., None] + ox - np.linspace(x, X, 5)), axis=2)
    dy0 = np.min(abs(yy0[..., None] + oy - np.linspace(y, Y, 5)), axis=2)
    gap_zone = (
        (np.minimum(dx0, dy0) >= 17) & (np.minimum(dx0, dy0) <= 30) & (np.maximum(dx0, dy0) > 65)
    )
    if CONFIG["virtual_grid"]:
        gap_zone[:] = False
    body_field = body_response(contrast_field)

    if CONFIG["neutral_proposals"]:
        line_seeds = (
            (contrast_field >= CONFIG["neutral_seed"]) & (body_field >= CONFIG["neutral_B"])
        ) | (contrast_field >= CONFIG["neutral_strong"])
        line_support = ndi.binary_propagation(
            line_seeds, mask=(contrast_field >= CONFIG["neutral_grow"])
        )
    else:
        line_seeds = (contrast_field >= 20) | (
            (contrast_field >= 14) & (sharp_field >= 8) & gap_zone
        )
        line_support = ndi.binary_propagation(line_seeds, mask=(contrast_field >= 8))
    chromatic_support = support.copy()
    support |= line_support

    for vertical, distance in (
        [(True, dx0), (False, dy0)]
        if CONFIG["line_band_closing"] and not CONFIG["virtual_grid"]
        else []
    ):
        kernel = np.ones((1, 35) if vertical else (35, 1), np.uint8)
        bridge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        support |= bridge & (distance <= 18)
    labels = label(support, connectivity=2)
    set3 = CONFIG["virtual_grid"] and CONFIG["set3_detection"]
    particle_score = None
    if set3:
        labels, particle_score = set3_segment(tile, bg, contrast_field)
    candidates = []
    rejected = []
    for p in regionprops(labels):
        if p.area < MIN_AREA:
            continue
        mask = labels[p.slice] == p.label
        obj = object_record(mask, (ox + p.bbox[1], oy + p.bbox[0]), inv)
        if set3:
            obj.update(
                id=f"A{len(candidates)+len(rejected)+1:04d}",
                contrast_dL=support_contrast(contrast_field, obj["_pts"], ox, oy),
                _line_only_fraction=0.0,
                flags=obj["flags"] + ";set3_soft_body",
            )
            reason = boundary_reason(obj["_pts"], box) or set3_measure(
                obj, contrast_field, particle_score, ox, oy
            )
            if reason:
                obj["reason"] = reason
                rejected.append(obj)
            else:
                candidates.append(obj)
            continue
        obj["_line_only_fraction"] = (
            0.0
            if CONFIG["virtual_grid"]
            else float(np.mean((line_support & ~chromatic_support)[p.slice][mask]))
        )
        pts = obj["_pts"]
        reason = None
        parent_boundary = boundary_reason(pts, box)
        center = pts[np.argmin(np.sum((pts - pts.mean(0)) ** 2, axis=1))]
        center_c = fixed_contrast(contrast_field, center[0] - ox, center[1] - oy)
        if center_c < 6 and len(pts) >= 400:
            q = (pts - [ox, oy]).astype(int)
            probe = q[np.linspace(0, len(q) - 1, min(128, len(q))).astype(int)]
            cs = np.array([fixed_contrast(contrast_field, a, b) for a, b in probe])
            if cs.max() >= 15:
                peak = probe[cs.argmax()]
                core = mask & (contrast_field[p.slice] >= 6)
                labs = label(core, connectivity=2)
                ids = labs[peak[1] - p.bbox[0], peak[0] - p.bbox[1]]
                if ids and (labs == ids).sum() >= MIN_AREA:
                    obj = object_record(labs == ids, (ox + p.bbox[1], oy + p.bbox[0]), inv)
                    obj["flags"] = "core_recovered"
                    obj["_line_only_fraction"] = 0
                    pts = obj["_pts"]
        reason = (
            parent_boundary if CONFIG["parent_boundary_preservation"] else None
        ) or boundary_reason(pts, box)
        if (
            not CONFIG["virtual_grid"]
            and reason is None
            and p.axis_major_length / max(p.axis_minor_length, 1) > 6
        ):
            reason = "line_artefact"
        if (
            not CONFIG["virtual_grid"]
            and reason is None
            and (p.bbox[2] - p.bbox[0] > 90 or p.bbox[3] - p.bbox[1] > 90)
            and p.solidity < 0.3
        ):
            reason = "line_artefact"
        obj["contrast_dL"] = support_contrast(contrast_field, obj["_pts"], ox, oy)
        if reason is None and obj["contrast_dL"] < CONTRAST_THRESHOLD:
            reason = "below_contrast"
        q = (obj["_pts"] - [ox, oy]).astype(int)
        xx = q[:, 0]
        yy = q[:, 1]
        dx = min(abs(obj["_pts"][:, 0].mean() - v) for v in np.linspace(x, X, 5))
        dy = min(abs(obj["_pts"][:, 1].mean() - v) for v in np.linspace(y, Y, 5))
        ww = np.ptp(xx) + 1
        hh = np.ptp(yy) + 1
        chromatic_body = int(np.sum((clean[yy, xx] >= 5.5) & (stain[yy, xx] * 255 >= 5.5))) >= max(
            8, 0.25 * len(xx)
        )
        junction = not CONFIG["virtual_grid"] and dx <= 25 and dy <= 25 and obj["area_px"] <= 120
        sliver = not CONFIG["virtual_grid"] and (
            (ww <= 6 and hh >= 2.5 * ww and dx <= 30) or (hh <= 6 and ww >= 2.5 * hh and dy <= 30)
        )
        obj["sharpness"] = float(np.percentile(sharp_field[yy, xx], 90))
        obj["junction_flag"] = bool(junction)
        obj["sliver_flag"] = bool(sliver)
        if (
            not CONFIG["virtual_grid"]
            and reason in (None, "below_contrast")
            and min(dx, dy) <= 55
            and obj["area_px"] > 120
            and obj["sharpness"] < 3
        ):
            reason = "line_artefact"
        if CONFIG["shared_body_gate"]:
            if (
                reason in (None, "below_contrast")
                and body_value(obj, body_field, ox, oy) < BODY_THRESHOLD
            ):
                reason = "not_stained"
        elif reason is None and not chromatic_body and obj["sharpness"] < 2:
            reason = "not_stained"
        obj["chromatic_body"] = bool(chromatic_body)
        if (
            not CONFIG["virtual_grid"]
            and reason in (None, "below_contrast")
            and (junction or sliver)
            and (
                body_value(obj, body_field, ox, oy) < BODY_THRESHOLD
                if CONFIG["shared_body_gate"]
                else (not chromatic_body or obj["contrast_dL"] < 20)
            )
        ):
            reason = "line_artefact"
        obj["id"] = f"A{len(candidates)+len(rejected)+1:04d}"
        if reason:
            obj["reason"] = reason
            rejected.append(obj)
        else:
            candidates.append(obj)
    rescued = []
    for parent in (list(rejected) if CONFIG["rejected_mask_core_recovery"] and not set3 else []):
        if (
            parent.get("reason") not in ("below_contrast", "not_stained", "line_artefact")
            or parent["area_px"] < 120
        ):
            continue
        if boundary_reason(parent["_pts"], box):
            continue
        q = np.rint(parent["_pts"] - [ox, oy]).astype(int)
        local = np.zeros(contrast_field.shape, np.uint8)
        local[q[:, 1], q[:, 0]] = 1
        cores = label((local > 0) & (contrast_field >= 6), connectivity=2)
        for part in regionprops(cores):
            if part.area < MIN_AREA:
                continue
            obj = object_record(
                cores[part.slice] == part.label, (ox + part.bbox[1], oy + part.bbox[0]), inv
            )
            obj["contrast_dL"] = support_contrast(contrast_field, obj["_pts"], ox, oy)
            if obj["contrast_dL"] < 6 or body_value(obj, body_field, ox, oy) < BODY_THRESHOLD:
                continue
            if any(
                cKDTree(o["_pts"]).query(obj["_pts"])[0].min() <= 3 for o in candidates + rescued
            ):
                continue
            obj.update(id=f"K{len(rescued)+1:04d}", flags="core_recovered;parent:" + parent["id"])
            rescued.append(obj)
    candidates.extend(rescued)
    baseline = out / "baseline_r2_objects.csv"
    if baseline.exists():
        probes = []
        for rec in pd.read_csv(baseline).to_dict("records"):
            pt = cv2.transform(np.array([[[rec["x"], rec["y"]]]], np.float32), M)[0, 0]
            if ox <= pt[0] < ex and oy <= pt[1] < ey:
                probes.append(
                    dict(
                        id=rec["id"],
                        x=rec["x"],
                        y=rec["y"],
                        old_C=rec["contrast_dL"],
                        wide_band_C=fixed_contrast(contrast_field, pt[0] - ox, pt[1] - oy),
                        in_wide_band=bool(gc[int(pt[0]) - ox] or gr[int(pt[1]) - oy]),
                    )
                )
        pd.DataFrame(probes).to_csv(out / "background_audit.csv", index=False)
    samples = []
    for name in ["reference_r1.csv", "r2_points.csv"]:
        path = out / name
        if path.exists():
            for rec in pd.read_csv(path).to_dict("records"):
                pt = cv2.transform(np.array([[[rec["x"], rec["y"]]]], np.float32), M)[0, 0]
                if ox <= pt[0] < ex and oy <= pt[1] < ey:
                    samples.append(
                        dict(
                            **rec,
                            contrast_dL=fixed_contrast(contrast_field, pt[0] - ox, pt[1] - oy),
                            sample_source=name,
                        )
                    )
    if samples:
        pd.DataFrame(samples).to_csv(out / "r2_measured_points.csv", index=False)
    correction_audit = []
    correction_data = {}
    if corrections and Path(corrections).exists():
        correction_data = json.loads(Path(corrections).read_text(encoding="utf-8-sig"))
        for ri, removal in enumerate(correction_data.get("remove", []), 1):

            assert removal.get("reason") and removal.get("locked_contours")
            payload = json.dumps(removal["locked_contours"], sort_keys=True)
            assert (
                hashlib.sha256(payload.encode()).hexdigest() == removal["geometry_sha256"]
            ), "Removal geometry changed"
            mask = np.zeros(rect.shape[:2], np.uint8)
            for contour in removal["locked_contours"]:
                points = (
                    cv2.transform(np.array([contour], np.float32), M)[0].round().astype(np.int32)
                )
                cv2.fillPoly(mask, [points], 1)
            yy, xx = np.where(mask)
            assert (
                abs(len(xx) / removal["expected_area_px"] - 1) <= 0.25
            ), "Removal area guard failed"
            removed = 0
            for collection in (candidates, rejected):
                for old in list(collection):
                    pts = old["_pts"].round().astype(int)
                    hit = mask[pts[:, 1], pts[:, 0]] > 0
                    if not hit.any():
                        continue
                    removed += int(hit.sum())
                    collection.remove(old)
                    remain = pts[~hit]
                    if len(remain) >= MIN_AREA:
                        obj = record_points(remain, inv, old)
                        obj["contrast_dL"] = support_contrast(contrast_field, obj["_pts"], ox, oy)
                        if obj.get("reason") == "below_contrast":
                            obj.pop("reason")
                        if not obj.get("reason") and obj["contrast_dL"] < 6:
                            obj["reason"] = "below_contrast"
                        (rejected if obj.get("reason") else candidates).append(obj)
            obj = object_record(
                mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1], (xx.min(), yy.min()), inv
            )
            obj.update(
                id=f"R{ri:04d}",
                reason="manual_remove",
                review_reason=removal["reason"],
                contrast_dL=fixed_contrast(contrast_field, xx.mean() - ox, yy.mean() - oy),
            )
            rejected.append(obj)
            correction_audit.append(
                dict(
                    kind="remove",
                    id=obj["id"],
                    locked_area=len(xx),
                    removed_support_px=removed,
                    status="applied_locked_region",
                )
            )
        for i, addition in enumerate(correction_data.get("add", []), 1):
            if not addition.get("reason"):
                raise ValueError("Every correction requires a reason")
            cls = addition["class"]
            if cls not in CLASSES:
                raise ValueError("Invalid manual class")
            mask = np.zeros(rect.shape[:2], np.uint8)
            clipped = False
            click_outside = False
            if "polygon" in addition:
                points = (
                    cv2.transform(np.array([addition["polygon"]], np.float32), M)[0]
                    .round()
                    .astype(np.int32)
                )
                cv2.fillPoly(mask, [points], 1)
            else:
                pt = cv2.transform(np.array([[[addition["x"], addition["y"]]]], np.float32), M)[
                    0, 0
                ]
                cx, cy = np.round(pt).astype(int)
                rad = int(addition["radius"])
                choices = []
                for collection in (candidates, rejected):
                    for owner in collection:
                        if owner.get("reason") in ("manual_remove", "merged_duplicate"):
                            continue
                        if owner["contrast_dL"] < 6 or (
                            not set3 and body_value(owner, body_field, ox, oy) < BODY_THRESHOLD
                        ):
                            continue
                        if set3 and owner.get("reason"):
                            continue
                        dists = np.sum((owner["_pts"] - pt) ** 2, axis=1)
                        gap = float(np.sqrt(dists.min()))
                        if gap <= (3 if set3 else 25):
                            choices.append((gap, owner))
                if choices and CONFIG["click_snapping_dedup"]:
                    _, nearest = min(choices, key=lambda item: item[0])
                    q = np.rint(nearest["_pts"] - [ox, oy]).astype(int)
                    valid = (
                        (q[:, 0] >= 0)
                        & (q[:, 0] < contrast_field.shape[1])
                        & (q[:, 1] >= 0)
                        & (q[:, 1] < contrast_field.shape[0])
                    )
                    q = q[valid]
                    if len(q):
                        peak = q[np.argmax(contrast_field[q[:, 1], q[:, 0]])]
                        cx, cy = peak + [ox, oy]
                if not (ox <= cx < ex and oy <= cy < ey):
                    obj = object_record(
                        np.ones((3, 3), np.uint8), (cx - 1, cy - 1), inv, "manual_add", cls
                    )
                    obj.update(
                        id=f"M{i:04d}",
                        contrast_dL=-999.0,
                        reason="out_of_group",
                        review_reason=addition["reason"],
                    )
                    rejected.append(obj)
                    correction_audit.append(dict(kind="add", id=obj["id"], status="out_of_group"))
                    continue
                a, b = max(ox, cx - rad), max(oy, cy - rad)
                c, d = min(ex, cx + rad + 1), min(ey, cy + rad + 1)

                clipped = True
                click_outside = False
                a, b = max(ox, cx - 80), max(oy, cy - 80)
                c, d = min(ex, cx + 81), min(ey, cy + 81)
                patch = contrast_field[b - oy : d - oy, a - ox : c - ox]
                level = float(patch[cy - b, cx - a])
                part = np.zeros_like(patch, dtype=bool)
                part[cy - b, cx - a] = True
                if level <= 2.7:
                    click_outside = True
                    clipped = False
                else:
                    for threshold in np.arange(max(2.7, 0.5 * level), level, 0.75):
                        labs = label(patch > threshold)
                        chosen = labs[cy - b, cx - a]
                        if not chosen:
                            break
                        part = labs == chosen
                        clipped = bool(
                            part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any()
                        )
                        if not clipped:
                            break
                mask[b:d, a:c] = part
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError("Empty manual object")
            obj = object_record(
                mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
                (xs.min(), ys.min()),
                inv,
                "manual_add",
                cls,
            )
            obj["id"] = f"M{i:04d}"
            obj["flags"] = addition.get("flags", "")
            obj["review_reason"] = addition["reason"]
            obj["contrast_dL"] = support_contrast(contrast_field, obj["_pts"], ox, oy)
            center = cv2.transform(
                np.array(
                    [[[addition.get("x", obj["x"]), addition.get("y", obj["y"])]]], np.float32
                ),
                M,
            )[0, 0]
            overlap = [
                o
                for o in candidates
                if (
                    np.any(mask[o["_pts"][:, 1].astype(int), o["_pts"][:, 0].astype(int)] > 0)
                    if CONFIG["click_snapping_dedup"] and not set3
                    else np.min(np.linalg.norm(o["_pts"] - center, axis=1)) <= 3
                )
            ]
            if overlap:
                correction_audit.append(
                    dict(kind="add", id=obj["id"], owner=overlap[0]["id"], status="counted_owner")
                )
                continue

            for owner in list(rejected):
                if owner.get("reason") == "manual_remove" or (
                    CONFIG["audit_robustness"]
                    and (
                        owner.get("source") == "manual_add"
                        or any(r.get("owner") == owner["id"] for r in correction_audit)
                    )
                ):
                    continue
                pts = owner["_pts"].round().astype(int)
                hit = mask[pts[:, 1], pts[:, 0]] > 0
                if hit.any():
                    rejected.remove(owner)
                    if (~hit).sum() >= MIN_AREA:
                        carved = record_points(pts[~hit], inv, owner)
                        carved["flags"] += ";manual_split:" + obj["id"]
                        rejected.append(carved)
            if clipped:
                obj["flags"] += ";roi_clipped"
            reason = boundary_reason(obj["_pts"], box)
            if reason is None and click_outside:
                reason = "manual_click_outside_support"
            if reason is None and clipped:
                reason = "manual_support_clipped"
            if reason is None and obj["area_px"] < MIN_AREA:
                reason = "small_area"
            if reason is None and obj["contrast_dL"] < CONTRAST_THRESHOLD:
                reason = "below_contrast"
            if reason:
                obj["reason"] = reason
                rejected.append(obj)
            else:
                candidates.append(obj)
            correction_audit.append(dict(kind="add", id=obj["id"], status=reason or "counted"))
    if not CONFIG["shared_body_gate"] and not set3:

        for obj in list(candidates) + [o for o in rejected if o.get("reason") == "below_contrast"]:
            q = (obj["_pts"] - [ox, oy]).astype(int)
            xx = q[:, 0]
            yy = q[:, 1]
            valid = (xx >= 0) & (xx < clean.shape[1]) & (yy >= 0) & (yy < clean.shape[0])
            xx = xx[valid]
            yy = yy[valid]
            violet = np.sum((clean[yy, xx] >= 4.5) & (stain[yy, xx] * 255 >= 4.5)) >= 8
            strong = bool(len(xx) and np.percentile(sharp_field[yy, xx], 90) >= 3)
            if not violet and not strong:
                (rejected if obj.get("reason") else candidates).remove(obj)
                obj["reason"] = "not_stained"
                rejected.append(obj)

    for obj in (
        list(candidates) + [o for o in rejected if o.get("reason") == "below_contrast"]
        if CONFIG["shared_body_gate"] and not set3
        else []
    ):
        obj["body_response"] = body_value(obj, body_field, ox, oy)
        q = np.rint(obj["_pts"] - [ox, oy]).astype(int)
        valid = (
            (q[:, 0] >= 0)
            & (q[:, 0] < clean.shape[1])
            & (q[:, 1] >= 0)
            & (q[:, 1] < clean.shape[0])
        )
        q = q[valid]
        if not len(q):
            continue
        xx, yy = q.T
        dx = min(abs(obj["_pts"][:, 0].mean() - v) for v in np.linspace(x, X, 5))
        dy = min(abs(obj["_pts"][:, 1].mean() - v) for v in np.linspace(y, Y, 5))
        ww, hh = np.ptp(xx) + 1, np.ptp(yy) + 1
        junction = not CONFIG["virtual_grid"] and dx <= 25 and dy <= 25 and obj["area_px"] <= 120
        sliver = not CONFIG["virtual_grid"] and (
            (ww <= 6 and hh >= 2.5 * ww and dx <= 30) or (hh <= 6 and ww >= 2.5 * hh and dy <= 30)
        )
        violet = np.sum((clean[yy, xx] >= 5.5) & (stain[yy, xx] * 255 >= 5.5)) >= max(
            8, 0.25 * len(xx)
        )
        reason = None
        if obj["body_response"] < BODY_THRESHOLD:
            reason = "not_stained"
        elif (junction or sliver) and (not violet or obj["contrast_dL"] < 20):
            reason = "line_artefact"
        elif (
            not CONFIG["virtual_grid"]
            and min(dx, dy) <= 55
            and obj["area_px"] > 120
            and np.percentile(sharp_field[yy, xx], 90) < 3
        ):
            reason = "line_artefact"
        if reason:
            (rejected if obj.get("reason") else candidates).remove(obj)
            obj["reason"] = reason
            rejected.append(obj)
    if set3:
        for obj in list(candidates) + [o for o in rejected if o.get("reason") == "below_contrast"]:
            reason = set3_measure(obj, contrast_field, particle_score, ox, oy)
            if reason and reason != obj.get("reason"):
                (rejected if obj.get("reason") else candidates).remove(obj)
                obj["reason"] = reason
                rejected.append(obj)
    merge_audit, near_pairs = consolidate(candidates, rejected, inv, contrast_field, ox, oy, box)

    for pair in correction_data.get("merge", []):
        owners = []
        for anchor in pair["points"]:
            if not CONFIG["support_merge_anchors"]:
                o = min(candidates, key=lambda o: np.hypot(o["x"] - anchor[0], o["y"] - anchor[1]))
                if np.hypot(o["x"] - anchor[0], o["y"] - anchor[1]) <= pair.get("tolerance_px", 60):
                    owners.append(o)
                continue
            if not candidates:
                continue
            point = cv2.transform(np.array([[anchor]], np.float32), M)[0, 0]
            distances = [float(np.linalg.norm(o["_pts"] - point, axis=1).min()) for o in candidates]
            j = int(np.argmin(distances))
            if distances[j] <= pair.get("tolerance_px", 60):
                owners.append(candidates[j])

        distinct = []
        [distinct.append(o) for o in owners if not any(o is d for d in distinct)]
        if len(distinct) < 2:
            merge_audit.append(
                dict(
                    kept=None,
                    suppressed=None,
                    gap_px=None,
                    review_reason=pair["reason"],
                    status=(
                        "already_consolidated"
                        if CONFIG["audit_robustness"] and len(distinct) == 1 and len(owners) >= 2
                        else "merge_not_applied: fewer than 2 distinct counted objects near the anchors"
                    ),
                    anchors=pair["points"],
                )
            )
            continue
        a = distinct[0]
        others = distinct[1:]
        points = np.unique(np.concatenate([o["_pts"] for o in distinct]), axis=0)
        merged = record_points(points, inv, a)
        merged["contrast_dL"] = (
            max(
                max(o["contrast_dL"] for o in distinct),
                support_contrast(contrast_field, points, ox, oy),
            )
            if CONFIG["merge_core_preservation"]
            else support_contrast(contrast_field, points, ox, oy)
        )
        merged["flags"] += ";clump;reviewed_merge:" + ",".join(o["id"] for o in others)
        merged["review_reason"] = pair["reason"]
        reason = boundary_reason(points, box)
        if reason is None and merged["contrast_dL"] < 6:
            reason = "below_contrast"
        for o in distinct:
            candidates.remove(o)
        if reason:
            merged["reason"] = reason
            rejected.append(merged)
        else:
            candidates.append(merged)
        for b in others:
            b["reason"] = "merged_duplicate"
            rejected.append(b)
            merge_audit.append(
                dict(kept=a["id"], suppressed=b["id"], gap_px=None, review_reason=pair["reason"])
            )
    ghost_audit = []
    raw = im.astype(np.float32).mean(2)
    hp = raw - cv2.GaussianBlur(raw, (0, 0), 12)
    for obj in (list(candidates) if CONFIG["ghost_screen"] else []):
        cx, cy = round(obj["x"]), round(obj["y"])
        if cx < 16 or cy < 24 or cx + 81 >= w or cy + 24 >= h:
            continue
        template = hp[cy - 15 : cy + 16, cx - 15 : cx + 16]
        search = hp[cy - 23 : cy + 24, cx + 45 - 15 : cx + 66 + 15]
        if template.std() < 0.15:
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(response)
        dx, dy = 45 + loc[0], loc[1] - 8
        point = cv2.transform(np.array([[[cx + dx, cy + dy]]], np.float32), M)[0, 0]
        owners = [
            o
            for o in candidates
            if o is not obj
            and np.linalg.norm(o["_pts"] - point, axis=1).min() <= 10
            and o["contrast_dL"] >= 1.7 * obj["contrast_dL"]
        ]
        if score >= 0.75 and owners:
            candidates.remove(obj)
            obj["reason"] = "ghost"
            obj["review_reason"] = "Faint same-shape copy left of sharper body"
            rejected.append(obj)
            ghost_audit.append(
                dict(id=obj["id"], owner=owners[0]["id"], offset_x=dx, offset_y=dy, ncc=score)
            )
    (out / "ghost_audit.json").write_text(json.dumps(ghost_audit, indent=2))
    extra_merges, near_pairs = consolidate(candidates, rejected, inv, contrast_field, ox, oy, box)
    merge_audit += extra_merges
    visible_body_overrides = []
    for i, entry in enumerate(correction_data.get("visible_body_overrides", []), 1):
        if not entry.get("reason") or not entry.get("evidence_crop") or not entry.get("polygon"):
            raise ValueError("Visible-body override needs reason, evidence_crop and polygon")
        evidence = Path(entry["evidence_crop"])
        if not evidence.is_file():
            evidence = Path(corrections).parent / evidence
        if not evidence.is_file():
            raise ValueError("Missing visible-body evidence crop")
        mask = np.zeros(rect.shape[:2], np.uint8)
        points = (
            cv2.transform(np.array([entry["polygon"]], np.float32), M)[0].round().astype(np.int32)
        )
        cv2.fillPoly(mask, [points], 1)
        yy, xx = np.where(mask)
        obj = object_record(
            mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1],
            (xx.min(), yy.min()),
            inv,
            "visible_body_override",
            entry["class"],
        )
        if entry["class"] not in ("intact", "broken"):
            raise ValueError("Override class must be intact or broken")
        obj.update(
            id=f"V{i:04d}",
            flags="low_evidence;visible_body_override",
            review_reason=entry["reason"],
        )
        obj["contrast_dL"] = support_contrast(contrast_field, obj["_pts"], ox, oy)
        reason = boundary_reason(obj["_pts"], box)
        if reason is None and obj["area_px"] < MIN_AREA:
            reason = "small_area"
        overlap = any(cKDTree(o["_pts"]).query(obj["_pts"])[0].min() <= 0 for o in candidates)
        status = reason or ("counted_owner_overlap" if overlap else "counted_visible_override")
        if reason:
            obj["reason"] = reason
            rejected.append(obj)
        elif not overlap:
            candidates.append(obj)
        record = dict(
            entry,
            id=obj["id"],
            status=status,
            measured_C=obj["contrast_dL"],
            measured_B=body_value(obj, body_field, ox, oy),
        )
        visible_body_overrides.append(record)
        correction_audit.append(dict(kind="visible_body_override", **record))

    all_ids = {o["id"]: o for o in candidates + rejected}
    for rec in correction_audit:
        if rec["kind"] == "add":
            target = rec.get("owner", rec["id"])
            while any(m.get("suppressed") == target for m in merge_audit):
                target = next(m["kept"] for m in merge_audit if m["suppressed"] == target)
            if target not in all_ids:
                if not CONFIG["audit_robustness"]:
                    raise ValueError("Addition audit lost identity: " + str(rec))
                rec.update(
                    final_id=None,
                    final_decision="audit_conflict",
                    conflict="Referenced support was removed by another correction",
                )
                continue
            rec["final_id"] = target
            rec["final_decision"] = all_ids[target].get("reason", "counted")
            addition = correction_data["add"][int(rec["id"][1:]) - 1]
            if "expected_decision" in addition:
                if not CONFIG["audit_robustness"]:
                    assert rec["final_decision"] == addition["expected_decision"]
                rec["expected_decision"] = addition["expected_decision"]
                rec["expectation_matches"] = rec["final_decision"] == addition["expected_decision"]
            if rec["final_decision"] == "counted" and target not in {o["id"] for o in candidates}:
                rec.update(final_decision="audit_conflict", conflict="Counted owner absent")

    (out / "correction_audit.json").write_text(json.dumps(correction_audit, indent=2))
    (out / "merge_audit.json").write_text(json.dumps(merge_audit, indent=2))
    pd.DataFrame(near_pairs, columns=["a", "b", "gap_px"]).to_csv(
        out / "near_pairs.csv", index=False
    )
    for obj in candidates + rejected:
        q = (obj["_pts"] - [ox, oy]).astype(int)
        xx = q[:, 0]
        yy = q[:, 1]
        valid = (xx >= 0) & (xx < clean.shape[1]) & (yy >= 0) & (yy < clean.shape[0])
        xx = xx[valid]
        yy = yy[valid]
        if not len(xx):
            continue
        dx = min(abs(obj["_pts"][:, 0].mean() - v) for v in np.linspace(x, X, 5))
        dy = min(abs(obj["_pts"][:, 1].mean() - v) for v in np.linspace(y, Y, 5))
        ww = np.ptp(xx) + 1
        hh = np.ptp(yy) + 1
        obj["junction_flag"] = bool(
            not CONFIG["virtual_grid"] and dx <= 25 and dy <= 25 and obj["area_px"] <= 120
        )
        obj["sliver_flag"] = bool(
            not CONFIG["virtual_grid"]
            and (
                (ww <= 6 and hh >= 2.5 * ww and dx <= 30)
                or (hh <= 6 and ww >= 2.5 * hh and dy <= 30)
            )
        )
        obj["sharpness"] = float(np.percentile(sharp_field[yy, xx], 90))
        obj["chromatic_body"] = bool(
            np.sum((clean[yy, xx] >= 5.5) & (stain[yy, xx] * 255 >= 5.5)) >= max(8, 0.25 * len(xx))
        )
    for obj in candidates + rejected:
        obj["body_response"] = body_value(obj, body_field, ox, oy)
    pair_ids = {r[k] for r in near_pairs for k in ["a", "b"]}
    candidate_sheet(im, [o for o in candidates if o["id"] in pair_ids], out / "near_pairs.png")
    eligible = candidates + [o for o in rejected if o.get("reason") == "below_contrast"]
    count_strict = sum(o["contrast_dL"] >= CONTRAST_THRESHOLD + 2 for o in eligible)
    count_inclusive = sum(o["contrast_dL"] >= CONTRAST_THRESHOLD - 2 for o in eligible)
    candidate_sheet(im, candidates + rejected, out / "candidates.png")
    geometry = []
    for obj in candidates + rejected:
        pts = obj["_pts"]
        obj["left_gap_px"] = float(max(x - pts[:, 0].max(), 0))
        obj["top_gap_px"] = float(max(y - pts[:, 1].max(), 0))
        obj["right_clearance_px"] = float(X - pts[:, 0].max())
        obj["bottom_clearance_px"] = float(Y - pts[:, 1].max())
        geometry.append(
            dict(
                id=obj["id"],
                rectified_contours=[
                    cv2.transform(ct.astype(np.float32)[None], M)[0].tolist()
                    for ct in obj["_contours"]
                ],
                original_contours=[ct.tolist() for ct in obj["_contours"]],
            )
        )
    (out / "candidate_masks.json").write_text(json.dumps(geometry), encoding="utf-8")
    scale = 50 / pitch
    for obj in candidates + rejected:
        obj["eq_diam_um_grid50"] = math.sqrt(4 * obj["area_px"] / math.pi) * scale
        obj["area_um2_meta"] = obj["area_px"] * 0.37**2
        obj["area_um2_grid50"] = obj["area_px"] * scale**2
    columns = [
        "id",
        "x",
        "y",
        "area_px",
        "eq_diam_um_meta",
        "eq_diam_um_grid50",
        "area_um2_meta",
        "area_um2_grid50",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
        "class",
        "flags",
        "source",
        "circularity",
        "solidity",
        "reason",
        "review_reason",
        "contrast_dL",
        "left_gap_px",
        "top_gap_px",
        "right_clearance_px",
        "bottom_clearance_px",
        "junction_flag",
        "sliver_flag",
        "chromatic_body",
        "sharpness",
        "body_response",
    ]
    if set3:
        columns += [
            "stain_score",
            "edge_gradient",
            "interior_std",
            "interior_variance",
            "minor_axis_px",
            "eccentricity",
            "focus_diameter_px",
            "structure_std",
        ]

    def export(items, path):
        pd.DataFrame(
            [{k: v for k, v in o.items() if not k.startswith("_")} for o in items]
        ).reindex(columns=columns).to_csv(path, index=False)

    export(candidates, out / "objects.csv")
    export(rejected, out / "rejected.csv")
    lo = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 30, 0)
    hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 31, [w, h])
    crop = im[lo[1] : hi[1], lo[0] : hi[0]].copy()
    overlay = crop.copy()
    if not CONFIG["virtual_grid"]:
        cv2.polylines(
            overlay, [(corners - lo).round().astype(np.int32)], True, (255, 180, 0), 1, cv2.LINE_AA
        )
    if CONFIG["virtual_grid"]:
        for target in (crop, overlay):
            guide = target.copy()
            for i in range(5):
                vx = int(round(x + i * pitch - lo[0]))
                vy = int(round(y + i * pitch - lo[1]))
                cv2.line(
                    guide,
                    (vx, int(y - lo[1])),
                    (vx, int(Y - lo[1])),
                    (200, 155, 75),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    guide,
                    (int(x - lo[0]), vy),
                    (int(X - lo[0]), vy),
                    (200, 155, 75),
                    1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(guide, 0.4, target, 0.6, 0, dst=target)
    codes = {
        "ghost": "G",
        "manual_click_outside_support": "Q",
        "merged_duplicate": "D",
        "manual_support_clipped": "Q",
        "out_of_group": "O",
        "edge_rule": "E",
        "not_stained": "S",
        "line_artefact": "L",
        "manual_remove": "M",
        "below_contrast": "C",
        "small_area": "S",
    }
    for o in rejected + candidates:
        reject = "reason" in o
        color = (150, 150, 150) if reject else CLASSES[o["class"]][::-1]
        cs = [ct - lo for ct in o["_contours"]]
        cv2.polylines(overlay, cs, True, color, 1, cv2.LINE_AA)
        px = int(o["x"] - lo[0])
        py = int(o["y"] - lo[1])
        txt = o["id"] + (":" + codes.get(o["reason"], o["reason"]) if reject else "")
        if -10 < px < overlay.shape[1] and -10 < py < overlay.shape[0]:
            pos = (px + 5, py - 4)
            cv2.putText(
                overlay, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA
            )
            cv2.putText(overlay, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    save_image(out / "crop.png", crop)
    save_image(out / "overlay.png", overlay)
    summary = dict(
        image=Path(image_path).name,
        grid_pitch_px=pitch,
        um_per_px_meta=0.37,
        um_per_px_grid50=scale,
        group_bbox_px=[
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        ],
        group_corners_px=corners.tolist(),
        group_rectified_bbox_px=box,
        group_visible_fraction=fraction,
        group_complete=bool(complete or fraction >= 0.99),
        group_fully_inside=bool(complete),
        group_selection_ambiguous=bool(
            not (complete or fraction >= 0.99) and len(diag[0].get("selection_candidates", [])) > 1
        ),
        count=len(candidates),
        count_strict=count_strict,
        count_inclusive=count_inclusive,
        contrast_threshold=CONTRAST_THRESHOLD,
        min_area_px=MIN_AREA,
        n_intact=sum(o["class"] == "intact" for o in candidates),
        n_broken=sum(o["class"] == "broken" for o in candidates),
        n_uncertain=sum(o["class"] == "uncertain" for o in candidates),
        n_clump=sum("clump" in o["flags"] for o in candidates),
        objects_per_mL=None if CONFIG["virtual_grid"] else len(candidates) * 250000 / fraction,
        density_assumption=(
            "No chamber depth: objects/mL unavailable; count per virtual square only."
            if CONFIG["virtual_grid"]
            else "0.2 mm x 0.2 mm x 0.1 mm = 4 nL; no dilution correction; normalized by visible fraction"
        ),
        median_eq_diam_um=(
            float(np.median([o["eq_diam_um_meta"] for o in candidates])) if candidates else None
        ),
        median_eq_diam_um_grid50=(
            float(np.median([o["eq_diam_um_grid50"] for o in candidates])) if candidates else None
        ),
        visible_body_overrides=visible_body_overrides,
        coverage_fraction=float(
            sum(
                np.sum(
                    (o["_pts"][:, 0] >= x)
                    & (o["_pts"][:, 0] < X)
                    & (o["_pts"][:, 1] >= y)
                    & (o["_pts"][:, 1] < Y)
                )
                for o in candidates
            )
            / ((X - x) * (Y - y))
        ),
        counter_confidence=0,
        counter_notes=correction_data.get(
            "notes",
            "Automatic first pass; count not visually verified. Shape labels are provisional.",
        ),
        round=(2 if set3 else 1) if CONFIG["virtual_grid"] else 6,
        config=CONFIG.copy(),
        body_response_threshold=BODY_THRESHOLD,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grid_detection_failed=False,
        grid_diagnostics=diag,
        crop_origin_px=lo.tolist(),
    )
    if set3:
        summary["admission_rule"] = (
            f"Set3 C90>=6, area>=8, stain_score90>={CONFIG['set3_seed']}, measured outline/structure gate; strict C>=8, inclusive C>=4. Legacy B exported for comparison only."
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corrections")
    ap.add_argument("--config", help="JSON configuration file")
    args = ap.parse_args()
    configure(json.loads(Path(args.config).read_text()) if args.config else None)
    print(json.dumps(count(args.image, args.out, args.corrections), indent=2))
