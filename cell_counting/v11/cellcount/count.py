# Applies the counting rules on top of the segmentation: an object counts if part of it lies inside
# the counting area and it does not touch the bottom or right edge. It also measures coverage, the
# stained share of the area, and draws overlay images.

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import core

MIN_AREA_PX = 40
HALO = 2


@dataclass
class Result:
    image: str
    area: tuple
    window: tuple
    count: int
    coverage: float
    seconds: float
    objects: list = field(default_factory=list)
    labels: np.ndarray = None


def resolve_area(area, image_size=None):
    if area in (None, "full"):
        return None
    if area == "square":
        return core.SQUARE
    if isinstance(area, str):
        area = tuple(int(v) for v in area.split(","))
    if len(area) != 4 or area[2] <= area[0] or area[3] <= area[1]:
        raise ValueError(f"bad area {area}: expected x0,y0,x1,y1")
    return tuple(int(v) for v in area)


def count_image(image_path, area="square", margin=core.MARGIN, um_per_px=0.37, keep_labels=False):
    t = time.perf_counter()
    a = resolve_area(area)
    lab, window = core.segment_area(image_path, a, margin)
    seconds = time.perf_counter() - t
    if a is None:
        a = (0, 0, window[2], window[3])

    ids, sizes = np.unique(lab, return_counts=True)
    keep = ids[(ids > 0) & (sizes >= MIN_AREA_PX)]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    lab = lut[lab]
    n = len(keep)
    ax0, ay0, ax1, ay1 = a[0] - window[0], a[1] - window[1], a[2] - window[0], a[3] - window[1]
    h, w = lab.shape
    inside = set(
        np.unique(
            lab[max(0, ay0 - HALO) : min(h, ay1 + HALO), max(0, ax0 - HALO) : min(w, ax1 + HALO)]
        ).tolist()
    )
    edge = set(np.unique(lab[:, max(0, ax1 - HALO) :]).tolist()) | set(
        np.unique(lab[max(0, ay1 - HALO) :, :]).tolist()
    )
    counted = sorted((inside - edge) - {0})
    is_counted = np.zeros(n + 1, bool)
    is_counted[counted] = True
    sub = lab[max(0, ay0) : ay1, max(0, ax0) : ax1]
    coverage = float(is_counted[sub].sum()) / float((a[2] - a[0]) * (a[3] - a[1]))
    objs = []
    if n:
        idx = np.arange(1, n + 1)
        cm = ndi.center_of_mass(np.ones(lab.shape), lab, idx)
        ar = ndi.sum_labels(np.ones(lab.shape), lab, idx)
        sl = ndi.find_objects(lab)
        for i, c, ap, s in zip(idx, cm, ar, sl):
            objs.append(
                dict(
                    id=int(i),
                    x=round(window[0] + c[1], 1),
                    y=round(window[1] + c[0], 1),
                    area_px=int(ap),
                    area_um2=round(ap * um_per_px**2, 2),
                    eq_diam_um=round(float(np.sqrt(4 * ap / np.pi)) * um_per_px, 2),
                    bbox_x0=window[0] + s[1].start,
                    bbox_y0=window[1] + s[0].start,
                    bbox_x1=window[0] + s[1].stop,
                    bbox_y1=window[1] + s[0].stop,
                    in_area=bool(i in inside),
                    counted=bool(is_counted[i]),
                )
            )
    return Result(
        image=str(image_path),
        area=a,
        window=window,
        count=len(counted),
        coverage=coverage,
        seconds=seconds,
        objects=objs,
        labels=lab if keep_labels else None,
    )


def overlay(image_path, res, out_path, scale=1.0):
    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = res.window
    crop = im[y0:y1, x0:x1].copy()
    lab = res.labels
    counted = np.zeros(int(lab.max()) + 1, bool)
    for o in res.objects:
        counted[o["id"]] = o["counted"]
    edge = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)) != cv2.erode(
        lab.astype(np.float32), np.ones((3, 3), np.uint8)
    )
    edge &= lab > 0
    crop[edge & counted[lab]] = (60, 220, 60)
    crop[edge & ~counted[lab]] = (150, 150, 150)
    a = res.area
    cv2.rectangle(crop, (a[0] - x0, a[1] - y0), (a[2] - x0 - 1, a[3] - y0 - 1), (255, 220, 0), 2)
    cv2.putText(
        crop,
        f"{Path(res.image).stem}: {res.count} objects, coverage {100 * res.coverage:.1f}%",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        crop,
        f"{Path(res.image).stem}: {res.count} objects, coverage {100 * res.coverage:.1f}%",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
