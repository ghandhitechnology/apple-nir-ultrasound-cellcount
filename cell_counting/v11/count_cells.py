# Command-line entry point of the v11 cell counter. It counts every image in the given files or
# folders and writes a summary table, one object table per image, and optional overlays and label
# images. Run it as: python count_cells.py <images or folders> --out <dir> [--overlay].

import argparse, csv, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

USAGE = """Count stained apple-flesh particles in microscope images.

  python count_cells.py IMAGES_OR_FOLDERS... --out OUT_DIR [options]

Examples
  python count_cells.py D:/set4/full --out D:/set4/counts --overlay
  python count_cells.py img1.jpg img2.jpg --out counts --area full
  python count_cells.py D:/set4/full --out counts --area 1372,532,2468,1628 --set faint_T=0

Writes
  OUT/summary.csv          one row per image: count, coverage (stained fraction of the area), objects in window, seconds
  OUT/objects/<image>.csv  one row per object >= 40 px: centroid, area, size, bbox, in_area, counted
  OUT/overlays/<image>.jpg (--overlay) counted objects green, others grey, counting area cyan
  OUT/labels/<image>.npz   (--labels) int32 label image of the processed window + its offset
  OUT/run.json             settings, versions and timing for the run
"""

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            out.append(p)
        else:
            raise SystemExit(f"not an image or folder: {p}")
    return out


def parse_set(items):
    cfg = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        v = v.strip()
        cfg[k.strip()] = (
            (v.lower() == "true")
            if v.lower() in ("true", "false")
            else (float(v) if any(c in v for c in ".e") else int(v))
        )
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="image files and/or folders of images")
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument(
        "--area",
        default="square",
        help="'square' (set-3 virtual 1096 px square, default), 'full', or x0,y0,x1,y1",
    )
    ap.add_argument(
        "--margin", type=int, default=64, help="context processed around the area, px (default 64)"
    )
    ap.add_argument(
        "--um-per-px",
        type=float,
        default=0.37,
        help="pixel size for um outputs (default 0.37, 10x objective + 0.5x adapter)",
    )
    ap.add_argument("--overlay", action="store_true", help="write overlay JPEGs")
    ap.add_argument(
        "--overlay-scale", type=float, default=0.5, help="overlay downscale factor (default 0.5)"
    )
    ap.add_argument("--labels", action="store_true", help="write label images (.npz)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default 8; Windows always runs in one process)",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        help="override engine parameters, e.g. faint_T=0 pp=false",
    )
    a = ap.parse_args()

    from cellcount import core
    from cellcount.count import count_image, overlay

    cfg = parse_set(a.set)
    if a.workers is not None:
        cfg["workers"] = a.workers
    core.configure(cfg)

    images = collect(a.inputs)
    out = Path(a.out)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    if a.overlay:
        (out / "overlays").mkdir(exist_ok=True)
    if a.labels:
        (out / "labels").mkdir(exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for i, img in enumerate(images, 1):
        res = count_image(img, a.area, a.margin, a.um_per_px, keep_labels=a.overlay or a.labels)
        with open(out / "objects" / f"{img.stem}.csv", "w", newline="", encoding="utf-8") as f:
            cols = [
                "id",
                "x",
                "y",
                "area_px",
                "area_um2",
                "eq_diam_um",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "in_area",
                "counted",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(res.objects)
        if a.overlay:
            overlay(img, res, out / "overlays" / f"{img.stem}.jpg", a.overlay_scale)
        if a.labels:
            import numpy as np

            np.savez_compressed(
                out / "labels" / f"{img.stem}.npz", labels=res.labels, window=np.array(res.window)
            )
        rows.append(
            dict(
                image=img.name,
                count=res.count,
                coverage=round(res.coverage, 5),
                objects_in_window=len(res.objects),
                area=",".join(map(str, res.area)),
                seconds=round(res.seconds, 3),
            )
        )
        print(
            f"[{i}/{len(images)}] {img.name}: {res.count} objects, coverage {100 * res.coverage:.2f}%, {res.seconds:.2f} s",
            flush=True,
        )
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "count", "coverage", "objects_in_window", "area", "seconds"]
        )
        w.writeheader()
        w.writerows(rows)
    import cv2, numpy, scipy, skimage

    (out / "run.json").write_text(
        json.dumps(
            dict(
                counter="cellcount v11",
                config=core.CFG,
                area=a.area,
                margin=a.margin,
                um_per_px=a.um_per_px,
                parallel=core.use_pool(),
                images=len(images),
                total_seconds=round(time.perf_counter() - t0, 2),
                python=platform.python_version(),
                platform=platform.platform(),
                versions=dict(
                    numpy=numpy.__version__,
                    scipy=scipy.__version__,
                    opencv=cv2.__version__,
                    skimage=skimage.__version__,
                ),
            ),
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"done: {len(images)} images -> {out}")


if __name__ == "__main__":
    main()
