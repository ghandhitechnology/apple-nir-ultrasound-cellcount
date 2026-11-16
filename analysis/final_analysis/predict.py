# Applies the saved final models (final_models.json) to new apples. It needs the NIR table with all
# 12 files per apple, plus an ultrasound table for the models that use ultrasound. Run it as: python
# predict.py nir_files.csv [us_table.csv].

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = [
    f"{w}nm"
    for w in [
        410,
        435,
        460,
        485,
        510,
        535,
        560,
        585,
        610,
        645,
        680,
        705,
        730,
        760,
        810,
        860,
        900,
        940,
    ]
]
US_COLS = (
    [
        "level_rms_dB",
        "level_med_dB",
        "sat_frac",
        "gd_ms",
        "gd_rel_ms",
        "tilt_dB",
        "roughness_dB",
        "msc_weighted",
        "A_rms_V",
        "A_crest",
        "h3_dB_linear_bins",
    ]
    + [f"shape8_{i}" for i in range(8)]
    + [
        "insertion_proxy_db",
        "centroid_shift_kHz",
        "repeat_cv",
        "envelope_dB",
        "local_gap_dB",
        "broad_coherent_dB",
        "broad_gap_dB",
    ]
)
BIN_COLS = [f"bin{i}" for i in range(32)]


def _snv(M):
    M = np.asarray(M, float)
    return (M - M.mean(1, keepdims=True)) / M.std(1, ddof=1, keepdims=True)


def nir_blocks(nir_files):
    f = nir_files.copy()
    f["apple"] = f.apple.astype(int)
    apples = sorted(f.apple.unique())
    raw = f.groupby("apple")[CH].mean().loc[apples]
    lf = pd.DataFrame(_snv(np.log(f[CH].to_numpy(float))), columns=CH)
    lf["apple"] = f.apple.values
    lf["angle"] = f.angle.values
    ang = lf.groupby(["apple", "angle"])[CH].mean().unstack("angle").loc[apples]
    A = np.stack([ang.xs(a, axis=1, level="angle")[CH].to_numpy() for a in [0, 90, 180, 270]], 1)
    i680, i645 = CH.index("680nm"), CH.index("645nm")

    def srt(key):
        return np.take_along_axis(A, np.argsort(key, 1)[:, :, None], 1)

    def ext(S):
        return np.hstack([S[:, 0], S[:, 3]])

    S680 = srt(A[:, :, i680])
    n = len(apples)
    blocks = {
        "NIR mean SNV": _snv(raw.to_numpy(float)),
        "NIR angles sorted by 680": S680.reshape(n, -1),
        "NIR extreme angles (680)": ext(S680),
        "NIR extreme angles (680-645)": ext(srt(A[:, :, i680] - A[:, :, i645])),
        "NIR mean + 680 contrast": np.hstack([A.mean(1), S680[:, 3] - S680[:, 0]]),
    }
    return apples, blocks


def features(name, nb, us, apples):
    if " + US " in name:
        a, b = name.split(" + US ", 1)
        return np.hstack([features(a, nb, us, apples), features("US " + b, nb, us, apples)])
    if name in nb:
        return nb[name]
    if us is None:
        raise ValueError("ultrasound table needed for " + name)
    u = us.loc[apples]
    if name == "US features + energy ratios (26)":
        return u[US_COLS].to_numpy(float)
    if name == "US transfer bins (32)":
        return u[BIN_COLS].to_numpy(float)
    raise ValueError("feature block needs lot-level processing, not supported here: " + name)


def predict(nir_files, us_table=None, model_file=HERE / "final_models.json"):
    M = json.load(open(model_file, encoding="utf-8"))["models"]
    apples, nb = nir_blocks(nir_files)
    out = pd.DataFrame(index=pd.Index(apples, name="apple"))
    for key, m in M.items():
        try:
            parts = [
                (
                    (features(mm["features"], nb, us_table, apples) - np.array(mm["mu"]))
                    / np.array(mm["sd"])
                )
                @ np.array(mm["w"])
                + mm["c0"]
                for mm in m["members"]
            ]
        except ValueError:
            continue
        out[key] = np.mean(parts, 0)
    return out


if __name__ == "__main__":
    nf = pd.read_csv(sys.argv[1])
    us = pd.read_csv(sys.argv[2], index_col=0) if len(sys.argv) > 2 else None
    print(predict(nf, us).round(3).to_string())
