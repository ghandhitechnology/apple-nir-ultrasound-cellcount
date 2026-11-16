# Checks that the generalised transform reproduces the earlier 40-apple features exactly on the
# original files of apples 31-70.

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
R = Path("C:/apple_project")
old = pd.read_csv(R / "output/opus_deep_analysis_20261113/ultrasound/features_run_level.csv")
new = pd.read_csv(OUT / "features_run_level.csv")
orig = new[
    ((new.set == "primary") & (new.apple <= 70) & (new.n_apple_runs == 8))
    | (new.set == "superseded")
]
m = old.merge(orig, on=["apple", "run"], suffixes=("_old", "_new"))
out = []
for c in [
    "level_rms_dB",
    "level_med_dB",
    "sat_frac",
    "gd_ms",
    "gd_rel_ms",
    "tilt_dB",
    "A_rms_V",
    "B_rms_V",
    "msc_weighted",
]:
    dif = (m[c + "_old"] - m[c + "_new"]).abs()
    out.append(dict(feature=c, runs=len(m), max_abs_diff=dif.max()))
leg_old = pd.read_csv(R / "output/full_analysis_20261113/ultrasound_apples40.csv", index_col=0)
a = orig[orig.condition == "apple"].groupby("apple")
r = orig[orig.condition == "air"].groupby("apple")
leg = pd.DataFrame(
    dict(
        insertion_proxy_db=20 * np.log10(r.leg_H.mean() / a.leg_H.mean()),
        centroid_shift_kHz=(a.leg_centroid.mean() - r.leg_centroid.mean()) / 1000,
        repeat_cv=a.leg_H.std() / a.leg_H.mean(),
    )
)
for c in leg:
    out.append(
        dict(
            feature="legacy:" + c,
            runs=len(leg),
            max_abs_diff=(leg[c] - leg_old.loc[leg.index, c]).abs().max(),
        )
    )
out = pd.DataFrame(out)
out.to_csv(OUT / "reproduction_check.csv", index=False)
print(out.to_string(index=False))
