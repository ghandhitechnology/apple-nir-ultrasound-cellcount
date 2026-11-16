# Looks at why pass-to-pass noise is higher in apples 71-100, without using labels. It compares the
# out-of-band noise floor per run, the spread between and within apples by intake group, and noise
# against signal level.

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

OUT = Path(__file__).resolve().parents[1]
R0 = Path("C:/apple_project")
FS = 1 / 3.1e-6
R = pd.read_csv(OUT / "features_run_level.csv")
R["group"] = np.where(
    R.set == "superseded",
    "superseded",
    np.where(R.n_apple_runs == 4, "replacement", np.where(R.apple <= 70, "31-70", "71-100")),
)
prim = pd.read_csv(
    R0 / "datasets/본실험_031-100/ultrasound/primary_recordings.csv", keep_default_na=False
)
rows = []
for r in prim.itertuples():
    p = R0 / r.source.replace(chr(92), "/")
    raw = zipfile.ZipFile(p).read(r.archive_member) if r.archive_member else p.read_bytes()
    d = pd.read_csv(io.BytesIO(raw))
    for k in range(1, int(r.runs) + 1):
        a = d[f"Voltage, Ch A (V) Run #{k}"].to_numpy(float)
        b = d[f"Voltage, Ch B (V) Run #{k}"].to_numpy(float)
        f, pa = welch(a - a.mean(), fs=FS, nperseg=2048)
        _, pb = welch(b - b.mean(), fs=FS, nperseg=2048)

        def bandpow(p, lo, hi):
            return 10 * np.log10(np.sum(p[(f >= lo) & (f < hi)]) * (f[1] - f[0]))

        rows.append(
            dict(
                apple=int(r.apple),
                run=k,
                A_noise_55_75k_dB=bandpow(pa, 55e3, 75e3),
                A_noise_lt10k_dB=bandpow(pa, 1e3, 10e3),
                A_noise_140_160k_dB=bandpow(pa, 140e3, 160e3),
                B_noise_55_75k_dB=bandpow(pb, 55e3, 75e3),
                A_unique_levels=len(np.unique(a)),
            )
        )
N = pd.DataFrame(rows).merge(R[R.set == "primary"], on=["apple", "run"])
N.to_csv(OUT / "noise_floor_run_level.csv", index=False)
pd.set_option("display.width", 220)
print(
    N.groupby(["group", "condition"])[
        [
            "A_noise_55_75k_dB",
            "A_noise_lt10k_dB",
            "A_noise_140_160k_dB",
            "B_noise_55_75k_dB",
            "pass_rep_dB",
            "level_rms_dB",
            "A_unique_levels",
        ]
    ]
    .median()
    .round(2)
    .to_string()
)
ap = N[N.condition == "apple"]
print(
    "\npass_rep vs level (apple runs): corr %.2f"
    % ap[["pass_rep_dB", "level_rms_dB"]].corr().iloc[0, 1]
)
print(
    "pass_rep vs A noise 55-75k: corr %.2f"
    % ap[["pass_rep_dB", "A_noise_55_75k_dB"]].corr().iloc[0, 1]
)

for g, s in ap.groupby("group"):
    m = s[(s.level_rms_dB > -34) & (s.level_rms_dB < -30)]
    print(f"{g:12s} level -34..-30 dB: n={len(m)}, pass_rep median {m.pass_rep_dB.median():.2f} dB")

fa = ap.groupby("apple").agg(
    pass_rep=("pass_rep_dB", "median"),
    noise=("A_noise_55_75k_dB", "median"),
    order=("acq_order", "first"),
    group=("group", "first"),
)
print("\npass_rep by acquisition order (first/last 10 of the night):")
print(fa.sort_values("order").pass_rep.round(2).to_string().replace("\n", " | ")[:1200])

for g, s in ap.groupby("group"):
    gm = s.groupby("apple").level_rms_dB
    print(
        f"{g:12s} between-apple SD of medians {gm.median().std():.2f} dB, mean within-apple SD {gm.std().mean():.2f} dB, n apples {gm.ngroups}"
    )
