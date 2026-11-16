# Quality checks on the ultrasound recordings that use no labels. It covers recording settings by
# intake group, repeatability across the 70 apples, drift over the night and agreement for the six
# re-measured apples. It also writes the apple-level feature tables.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import SAT, icc_unbalanced

OUT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 220)
R = pd.read_csv(OUT / "features_run_level.csv")
z = np.load(OUT / "transfer_bins.npz")
H, Amed, H3 = z["H"], z["Amed"], z["H3"]
R["group"] = np.where(
    R.set == "superseded",
    "superseded originals",
    np.where(
        R.n_apple_runs == 4,
        "replacements (-2)",
        np.where(R.apple <= 70, "A31-A70 originals", "A71-A100 new"),
    ),
)
R["time"] = pd.to_datetime(R.zip_time)

chk = R.groupby(["group", "condition"]).agg(
    runs=("run", "size"),
    sweep_rate=("sweep_rate_Hz_per_ms", "median"),
    period_ms=("period_ms", "median"),
    fit_resid_Hz=("fit_resid_Hz", "median"),
    B_env=("B_env_median", "median"),
    B_rms=("B_rms_V", "median"),
    level_rms_dB=("level_rms_dB", "median"),
    level_sd=("level_rms_dB", "std"),
    sat_frac=("sat_frac", "median"),
    all_bins_sat=("sat_frac", lambda s: (s == 1).mean()),
    A_crest=("A_crest", "median"),
    pass_rep_dB=("pass_rep_dB", "median"),
    msc=("msc_weighted", "median"),
)
chk.to_csv(OUT / "recording_checks_by_group.csv")
print(chk.round(3).to_string())
odd = R[(R.sweep_rate_Hz_per_ms - 400).abs() > 5]
print("runs with sweep rate off by >5 Hz/ms:", len(odd), odd[["file", "run"]].values.tolist()[:10])

P = R.set == "primary"
ap_mask = (P & (R.condition == "apple")).to_numpy()
lv = Amed[ap_mask].ravel()
hv = 10 * np.log10(H3[ap_mask].ravel())
q = np.digitize(lv, np.arange(0, 0.3, 0.02))
cc = (
    pd.DataFrame(dict(level=lv, h3=hv, q=q))
    .groupby("q")
    .agg(level=("level", "median"), h3_med=("h3", "median"), n=("h3", "size"))
)
cc.to_csv(OUT / "compression_curve_apple_bins.csv")
print("\ncompression curve\n", cc.round(3).to_string())

ap = R[P & (R.condition == "apple")]
FEATS = [
    "level_med_dB",
    "level_rms_dB",
    "sat_frac",
    "gd_ms",
    "gd_rel_ms",
    "tilt_dB",
    "roughness_dB",
    "gd_lo_ms",
    "gd_hi_ms",
    "A_rms_V",
    "msc_weighted",
    "insertion_vs_air_dB",
]
rep = []
for c in FEATS:
    i70, sw, k0 = icc_unbalanced(ap, c)
    i40, _, _ = icc_unbalanced(ap[ap.apple <= 70], c)
    i30, _, _ = icc_unbalanced(ap[ap.apple > 70], c)
    ten = ap[ap.n_apple_runs == 8]
    pv = ten.pivot(index="apple", columns="apple_capture", values=c)
    h1, h2 = pv[[1, 2, 3, 4]].median(1), pv[[5, 6, 7, 8]].median(1)
    dev1 = pv[[1, 2, 3, 4]].sub(pv[[1, 2, 3, 4]].mean(1), axis=0).to_numpy().ravel()
    dev2 = pv[[5, 6, 7, 8]].sub(pv[[5, 6, 7, 8]].mean(1), axis=0).to_numpy().ravel()
    rep.append(
        dict(
            feature=c,
            ICC_single_70=i70,
            ICC_mean8_70=8 * i70 / (1 + 7 * i70),
            ICC_single_31_70=i40,
            ICC_single_71_100=i30,
            within_apple_sd=sw,
            between_apple_sd=ap.groupby("apple")[c].median().std(),
            split_half_r_64=np.corrcoef(h1, h2)[0, 1],
            spot_dev_corr_64=np.corrcoef(dev1, dev2)[0, 1],
        )
    )
rep = pd.DataFrame(rep)
rep.to_csv(OUT / "repeatability_icc.csv", index=False)
print("\nrepeatability\n", rep.round(3).to_string(index=False))

fl = (
    R[P]
    .groupby(["apple", "condition"])
    .agg(
        time=("time", "first"),
        level=("level_rms_dB", "median"),
        B_rms=("B_rms_V", "median"),
        A_rms=("A_rms_V", "median"),
        gd=("gd_ms", "median"),
    )
    .reset_index()
)
air = fl[fl.condition == "air"].sort_values("time")
app = fl[fl.condition == "apple"].sort_values("time")


def hrs(s):
    return (s - s.min()).dt.total_seconds() / 3600


for nm, df in [("air", air), ("apple", app)]:
    h = hrs(df.time)
    for c in ["level", "B_rms", "A_rms", "gd"]:
        r = np.corrcoef(h, df[c])[0, 1]
        print(
            f"drift {nm:5s} {c:6s} vs clock hour: r={r:+.2f}, range {df[c].min():.4f}..{df[c].max():.4f}"
        )
fl.to_csv(OUT / "file_level_by_time.csv", index=False)

both = R[
    (R.apple.isin([31, 41, 48, 65, 69, 70]))
    & (R.condition == "apple")
    & ((R.set == "superseded") | (R.n_apple_runs == 4))
]
rm = []
for a, g in both.groupby("apple"):
    o = g[g.set == "superseded"]
    n = g[g.set == "primary"]
    for c in ["level_rms_dB", "level_med_dB", "sat_frac", "gd_ms"]:
        rm.append(
            dict(
                apple=a,
                feature=c,
                original_median_8=o[c].median(),
                original_first4_median=o[o.apple_capture <= 4][c].median(),
                remeasure_median_4=n[c].median(),
                original_run_min=o[c].min(),
                original_run_max=o[c].max(),
                remeasure_run_min=n[c].min(),
                remeasure_run_max=n[c].max(),
                original_time=o.zip_time.iloc[0],
                remeasure_time=n.zip_time.iloc[0],
            )
        )
rm = pd.DataFrame(rm)
rm["difference"] = rm.remeasure_median_4 - rm.original_median_8
rm.to_csv(OUT / "remeasurement_agreement.csv", index=False)
print(
    "\nre-measurement\n",
    rm[rm.feature.isin(["level_rms_dB", "gd_ms"])].round(3).to_string(index=False),
)

ten = ap[ap.n_apple_runs == 8]
rng = np.random.default_rng(0)
dd = []
for a, g in ten.groupby("apple"):
    v = g.level_rms_dB.to_numpy()
    for _ in range(200):
        i = rng.permutation(8)
        dd.append(np.median(v[i[:4]]) - np.median(v))
print(
    "expected |4-run median - 8-run median| for level_rms_dB from resampling: median %.2f dB, 90th pct %.2f dB"
    % (np.median(np.abs(dd)), np.percentile(np.abs(dd), 90))
)
lv = rm[rm.feature == "level_rms_dB"]
print(
    "replaced apples: |remeasure - original| level median %.2f dB; corr of apple medians r=%.2f (n=6)"
    % (lv.difference.abs().median(), np.corrcoef(lv.original_median_8, lv.remeasure_median_4)[0, 1])
)

allorig = ap[(ap.apple <= 70)].copy()
allorig.loc[:, "dev_from_apple_median"] = allorig.level_rms_dB - allorig.groupby(
    "apple"
).level_rms_dB.transform("median")
supd = R[(R.set == "superseded") & (R.condition == "apple")].copy()
supd["dev"] = supd.level_rms_dB - supd.groupby("apple").level_rms_dB.transform("median")
all10 = ap[ap.n_apple_runs == 8].copy()
all10["dev"] = all10.level_rms_dB - all10.groupby("apple").level_rms_dB.transform("median")
pool = pd.concat(
    [
        all10[["apple", "run", "dev", "sat_frac"]].assign(src="primary-10run"),
        supd[["apple", "run", "dev", "sat_frac"]].assign(src="superseded"),
    ]
)
pool["abs_dev"] = pool.dev.abs()
out = pool.sort_values("abs_dev", ascending=False).head(15)
print(
    "\nlargest single-run deviations from own apple median (dB), primary 10-run files + superseded originals\n",
    out.round(2).to_string(index=False),
)
worst = pool.groupby(["src", "apple"]).abs_dev.max().sort_values(ascending=False)
print(
    "rank of superseded apples by their worst run among all 70 originals:",
    {
        a: int((worst.values >= worst.loc[("superseded", a)]).sum())
        for a in [31, 41, 48, 65, 69, 70]
    },
)

F = [
    "level_med_dB",
    "level_q25_dB",
    "level_q75_dB",
    "level_rms_dB",
    "sat_frac",
    "roughness_dB",
    "tilt_dB",
    "gd_ms",
    "gd_lo_ms",
    "gd_hi_ms",
    "gd_rel_ms",
    "h3_dB_linear_bins",
    "A_rms_V",
    "A_crest",
    "insertion_vs_air_dB",
    "msc_weighted",
] + [f"shape8_{j}" for j in range(8)]


def agg(df_ap, df_air):
    g = df_ap.groupby("apple")
    out = g[F].median()
    out["run_sd_level_dB"] = g.level_rms_dB.std()
    out["n_runs"] = g.size()
    for c in ["air_gd_ms", "air_level_rms_dB", "air_B_rms_V", "acq_order"]:
        out[c] = g[c].first()
    ga = df_air.groupby("apple")
    out["insertion_proxy_db"] = 20 * np.log10(ga.leg_H.mean() / g.leg_H.mean())
    out["centroid_shift_kHz"] = (g.leg_centroid.mean() - ga.leg_centroid.mean()) / 1000
    out["repeat_cv"] = g.leg_H.std() / g.leg_H.mean()
    return out


pr = R[P]
sup = R[R.set == "superseded"]
variants = {"primary": (pr[pr.condition == "apple"], pr[pr.condition == "air"])}
alt = pd.concat([pr[~pr.apple.isin(sup.apple.unique())], sup])
variants["originals_for_replaced"] = (alt[alt.condition == "apple"], alt[alt.condition == "air"])
f4 = pr[(pr.condition == "apple") & (pr.apple_capture <= 4)]
variants["first4_all"] = (f4, pr[pr.condition == "air"])
for k, (a_, r_) in variants.items():
    agg(a_, r_).to_csv(OUT / f"features_apple_level_{k}.csv")
print("\napple-level tables written:", list(variants))
