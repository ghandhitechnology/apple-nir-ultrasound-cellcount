# Draws the figures of the ultrasound report as PNG files.

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figs"
C1, C2, C3, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)
R = pd.read_csv(OUT / "features_run_level.csv")
R["time"] = pd.to_datetime(R.zip_time)
N = pd.read_csv(OUT / "noise_floor_run_level.csv")
N["time"] = pd.to_datetime(N.zip_time)


def grp(s):
    return np.where(
        s.n_apple_runs == 4, "replacements (-2)", np.where(s.apple <= 70, "A31–A70", "A71–A100")
    )


COL = {"A31–A70": C1, "A71–A100": C2, "replacements (-2)": C3}
t0 = pd.Timestamp("2026-11-11 19:00")


def hr(t):
    return (t - t0).dt.total_seconds() / 3600


ap = N[N.condition == "apple"].copy()
ap["g"] = grp(ap)
fa = ap.groupby("apple").agg(
    t=("time", "first"),
    pass_rep=("pass_rep_dB", "median"),
    lf=("A_noise_lt10k_dB", "median"),
    g=("g", "first"),
)
air = N[N.condition == "air"].copy()
air["g"] = grp(air)
fb = air.groupby("apple").agg(t=("time", "first"), B=("B_rms_V", "median"), g=("g", "first"))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for g, c in COL.items():
    s = fa[fa.g == g]
    ax[0].scatter(hr(s.t), s.pass_rep, s=22, c=c, label=g, edgecolor="white", linewidth=0.6)
    ax[1].scatter(hr(s.t), s.lf, s=22, c=c, edgecolor="white", linewidth=0.6)
    s = fb[fb.g == g]
    ax[2].scatter(hr(s.t), s.B, s=22, c=c, edgecolor="white", linewidth=0.6)
ax[0].set_title("Within-run noise (pass 1 vs pass 2) rises after midnight")
ax[0].set_ylabel("median |dB| between sweep passes, apple runs")
ax[1].set_title("Low-frequency (<10 kHz) noise on ch A, apple runs")
ax[1].set_ylabel("dB re 1 V²")
ax[2].set_title("Drive level (ch B RMS, air runs) creeps up")
ax[2].set_ylabel("V")
for a in ax:
    a.set_xlabel("hours after 19:00, 11 Nov (zip timestamp)")
ax[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "fig1_quality_over_night.png", dpi=130)
plt.close()

rep = (
    pd.read_csv(OUT / "repeatability_icc.csv")
    .set_index("feature")
    .loc[
        ["level_rms_dB", "sat_frac", "A_rms_V", "gd_ms", "msc_weighted", "roughness_dB", "tilt_dB"]
    ]
)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
y = np.arange(len(rep))
ax[0].barh(y + 0.2, rep.ICC_single_31_70, 0.38, color=C1, label="A31–A70")
ax[0].barh(y - 0.2, rep.ICC_single_71_100, 0.38, color=C2, label="A71–A100")
ax[0].set_yticks(y)
ax[0].set_yticklabels(rep.index)
ax[0].invert_yaxis()
ax[0].set_xlabel("ICC, single capture")
ax[0].legend(loc="lower right")
ax[0].set_title("Level repeatability roughly halves in A71–A100")
rr = R[
    (R.apple.isin([31, 41, 48, 65, 69, 70]))
    & (R.condition == "apple")
    & ((R.set == "superseded") | (R.n_apple_runs == 4))
]
for i, a in enumerate([31, 41, 48, 65, 69, 70]):
    o = rr[(rr.apple == a) & (rr.set == "superseded")].level_rms_dB
    n = rr[(rr.apple == a) & (rr.set == "primary")].level_rms_dB
    ax[1].scatter(
        np.full(len(o), i - 0.15), o, s=18, c=GREY, label="original (evening)" if i == 0 else None
    )
    ax[1].scatter(
        np.full(len(n), i + 0.15), n, s=18, c=C3, label="-2 re-measure (02:22)" if i == 0 else None
    )
    ax[1].plot([i - 0.3, i], [o.median()] * 2, c=INK, lw=2)
    ax[1].plot([i, i + 0.3], [n.median()] * 2, c=INK, lw=2)
ax[1].set_xticks(range(6))
ax[1].set_xticklabels([f"A{a}" for a in [31, 41, 48, 65, 69, 70]])
ax[1].set_ylabel("band level |A/B| (dB), each dot one capture")
ax[1].set_title("Replaced apples: original vs -2 (bar = median)")
ax[1].legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIG / "fig2_repeatability_remeasure.png", dpi=130)
plt.close()

S = pd.read_csv(OUT / "scores.csv")
S = S[S.variant == "primary"]
models = [
    "NIR+geom+US_level",
    "NIR+geom+US_phase",
    "NIR+geom+US_core",
    "NIR+geom+US_shape8",
    "NIR+geom+US_legacy",
    "NIR+geom+US_selected",
    "NIR+geom+AIR_control",
    "NIR+geom+ORDER_control",
]
lab = [m.replace("NIR+geom+", "+") for m in models]
fig, ax = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
for j, (sch, tg) in enumerate(
    [("LOBO", "brix"), ("LOBO", "hardness"), ("PROSPECTIVE", "brix"), ("PROSPECTIVE", "hardness")]
):
    s = S[(S.scheme == sch) & (S.target == tg)].set_index("model")
    b = s.loc["NIR+geom"]
    v = s.loc[models]
    pct = v.pct_vs_base
    lo = 100 * v.dRMSE_boot_lo / b.RMSE
    hi = 100 * v.dRMSE_boot_hi / b.RMSE
    cols = [GREY if "control" in m else C1 for m in models]
    ax[j].barh(range(len(models)), pct, color=cols, height=0.6)
    ax[j].errorbar(
        pct, range(len(models)), xerr=[pct - lo, hi - pct], fmt="none", ecolor=INK2, lw=1, capsize=2
    )
    ax[j].axvline(0, c=INK, lw=0.8)
    nm = "leave-one-batch-out, n=70" if sch == "LOBO" else "fit 31–70 → score 71–100"
    ax[j].set_title(f"{tg}: {nm}\nbase RMSE {b.RMSE:.2f}, R² {b.R2:.2f}", fontsize=9)
    ax[j].set_xlabel("RMSE change vs NIR+geom (%), 95% bootstrap")
ax[0].set_yticks(range(len(models)))
ax[0].set_yticklabels(lab)
ax[0].invert_yaxis()
fig.text(
    0.01,
    0.01,
    "grey = negative controls (air-only features, acquisition order); they contain no apple information",
    color=INK2,
    fontsize=8,
)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(FIG / "fig3_incremental_value.png", dpi=130)
plt.close()

rep = pd.read_csv(OUT / "replication_31_70_vs_71_100.csv")
d = pd.read_csv(OUT / "model_matrix_primary.csv", index_col=0)
P = (
    pd.read_csv(OUT / "heldout_predictions.csv")
    .query("variant=='primary' and model=='NIR+geom' and scheme=='PROSPECTIVE' and target=='brix'")
    .set_index("apple")
)
d["presid"] = P.actual - P.predicted
d["dmean"] = (d.d1 + d.d2) / 2


def wb(df, c):
    return df[c] - df.groupby("batch")[c].transform("mean")


fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].scatter(rep["31-70"], rep["71-100"], s=20, c=C1, edgecolor="white", linewidth=0.6)
ax[0].axhline(0, c=INK2, lw=0.6)
ax[0].axvline(0, c=INK2, lw=0.6)
ax[0].plot([-0.5, 0.5], [-0.5, 0.5], c=GREY, lw=0.8, ls="--")
ax[0].set_xlabel("within-batch Spearman, apples 31–70")
ax[0].set_ylabel("same, apples 71–100")
ax[0].set_title(
    f'Feature–label correlations do not replicate\n(r between halves = {np.corrcoef(rep["31-70"], rep["71-100"])[0, 1]:.2f}, {len(rep)} pairs)',
    fontsize=9,
)
n30 = d.loc[71:]
x, yv = wb(n30, "centroid_shift_kHz"), wb(n30, "presid")
r, _ = spearmanr(x, yv)
ax[1].scatter(x, yv, s=24, c=C2, edgecolor="white", linewidth=0.6)
ax[1].set_xlabel("centroid_shift_kHz (within batch)")
ax[1].set_ylabel("sugar residual of NIR+geom (°Bx, within batch)")
ax[1].set_title(
    f"Pre-stated lead tested on 30 new apples\nexpected ρ ≈ −0.45, observed ρ = {r:.2f}", fontsize=9
)
for sub, c, nm in [(d.loc[:70], C1, "A31–A70"), (d.loc[71:], C2, "A71–A100")]:
    rr_, _ = spearmanr(sub.dmean, sub.gd_ms)
    ax[2].scatter(
        sub.dmean,
        sub.gd_ms,
        s=22,
        c=c,
        edgecolor="white",
        linewidth=0.6,
        label=f"{nm}: ρ = {rr_:+.2f}",
    )
ax[2].set_xlabel("mean diameter (mm)")
ax[2].set_ylabel("group delay (ms, apple median)")
ax[2].legend(loc="lower right")
ax[2].set_title("Size check: delay tracks diameter only in the first half", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "fig4_replication.png", dpi=130)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for j, t in enumerate(["brix", "hardness"]):
    for b, g in d.groupby("batch"):
        c = C1 if b <= 6 else C2
        ax[j].scatter(
            np.full(len(g), b) + np.random.default_rng(b).uniform(-0.15, 0.15, len(g)),
            g[t],
            s=16,
            c=c,
            edgecolor="white",
            linewidth=0.5,
        )
        ax[j].plot([b - 0.3, b + 0.3], [g[t].mean()] * 2, c=INK, lw=2)
    ax[j].set_xticks(range(3, 10))
    ax[j].set_xticklabels([f"{10*b+1}–{10*b+10}" for b in range(3, 10)], rotation=30)
    ax[j].set_title(
        {
            "brix": "Sugar (°Bx) by batch",
            "hardness": "Hardness by batch: the new batches are much harder",
        }[t],
        fontsize=9,
    )
plt.tight_layout()
plt.savefig(FIG / "fig5_labels_by_batch.png", dpi=130)
plt.close()
print("figures written")
