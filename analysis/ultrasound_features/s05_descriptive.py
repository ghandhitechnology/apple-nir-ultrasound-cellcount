# Descriptive checks of ultrasound features against the labels on 70 apples, none of which feed
# model selection. It tests within-batch correlations with false-discovery control, whether signs
# seen in apples 31-70 repeat in 71-100, and one earlier lead on the 30 new apples. It also checks
# group delay and level against fruit size, and label reliability and drift.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

OUT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 220)
d = pd.read_csv(OUT / "model_matrix_primary.csv", index_col=0)
P = pd.read_csv(OUT / "heldout_predictions.csv").query("variant=='primary' and model=='NIR+geom'")
for t in ["brix", "hardness"]:
    p = P[(P.target == t) & (P.scheme == "LOBO")].set_index("apple")
    d[f"resid_{t}"] = p.actual - p.predicted
    q = P[(P.target == t) & (P.scheme == "PROSPECTIVE")].set_index("apple")
    d[f"presid_{t}"] = q.actual - q.predicted
d["dmean_mm"] = (d.d1 + d.d2) / 2
F = [
    "level_rms_dB",
    "level_med_dB",
    "sat_frac",
    "gd_ms",
    "gd_rel_ms",
    "tilt_dB",
    "roughness_dB",
    "h3_dB_linear_bins",
    "run_sd_level_dB",
    "msc_weighted",
    "insertion_proxy_db",
    "centroid_shift_kHz",
    "repeat_cv",
    "air_level_rms_dB",
    "air_gd_ms",
    "air_B_rms_V",
    "acq_order",
]
LAB = ["brix", "hardness", "resid_brix", "resid_hardness"]


def wb(df, c):
    return df[c] - df.groupby("batch")[c].transform("mean")


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty_like(p)
    n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rows = []
for f in F:
    for t in LAB + ["m", "dmean_mm"]:
        for coh, s in [("all70", d), ("31-70", d.loc[:70]), ("71-100", d.loc[71:])]:
            r, p = spearmanr(s[f], s[t])
            rw, pw = spearmanr(wb(s, f), wb(s, t))
            rows.append(
                dict(
                    feature=f,
                    vs=t,
                    cohort=coh,
                    n=len(s),
                    spearman=r,
                    p=p,
                    within_batch_spearman=rw,
                    within_batch_p=pw,
                )
            )
C = pd.DataFrame(rows)
m = (
    (C.cohort == "all70")
    & C.vs.isin(LAB)
    & ~C.feature.isin(["air_level_rms_dB", "air_gd_ms", "air_B_rms_V", "acq_order"])
)
C.loc[m, "within_batch_q_BH"] = bh(C.loc[m, "within_batch_p"])
C.to_csv(OUT / "descriptive_correlations.csv", index=False)
w = C[C.cohort == "all70"].pivot(index="feature", columns="vs", values="within_batch_spearman")
print("within-batch Spearman, 70 apples\n", w[LAB + ["m", "dmean_mm"]].round(2).to_string())
print("\nsmallest within-batch q (US features x 4 label targets, %d tests):" % m.sum())
print(
    C[m]
    .nsmallest(6, "within_batch_p")[
        ["feature", "vs", "within_batch_spearman", "within_batch_p", "within_batch_q_BH"]
    ]
    .round(4)
    .to_string(index=False)
)

rep = C[C.vs.isin(LAB) & C.cohort.isin(["31-70", "71-100"])].pivot_table(
    index=["feature", "vs"], columns="cohort", values="within_batch_spearman"
)
rep["same_sign"] = np.sign(rep["31-70"]) == np.sign(rep["71-100"])
print(
    "\nsign agreement 31-70 vs 71-100 (US features x labels): %d of %d"
    % (rep.same_sign.sum(), len(rep))
)
print(
    "corr of within-batch rho between halves: %.2f" % np.corrcoef(rep["31-70"], rep["71-100"])[0, 1]
)
rep.to_csv(OUT / "replication_31_70_vs_71_100.csv")

n30 = d.loc[71:]
r, p2 = spearmanr(wb(n30, "centroid_shift_kHz"), wb(n30, "presid_brix"))
p1 = p2 / 2 if r < 0 else 1 - p2 / 2
r40, _ = spearmanr(wb(d.loc[:70], "centroid_shift_kHz"), wb(d.loc[:70], "resid_brix"))
conf = dict(
    hypothesis="centroid_shift_kHz vs sugar residual, within batch, negative",
    rho_31_70_this_pipeline=r40,
    rho_71_100=r,
    p_one_sided=p1,
    n=len(n30),
)
print("\nconfirmatory:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in conf.items()})
pd.DataFrame([conf]).to_csv(OUT / "confirmatory_test.csv", index=False)

print("\nphysical sanity (Spearman, overall / within batch):")
for f in ["gd_ms", "level_rms_dB"]:
    for t in ["dmean_mm", "m", "h"]:
        for coh, s in [("31-70", d.loc[:70]), ("71-100", d.loc[71:]), ("all70", d)]:
            r, p = spearmanr(s[f], s[t])
            rw, pw = spearmanr(wb(s, f), wb(s, t))
            print(
                f"  {f:13s} vs {t:9s} {coh:7s}: {r:+.2f} (p={p:.3f}) / within {rw:+.2f} (p={pw:.3f})"
            )

lab = d
for nm, c1, c2 in [("brix", "BRIX1", "BRIX2"), ("hardness", "harditer1", "harditer2")]:
    for coh, s in [("31-70", lab.loc[:70]), ("71-100", lab.loc[71:]), ("all70", lab)]:
        x = s[[c1, c2]].dropna()
        rr = np.corrcoef(x[c1], x[c2])[0, 1]
        wbr = np.corrcoef(wb(x.join(s.batch), c1), wb(x.join(s.batch), c2))[0, 1]
        print(
            f"{nm:8s} {coh:7s} duplicate r={rr:.2f} (within batch {wbr:.2f}), mean reliability={2*rr/(1+rr):.2f}, within-pair SD={np.sqrt(np.mean((x[c1]-x[c2])**2)/2):.2f}, between SD={s[nm].std():.2f}, n={len(x)}"
        )
for t in ["brix", "hardness", "m"]:
    H, p = kruskal(*[g[t].to_numpy() for _, g in d.groupby("batch")])
    r, pr = spearmanr(d.acq_order, d[t])
    rw, pw = spearmanr(wb(d, "acq_order"), wb(d, t))
    print(
        f'{t:8s}: batch means {d.groupby("batch")[t].mean().round(2).tolist()}, Kruskal p={p:.2g}; vs acquisition order rho={r:+.2f} (p={pr:.2g}), within batch {rw:+.2f} (p={pw:.2g})'
    )
n = len(d)
print(
    "\nminimum |r| detectable at 80%% power, two-sided alpha .05, n=%d: %.2f"
    % (n, np.tanh((1.96 + 0.84) / np.sqrt(n - 3)))
)
