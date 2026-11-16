# Checks whether NIR agrees with water content (oven dry fraction) and Brix, as a plausibility check
# of the NIR data rather than a model for use. Drying was incomplete, so results are also shown
# within batch with the drying-day mean removed.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = Path(__file__).resolve().parents[1]
IN = Path(r"C:/apple_project/output/nir_guided_20261113/inputs")
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
d = pd.read_csv(IN / "analysis_data70.csv").set_index("apple").drop(columns="mass(after)")
w = pd.read_csv(IN / "labels_with_dry_mass.csv", index_col=0)
d["dry"] = (w["mass(after)"] / w["mass(before)"]).reindex(d.index)
d = d.dropna(subset=["dry"]).copy()
d["dry_w"] = d.dry - d.groupby("batch").dry.transform("mean")
d["brix_w"] = d.brix - d.groupby("batch").brix.transform("mean")
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


P("apples with water data:", len(d), list(d.index[[0, -1]]), "batches", sorted(d.batch.unique()))
P(
    "dry fraction range %.2f-%.2f; share of variance between batches %.0f%%"
    % (d.dry.min(), d.dry.max(), 100 * d.groupby("batch").dry.transform("mean").var() / d.dry.var())
)
P("\n== physical consistency: dry matter should rise with Brix (sugar is most of dry matter)")
for a, b in [("dry", "brix"), ("dry_w", "brix_w")]:
    P("%s vs %s: r %.2f p %.3f" % (a, b, *pearsonr(d[a], d[b])))
P(
    "dry vs hardness r %.2f, dry vs mass r %.2f"
    % (pearsonr(d.dry, d.hardness)[0], pearsonr(d.dry, d.m)[0])
)

X = d[CH].to_numpy(float)


def loo_ridge(X, y, alphas=(0.1, 1, 10, 100)):
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Xs = (X[tr] - mu) / sd
        ym = y[tr].mean()
        yc = y[tr] - ym
        best = None
        for a in alphas:
            G = np.linalg.inv(Xs.T @ Xs + a * np.eye(Xs.shape[1]))
            h = np.einsum("ij,jk,ik->i", Xs, G, Xs) + 1 / len(yc)
            e = (yc - Xs @ (G @ Xs.T @ yc)) / (1 - h)
            s = np.mean(e**2)
            if best is None or s < best[0]:
                best = (s, G)
        pred[i] = ((X[i] - mu) / sd) @ (best[1] @ Xs.T @ yc) + ym
    return pred


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


rng = np.random.default_rng(0)
P("\n== NIR -> target, each apple left out (ridge on 18 SNV channels)")
rows = []
d["dry_plus_brix"] = (d.dry - d.dry.mean()) / d.dry.std() + (d.brix - d.brix.mean()) / d.brix.std()
d["dry_plus_brix_w"] = (d.dry_w / d.dry_w.std()) + (d.brix_w / d.brix_w.std())
for t in ["brix", "dry", "dry_w", "brix_w", "dry_plus_brix", "dry_plus_brix_w"]:
    y = d[t].to_numpy(float)
    p = loo_ridge(X, y)
    null = [r2(yp, loo_ridge(X, yp)) for yp in (rng.permutation(y) for _ in range(100))]
    rows.append(
        dict(
            target=t,
            LOO_R2=r2(y, p),
            r=pearsonr(y, p)[0],
            perm_p=(1 + np.sum(np.array(null) >= r2(y, p))) / 101,
            null_95=np.percentile(null, 95),
        )
    )
    P(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "water_brix_nir_scores.csv", index=False)
P("\n== single channels vs dry fraction (within batch) and Brix (within batch)")
ch = pd.DataFrame(
    {
        c: [pearsonr(d[c], d.dry_w)[0], pearsonr(d[c], d.brix_w)[0], pearsonr(d[c], d.dry)[0]]
        for c in CH
    },
    index=["r_dry_within", "r_brix_within", "r_dry_raw"],
).T
P(ch.round(2).to_string())
ch.to_csv(OUT / "water_channel_correlations.csv")
P(
    "agreement of channel patterns (dry_within vs brix_within): r %.2f"
    % pearsonr(ch.r_dry_within, ch.r_brix_within)[0]
)
(OUT / "a03_water_nir_log.txt").write_text("\n".join(log), encoding="utf-8")
