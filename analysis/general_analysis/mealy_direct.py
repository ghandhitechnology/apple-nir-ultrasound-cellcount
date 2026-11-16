# Predicts panel mealiness directly from NIR or ultrasound on the 15 tasted apples, with every
# selection step done inside the cross-validation fold. It reports a label-permutation p-value for
# each approach. It also checks whether the 560 nm link to mealiness is explained by dry matter.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[1]
T = pd.read_csv(OUT / "data_table.csv", index_col=0)
SENS = T.dropna(subset=["sens_mealy"]).index.to_numpy()
y = T.loc[SENS, "sens_mealy"].to_numpy(float)
b = T.loc[SENS, "batch"].to_numpy()
nir = [c for c in T.columns if c.startswith("nir_") and c.endswith("nm")]
us = [c for c in T.columns if c.startswith("us_")]
rng = np.random.default_rng(7)


def best_single(X, y, tr, te):
    r = [abs(np.corrcoef(X[tr, j], y[tr])[0, 1]) for j in range(X.shape[1])]
    j = int(np.nanargmax(r))
    c = np.polyfit(X[tr, j], y[tr], 1)
    return np.polyval(c, X[te, j]), j


def ridge(X, y, tr, te, als=np.logspace(-2, 3, 20)):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1
    Xs = (X[tr] - mu) / sd
    ym = y[tr].mean()
    yc = y[tr] - ym
    best = None
    for a in als:
        G = np.linalg.inv(Xs.T @ Xs + a * np.eye(X.shape[1]))
        h = np.einsum("ij,jk,ik->i", Xs, G, Xs) + 1 / len(yc)
        e = (yc - Xs @ (G @ Xs.T @ yc)) / (1 - h)
        s = np.mean(e**2)
        if best is None or s < best[0]:
            best = (s, a)
    w = np.linalg.solve(Xs.T @ Xs + best[1] * np.eye(X.shape[1]), Xs.T @ yc)
    return ((X[te] - mu) / sd) @ w + ym


def loo(X, y, fn):
    p = np.empty(len(y))
    picks = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        te = ~tr
        out = fn(X, y, tr, te)
        if isinstance(out, tuple):
            p[te] = out[0]
            picks.append(out[1])
        else:
            p[te] = out
    return p, picks


rows = []
for name, cols, fn in [
    ("NIR best single channel (chosen in fold)", nir, best_single),
    ("NIR ridge, 18 channels", nir, ridge),
    ("US best single feature (chosen in fold)", us, best_single),
    ("US ridge, 26 features", us, ridge),
]:
    X = T.loc[SENS, cols].to_numpy(float)
    p, picks = loo(X, y, fn)
    r0 = np.corrcoef(y, p)[0, 1]
    null = []
    for _ in range(1000):
        yp = rng.permutation(y)
        pp, _ = loo(X, yp, fn)
        null.append(np.corrcoef(yp, pp)[0, 1])
    pv = (np.sum(np.array(null) >= r0) + 1) / 1001
    pk = pd.Series([cols[j] for j in picks]).value_counts().head(3).to_dict() if picks else {}
    rows.append(
        dict(
            model=name,
            n=len(y),
            cv_r=round(r0, 3),
            cv_R2=round(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), 3),
            p_perm=round(pv, 4),
            chosen_in_folds=pk,
        )
    )
R = pd.DataFrame(rows)
R.to_csv(OUT / "mealiness_direct_models.csv", index=False, encoding="utf-8-sig")
print(R.to_string(index=False))

x560 = T.loc[SENS, "nir_560nm"].to_numpy(float)
dm = T.loc[SENS, "dry_matter"].to_numpy(float)


def res(a, c):
    return a - np.polyval(np.polyfit(c, a, 1), c)


print(
    "560 vs mealiness r %.2f; partial | dry matter %.2f"
    % (np.corrcoef(x560, y)[0, 1], np.corrcoef(res(x560, dm), res(y, dm))[0, 1])
)
print(
    "dry matter vs mealiness partial | 560: %.2f" % np.corrcoef(res(dm, x560), res(y, x560))[0, 1]
)
for t in ["dry_matter", "brix", "hardness", "puncture2"]:
    m = T[[t, "nir_560nm"]].dropna()
    print(f"70 apples: 560 nm vs {t}: r {np.corrcoef(m[t], m.nir_560nm)[0, 1]:.2f}")
