# Core of the final analysis: it builds the NIR and ultrasound feature blocks and predicts hardness,
# sensory-calibrated hardness and Brix from them. Each model is a ridge regression on one feature
# block, and the best average of one to three models is chosen inside each training fold, so no test
# apple influences its own prediction. It also reads the sensory panel ratings and turns them into a
# texture score per apple.

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/apple_project/output/hardness_hunt_20261113")
from harness import CH, _within_r, load_all, r2, snv

ROOT = Path(__file__).resolve().parents[1]
RAND = Path("C:/apple_project")
SENS_CSV = RAND / "output/sensory_cells_water_20261114/data/sensory_responses_fitted.csv"
SENS_APPLES = [37, 39, 40, 42, 49, 51, 63, 65, 69, 70, 81, 88, 93, 94, 99]
AL = np.logspace(-2, 4, 25)

D = load_all()
L = D.labels.copy()
APPLES = L.index.to_numpy()
B = D.batch
N = len(L)
C = {w: CH.index(f"{w}nm") for w in [560, 645, 680, 760]}

f = D.nir_files
lf = snv(np.log(f[CH].to_numpy()))
F = pd.DataFrame(lf, columns=CH)
F["apple"] = f.apple.values
F["angle"] = f.angle.values
ang = F.groupby(["apple", "angle"])[CH].mean().unstack("angle").loc[L.index]
A = np.stack([ang.xs(a, axis=1, level="angle")[CH].to_numpy() for a in [0, 90, 180, 270]], 1)


def srt(key):
    return np.take_along_axis(A, np.argsort(key, 1)[:, :, None], 1)


def ext(S):
    return np.hstack([S[:, 0], S[:, 3]])


S680 = srt(A[:, :, C[680]])
NIR = D.nir_snv.to_numpy()
NIR_BLOCKS = {
    "NIR mean SNV": NIR,
    "NIR angles sorted by 680": S680.reshape(N, -1),
    "NIR extreme angles (680)": ext(S680),
    "NIR extreme angles (680-645)": ext(srt(A[:, :, C[680]] - A[:, :, C[645]])),
    "NIR mean + 680 contrast": np.hstack([A.mean(1), S680[:, 3] - S680[:, 0]]),
}

U = D.us_apple
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
    + ["insertion_proxy_db", "centroid_shift_kHz", "repeat_cv"]
)
val = (
    pd.read_csv(RAND / "output/us_validation_20261114/apple_metrics.csv")
    .set_index("apple")
    .loc[L.index]
)
EN_COLS = ["envelope_dB", "local_gap_dB", "broad_coherent_dB", "broad_gap_dB"]
US_BASE = U[US_COLS].copy()
US_EN = pd.concat([US_BASE, val[EN_COLS]], axis=1)

runs = D.us_runs
ap = runs[runs.condition == "apple"]
Hm = 20 * np.log10(np.abs(D.us_bins["H"][ap.index.to_numpy()]) + 1e-12)
BINS = pd.DataFrame(Hm).assign(apple=ap.apple.values).groupby("apple").median().loc[L.index]
BINS.columns = [f"bin{i}" for i in range(32)]

RUN_COLS = [
    "level_rms_dB",
    "level_med_dB",
    "gd_ms",
    "gd_lo_ms",
    "gd_hi_ms",
    "tilt_dB",
    "roughness_dB",
    "msc_weighted",
    "A_rms_V",
    "A_crest",
] + [f"shape8_{i}" for i in range(8)]
good = (ap.sat_frac <= 0.25) & (ap.pass_rep_dB <= 1.5)


def _qmed(g):
    gg = g[good.loc[g.index]]
    return (gg if len(gg) >= 2 else g)[RUN_COLS].median()


US_Q = ap.groupby("apple").apply(_qmed).loc[L.index]
US_Q["n_good"] = good.groupby(ap.apple).sum().loc[L.index]

order = U.acq_order.to_numpy(float)


def detrend(X, keep_mean=True):
    X = np.asarray(X, float).copy()
    for g in np.unique(B):
        m = B == g
        o = order[m] - order[m].mean()
        mu = X[m].mean(0)
        slope = (o[:, None] * (X[m] - mu)).sum(0) / max((o**2).sum(), 1e-12)
        X[m] = X[m] - o[:, None] * slope - (0 if keep_mean else mu)
    return X


US_BLOCKS = {
    "US features (22)": US_BASE.to_numpy(float),
    "US features + energy ratios (26)": US_EN.to_numpy(float),
    "US transfer bins (32)": BINS.to_numpy(float),
    "US quality-filtered runs (18)": US_Q[RUN_COLS].to_numpy(float),
    "US detrended within lot (26)": detrend(US_EN.to_numpy(float), keep_mean=True),
    "US within-lot only (26)": detrend(US_EN.to_numpy(float), keep_mean=False),
    "US bins detrended within lot (32)": detrend(BINS.to_numpy(float), keep_mean=True),
    "US 38.5 kHz delay lead (1)": (
        U[["gd_ms"]].to_numpy(float) if "gd_lo_ms" not in U else U[["gd_lo_ms"]].to_numpy(float)
    ),
}
TEACH = np.column_stack([np.log(L.m), np.log(L.m) - np.log(L.volume_cm3)])


def _std(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1
    return mu, sd


def _teacher(Xs, z):
    X = Xs - Xs.mean(0)
    t = z - z.mean()
    best = None
    for a in [0.1, 1, 10, 100]:
        G = np.linalg.inv(X.T @ X + a * np.eye(X.shape[1]))
        h = np.einsum("ij,jk,ik->i", X, G, X) + 1 / len(t)
        e = (t - X @ (G @ X.T @ t)) / (1 - h)
        s = np.mean(e**2)
        if best is None or s < best[0]:
            best = (s, a)
    c = np.linalg.solve(X.T @ X + best[1] * np.eye(X.shape[1]), X.T @ t)
    sdv = (Xs @ c).std()
    return c / sdv if sdv > 0 else c


def fit_member(X, y, tr, guided=None):
    if guided is True:
        guided = TEACH
    if guided is False:
        guided = None
    mu, sd = _std(X[tr])
    Xs = (X[tr] - mu) / sd
    p = X.shape[1]
    if guided is not None:
        Z = guided
        Cm = np.column_stack([_teacher(Xs, Z[tr, j]) for j in range(Z.shape[1])])
        M = np.hstack([Xs, Xs @ Cm])
        k = Cm.shape[1]
    else:
        Cm = None
        M = Xs
        k = 0
    ym = y[tr].mean()
    yc = y[tr] - ym
    mM = M.mean(0)
    Mc = M - mM
    best = None
    for a in AL:
        pen = np.r_[np.full(p, a), np.full(k, 1e-3)]
        G = np.linalg.inv(Mc.T @ Mc + np.diag(pen))
        beta = G @ Mc.T @ yc
        h = np.einsum("ij,jk,ik->i", Mc, G, Mc) + 1 / len(yc)
        e = (yc - Mc @ beta) / (1 - h)
        s = np.mean(e**2)
        if best is None or s < best[0]:
            best = (s, a, beta, y[tr] - e)
    _, a, beta, loo = best
    w = beta[:p] + (Cm @ beta[p:] if guided is not None else 0)
    c0 = ym - mM @ beta

    def pred(R):
        return ((R - mu) / sd) @ w + c0

    return pred, loo, dict(mu=mu, sd=sd, w=w, c0=c0, alpha=a)


def _pcs(X, k=3):
    Xs = (X - X.mean(0)) / X.std(0)
    u, s_, vt = np.linalg.svd(Xs, full_matrices=False)
    return u[:, :k] * s_[:k]


def members(family, us_perm=None):
    q = (lambda X: X) if us_perm is None else (lambda X: np.asarray(X)[us_perm])
    US_BLOCKS_ = {k: q(v) for k, v in US_BLOCKS.items()}
    US_EN_ = q(US_EN.to_numpy(float))
    out = {}
    if family == "NIR (US teacher)":
        Zus = np.column_stack([TEACH, _pcs(US_EN_), _pcs(US_BLOCKS_["US transfer bins (32)"])])
        out["NIR mean SNV, guided by size+US"] = (NIR, Zus)
        out["NIR extreme angles (680), guided by size+US"] = (ext(S680), Zus)
        out["NIR angles sorted by 680, guided by size+US"] = (S680.reshape(N, -1), Zus)
        out["NIR mean SNV, guided by US only"] = (NIR, Zus[:, 2:])
        for n, X in NIR_BLOCKS.items():
            out[n] = (X, False)
        out["NIR mean SNV, guided by size"] = (NIR, True)
        out["NIR extreme angles (680), guided by size"] = (ext(S680), True)
        return out
    if family in ("NIR", "NIR+US"):
        for n, X in NIR_BLOCKS.items():
            out[n] = (X, False)
        out["NIR mean SNV, guided by size"] = (NIR, True)
        out["NIR extreme angles (680), guided by size"] = (ext(S680), True)
    if family in ("US", "NIR+US"):
        for n, X in US_BLOCKS_.items():
            out[n] = (X, False)
        out["US features + energy ratios, guided by size"] = (US_EN_, True)
    if family == "NIR+US":
        for nn, XN in [("NIR mean SNV", NIR), ("NIR angles sorted by 680", S680.reshape(N, -1))]:
            for un in [
                "US features + energy ratios (26)",
                "US detrended within lot (26)",
                "US within-lot only (26)",
                "US transfer bins (32)",
            ]:
                out[f"{nn} + {un}"] = (np.hstack([XN, US_BLOCKS_[un]]), False)
        out["NIR mean SNV + US features, guided by size"] = (np.hstack([NIR, US_EN_]), True)
    return out


def within_batch_perm(rng):
    p = np.arange(N)
    for g in np.unique(B):
        m = np.flatnonzero(B == g)
        p[m] = rng.permutation(m)
    return p


def nested(y, family, groups, max_k=3, pool=None):
    pool = pool if pool is not None else members(family)
    names = list(pool)
    P = np.empty(N)
    Ps = {m: np.empty(N) for m in names}
    chosen = []
    for gv in np.unique(groups):
        te = groups == gv
        tr = ~te
        if tr.sum() < 10:
            continue
        inner = {m: fit_member(pool[m][0], y, tr, pool[m][1])[:2] for m in names}
        best = None
        for k in range(1, max_k + 1):
            for combo in itertools.combinations(names, k):
                s = r2(y[tr], np.mean([inner[m][1] for m in combo], 0))
                if best is None or s > best[0]:
                    best = (s, combo)
        for m in names:
            Ps[m][te] = inner[m][0](pool[m][0][te])
        P[te] = np.mean([Ps[m][te] for m in best[1]], 0)
        chosen.append(best[1])
    return P, Ps, chosen


def score(y, p):
    return dict(
        R2=r2(y, p), RMSE=float(np.sqrt(np.mean((y - p) ** 2))), within_r=_within_r(y, p, B)
    )


def sensory_ratings():
    rows = list(csv.reader(open(SENS_CSV, encoding="utf-8", newline="")))[1:]
    rec = []
    for r in rows:
        rd = 1 if r[3].startswith("1") else 2
        for k in range(5):
            rec.append(
                dict(
                    panel=int(r[1]),
                    rnd=rd,
                    pos=k + 1,
                    code=int(r[4 + 4 * k]),
                    mealy=int(r[5 + 4 * k]),
                    juicy=int(r[6 + 4 * k]),
                    crisp=int(r[7 + 4 * k]),
                )
            )
    d = pd.DataFrame(rec)
    r1 = d[d.rnd == 1].copy()
    r2_ = d[d.rnd == 2].copy()
    m = r1.assign(rk=r1.groupby("panel").code.rank().astype(int)).set_index(["panel", "rk"]).code
    r1["apple"] = r1.code
    r2_["apple"] = [m[(p, c)] for p, c in zip(r2_.panel, r2_.code)]
    allr = pd.concat([r1, r2_])
    for a in ["mealy", "juicy", "crisp"]:
        allr[a + "_c"] = allr[a] - allr.groupby(["panel", "rnd"])[a].transform("mean")
    return allr


def texture_index(allr, which=None):
    x = allr if which is None else allr[allr.rnd == which]
    g = x.groupby("apple")[["mealy_c", "juicy_c", "crisp_c"]].mean()
    z = (g - g.mean()) / g.std()
    z["mealy_c"] *= -1
    return z.mean(1).rename("texture")
