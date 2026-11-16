# Tests whether ultrasound features add predictive value on top of NIR and fruit geometry on 70
# apples, with the feature families and settings fixed in advance. It adds a forward test: fit on
# apples 31-70 and score on 71-100. Run it as: python s04_models.py [n_permutations].

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1]
R0 = Path("C:/apple_project")
ALPHAS = [0.1, 1, 10, 100]
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
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
GEO = ["m", "volume_cm3"]
BASE = CH + GEO
FAM = {
    "US_level": ["level_rms_dB", "level_med_dB", "sat_frac"],
    "US_phase": ["gd_ms", "gd_rel_ms"],
    "US_core": ["level_med_dB", "sat_frac", "gd_rel_ms", "tilt_dB", "roughness_dB"],
    "US_shape8": ["level_rms_dB"] + [f"sh{j}" for j in range(8)],
    "US_legacy": ["insertion_proxy_db", "centroid_shift_kHz", "repeat_cv"],
    "AIR_control": ["air_level_rms_dB", "air_gd_ms", "air_B_rms_V"],
    "ORDER_control": ["acq_order"],
}
SEL = ["none", "US_level", "US_phase", "US_core", "US_shape8"]
US_COLS = sorted(set(sum([FAM[k] for k in SEL[1:] + ["US_legacy"]], [])))


def build(variant):
    lab = pd.read_csv(R0 / "datasets/본실험_031-100/analysis_data70.csv").set_index("apple")
    us = pd.read_csv(OUT / f"features_apple_level_{variant}.csv", index_col=0)
    for j in range(8):
        us[f"sh{j}"] = us[f"shape8_{j}"] - us.level_rms_dB
    d = lab.join(us, how="inner")
    assert len(d) == 70 and d.index.is_unique and d[BASE + US_COLS].notna().all().all()
    return d


def fit_pred(X, y, Xt, a):
    sc = StandardScaler().fit(X)
    return Ridge(alpha=a).fit(sc.transform(X), y).predict(sc.transform(Xt))


def inner_err(X, y, g, a):
    p = np.empty(len(y))
    for v in np.unique(g):
        m = g == v
        p[m] = fit_pred(X[~m], y[~m], X[m], a)
    return np.mean((y - p) ** 2)


def run_model(d, target, cols_of, folds):
    y = d[target].to_numpy()
    g = d.batch.to_numpy()
    yh = np.full(len(y), np.nan)
    ch = []
    mats = {k: d[c].to_numpy(float) for k, c in cols_of.items()}
    for train, test in folds:
        best = min(
            (inner_err(mats[k][train], y[train], g[train], a), i, a)
            for i, k in enumerate(mats)
            for a in ALPHAS
        )
        k = list(mats)[best[1]]
        yh[test] = fit_pred(mats[k][train], y[train], mats[k][test], best[2])
        ch.append((k, best[2]))
    return yh, ch


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def specs_all():
    s = {"mean": None, "NIR+geom": {"none": BASE}}
    for f in FAM:
        s[f"NIR+geom+{f}"] = {f: BASE + FAM[f]}
    for f in ["US_level", "US_phase", "US_core", "US_shape8", "US_legacy"]:
        s[f"{f}_alone"] = {f: FAM[f]}
    s["NIR+geom+US_selected"] = {k: BASE + (FAM[k] if k != "none" else []) for k in SEL}
    return s


def folds_for(d, scheme):
    g = d.batch.to_numpy()
    n = len(d)
    idx = np.arange(n)
    if scheme == "LOBO":
        return [(g != v, g == v) for v in np.unique(g)]
    if scheme == "LOO":
        return [(idx != i, idx == i) for i in range(n)]
    if scheme == "PROSPECTIVE":
        return [(d.index.to_numpy() <= 70, d.index.to_numpy() > 70)]
    raise ValueError(scheme)


def evaluate(
    d, tag, schemes, nperm=0, seed=0, perm_fams=("US_level", "US_core", "US_selected", "US_legacy")
):
    rows, preds, perm_rows = [], [], []
    g = d.batch.to_numpy()
    specs = specs_all()
    rng = np.random.default_rng(seed)
    for scheme in schemes:
        folds = folds_for(d, scheme)
        scored = np.zeros(len(d), bool)
        for _, te in folds:
            scored |= te
        for target in ["brix", "hardness"]:
            y = d[target].to_numpy()
            P = {}
            for name, cols in specs.items():
                if cols is None:
                    yh = np.full(len(y), np.nan)
                    ch = []
                    for tr, te in folds:
                        yh[te] = y[tr].mean()
                        ch.append(("", np.nan))
                else:
                    yh, ch = run_model(d, target, cols, folds)
                P[name] = yh
                chosen = np.empty(len(y), object)
                alpha = np.full(len(y), np.nan)
                for (tr, te), c in zip(folds, ch):
                    chosen[te] = c[0]
                    alpha[te] = c[1]
                preds += [
                    dict(
                        variant=tag,
                        scheme=scheme,
                        target=target,
                        model=name,
                        apple=int(d.index[i]),
                        batch=int(g[i]),
                        chosen_family=chosen[i],
                        alpha=alpha[i],
                        actual=y[i],
                        predicted=yh[i],
                    )
                    for i in np.flatnonzero(scored)
                ]
            ys = y[scored]
            e0 = (ys - P["NIR+geom"][scored]) ** 2
            rb = np.sqrt(e0.mean())
            Bi = rng.integers(0, len(ys), (2000, len(ys)))
            for name, yh in P.items():
                e1 = (ys - yh[scored]) ** 2
                r = np.sqrt(e1.mean())
                boot = np.sqrt(e1[Bi].mean(1)) - np.sqrt(e0[Bi].mean(1))
                rows.append(
                    dict(
                        variant=tag,
                        scheme=scheme,
                        target=target,
                        model=name,
                        n=len(ys),
                        RMSE=r,
                        R2=1 - e1.sum() / np.sum((ys - ys.mean()) ** 2),
                        dRMSE_vs_base=r - rb,
                        pct_vs_base=100 * (r - rb) / rb,
                        dRMSE_boot_lo=np.percentile(boot, 2.5),
                        dRMSE_boot_hi=np.percentile(boot, 97.5),
                        frac_apples_improved=np.mean(e1 < e0),
                    )
                )
            if nperm and scheme in ("LOBO", "PROSPECTIVE"):
                for fam in perm_fams:
                    obs = rmse(ys, P[f"NIR+geom+{fam}"][scored]) - rb
                    null = []
                    for _ in range(nperm):
                        perm = np.arange(len(y))
                        for v in np.unique(g):
                            ii = np.flatnonzero(g == v)
                            perm[ii] = rng.permutation(ii)
                        dp = d.copy()
                        dp[US_COLS] = d[US_COLS].to_numpy()[perm]
                        yh, _ = run_model(dp, target, specs[f"NIR+geom+{fam}"], folds)
                        null.append(rmse(ys, yh[scored]) - rb)
                    null = np.array(null)
                    perm_rows.append(
                        dict(
                            variant=tag,
                            scheme=scheme,
                            target=target,
                            family=fam,
                            observed_dRMSE=obs,
                            null_median=np.median(null),
                            null_p05=np.percentile(null, 5),
                            p_value=(1 + np.sum(null <= obs)) / (1 + nperm),
                            nperm=nperm,
                        )
                    )
                    print(tag, scheme, target, fam, "perm done", flush=True)
    return rows, preds, perm_rows


if __name__ == "__main__":
    allrows, allpreds, allperm = [], [], []
    plan = [
        ("primary", "primary", None, ("LOBO", "LOO", "PROSPECTIVE"), NPERM),
        ("originals_for_replaced", "originals_for_replaced", None, ("LOBO", "PROSPECTIVE"), 0),
        ("first4_all", "first4_all", None, ("LOBO",), 0),
        ("drop_replaced6", "primary", [31, 41, 48, 65, 69, 70], ("LOBO", "PROSPECTIVE"), 0),
    ]
    for tag, var, drop, schemes, nperm in plan:
        d = build(var)
        if drop:
            d = d.drop(drop)
        r, p, pr = evaluate(d, tag, schemes, nperm=nperm)
        allrows += r
        allpreds += p
        allperm += pr
        if tag == "primary":
            d.to_csv(OUT / "model_matrix_primary.csv")
        print(tag, "done", flush=True)
    S = pd.DataFrame(allrows)
    S.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(allpreds).to_csv(OUT / "heldout_predictions.csv", index=False)
    if allperm:
        pd.DataFrame(allperm).to_csv(OUT / "permutation_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(
        S[
            [
                "variant",
                "scheme",
                "target",
                "model",
                "n",
                "RMSE",
                "R2",
                "pct_vs_base",
                "dRMSE_boot_lo",
                "dRMSE_boot_hi",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    if allperm:
        print(pd.DataFrame(allperm).round(4).to_string(index=False))
