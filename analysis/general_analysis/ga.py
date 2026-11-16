# General analysis: how well NIR and ultrasound predict sweetness (Brix), hardness and mealiness.
# The sensory panel, dry matter, fruit size, cell counts and measurement lot are used only to
# calibrate, following published methods. It writes the correlation table, reliabilities and model
# results.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[1]
RAND = Path("C:/apple_project")
FA = RAND / "output/final_analysis_20261114"
sys.path.insert(0, str(FA / "scripts"))
import core
from core import (
    APPLES,
    CH,
    NIR,
    S680,
    SENS_APPLES,
    US_EN,
    A,
    B,
    L,
    N,
    _within_r,
    ext,
    fit_member,
    members,
    nested,
    r2,
    sensory_ratings,
    snv,
    srt,
)
from core import (
    C as CI,
)

rng = np.random.default_rng(20261114)

moist = (
    pd.read_csv(RAND / "output/moisture_second_dry_20261114/moisture_second_dry.csv")
    .set_index("apple")
    .loc[APPLES]
)
V = pd.DataFrame(index=pd.Index(APPLES, name="apple"))
V["batch"] = B
V["brix"] = L.brix
V["brix1"] = L.BRIX1
V["brix2"] = L.BRIX2
V["hardness"] = L.hardness
V["puncture1"] = L.harditer1
V["puncture2"] = L.harditer2
V["dry_matter"] = moist.dry_matter_pct
V["mass"] = L.m
V["volume"] = L.volume_cm3
V["density"] = L.m / L.volume_cm3

allr = sensory_ratings()
for a in ["mealy", "juicy", "crisp"]:
    g = allr.groupby(["panel", "rnd"])[a]
    allr[a + "_z"] = (allr[a] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    allr[a + "_z"] = allr[a + "_z"].fillna(0)
S = allr.groupby("apple")[["mealy_z", "juicy_z", "crisp_z"]].mean()
for a in ["mealy", "juicy", "crisp"]:
    V[f"sens_{a}"] = S[f"{a}_z"]
S_r = {
    rd: allr[allr.rnd == rd].groupby("apple")[["mealy_z", "juicy_z", "crisp_z"]].mean()
    for rd in (1, 2)
}

cells = pd.read_csv(RAND / "output/cells_main_20261114/cell_counts_per_apple.csv").set_index(
    "apple"
)
for c in ["cell_count", "intact_share", "coverage", "v13b_fullframe_count"]:
    V[c] = cells[c]
cimg = pd.read_csv(RAND / "output/cells_main_20261114/cell_counts_per_image.csv")

NIRV = pd.DataFrame(NIR, index=V.index, columns=[f"nir_{c}" for c in CH])
NIRV["nir_680_angle_range"] = S680[:, 3, CI[680]] - S680[:, 0, CI[680]]
d68 = A[:, :, CI[680]] - A[:, :, CI[645]]
NIRV["nir_680m645_angle_range"] = d68.max(1) - d68.min(1)

X = NIR - NIR.mean(0)
z = np.log(L.m.to_numpy())
z = (z - z.mean()) / z.std()
dvec = X.T @ z / len(z)
P = dvec / np.linalg.norm(dvec)
NIR_EPO = NIR - np.outer(X @ P, P)
NIRE = pd.DataFrame(NIR_EPO, index=V.index, columns=[f"nirEPO_{c}" for c in CH])
USV = US_EN.copy()
USV.index = V.index
USV.columns = [f"us_{c}" for c in USV.columns]


def sb(r, k=2):
    return k * r / (1 + (k - 1) * r) if r > 0 else np.nan


REL = {}
REL["brix"] = sb(np.corrcoef(V.brix1, V.brix2)[0, 1])
ok = V[["puncture1", "puncture2"]].dropna()
r12 = np.corrcoef(ok.puncture1, ok.puncture2)[0, 1]
REL["hardness"] = sb(r12)
REL["puncture2"] = r12
for a in ["mealy", "juicy", "crisp"]:
    REL[f"sens_{a}"] = sb(
        np.corrcoef(S_r[1].loc[SENS_APPLES, f"{a}_z"], S_r[2].loc[SENS_APPLES, f"{a}_z"])[0, 1]
    )
w = cimg.pivot(index="apple", columns="field", values="cell_count")
REL["cell_count"] = sb(np.corrcoef(w[1], w[2])[0, 1])
w = cimg.pivot(index="apple", columns="field", values="intact_share")
REL["intact_share"] = sb(np.corrcoef(w[1], w[2])[0, 1])
w = cimg.pivot(index="apple", columns="field", values="coverage")
REL["coverage"] = sb(np.corrcoef(w[1], w[2])[0, 1])

q = 0.1 / np.sqrt(12)
err = 100 * np.sqrt(
    (q / moist["mass(before)"]) ** 2 + (q * moist["mass(after2)"] / moist["mass(before)"] ** 2) ** 2
)
REL["dry_matter"] = 1 - (err**2).mean() / V.dry_matter.var()
REL["mass"] = REL["volume"] = REL["density"] = np.nan

f = core.D.nir_files
lf = pd.DataFrame(snv(np.log(f[CH].to_numpy())), columns=CH)
lf["apple"] = f.apple.values
for c in CH:
    g = lf.groupby("apple")[c]
    k = g.size().mean()
    msb = k * g.mean().var(ddof=1)
    msw = g.var(ddof=1).mean()
    REL[f"nir_{c}"] = max((msb - msw) / msb, 0)
    REL[f"nirEPO_{c}"] = REL[f"nir_{c}"]

icc = pd.read_csv(RAND / "output/us70_20261113/repeatability_icc.csv").set_index("feature")[
    "ICC_mean8_70"
]
for c in US_EN.columns:
    REL[f"us_{c}"] = icc.get(c, np.nan)
pd.Series(REL, name="reliability").round(3).to_csv(OUT / "reliabilities.csv", encoding="utf-8-sig")


def within(x, g):
    x = pd.Series(np.asarray(x, float))
    return (x - x.groupby(np.asarray(g)).transform("mean")).to_numpy()


def pair(x, y, name_x, name_y, family):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[m], y[m], B[m]
    n = int(m.sum())
    r, p = stats.pearsonr(x, y)
    rho = stats.spearmanr(x, y)[0]
    zz = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    xw, yw = within(x, g), within(y, g)
    nb = len(np.unique(g))
    dfw = n - 1 - nb
    rw = np.corrcoef(xw, yw)[0, 1] if xw.std() > 0 and yw.std() > 0 else np.nan
    tw = rw * np.sqrt(dfw / max(1e-12, 1 - rw**2)) if dfw > 0 and not np.isnan(rw) else np.nan
    pw = 2 * stats.t.sf(abs(tw), dfw) if dfw > 0 and not np.isnan(tw) else np.nan
    rx, ry = REL.get(name_x, np.nan), REL.get(name_y, np.nan)
    rc = r / np.sqrt(rx * ry) if (rx > 0 and ry > 0) else np.nan
    return dict(
        family=family,
        x=name_x,
        y=name_y,
        n=n,
        r=r,
        ci_lo=np.tanh(zz - 1.96 * se),
        ci_hi=np.tanh(zz + 1.96 * se),
        p=p,
        spearman=rho,
        r_within_lot=rw,
        p_within_lot=pw,
        df_within=dfw,
        r_disattenuated=rc,
        rel_x=rx,
        rel_y=ry,
    )


OUTS = ["brix", "hardness", "puncture2", "sens_mealy"]
AUX = [
    "dry_matter",
    "mass",
    "density",
    "sens_juicy",
    "sens_crisp",
    "cell_count",
    "intact_share",
    "coverage",
    "v13b_fullframe_count",
]
TAB = pd.concat([V, NIRV, NIRE, USV], axis=1)
rows = []
for fam, xs in [
    ("NIR -> output", list(NIRV.columns)),
    ("NIR (size removed, EPO) -> output", list(NIRE.columns)),
    ("US -> output", list(USV.columns)),
    ("other data -> output", AUX),
]:
    for xn in xs:
        for yn in OUTS:
            rows.append(pair(TAB[xn].to_numpy(float), TAB[yn].to_numpy(float), xn, yn, fam))
for fam, xs in [("NIR -> other data", list(NIRV.columns)), ("US -> other data", list(USV.columns))]:
    for xn in xs:
        for yn in ["dry_matter", "mass", "density"]:
            rows.append(pair(TAB[xn].to_numpy(float), TAB[yn].to_numpy(float), xn, yn, fam))
for i, a in enumerate(OUTS + AUX):
    for b in (OUTS + AUX)[i + 1 :]:
        if a in OUTS or b in OUTS:
            rows.append(
                pair(
                    TAB[a].to_numpy(float), TAB[b].to_numpy(float), a, b, "output <-> output/other"
                )
            )
R = pd.DataFrame(rows)
R["q_fdr"] = np.nan
for fam, g in R.groupby("family"):
    R.loc[g.index, "q_fdr"] = stats.false_discovery_control(g.p.to_numpy(), method="bh")
R["q_fdr_within"] = np.nan
for fam, g in R.groupby("family"):
    pw = g.p_within_lot.fillna(1).to_numpy()
    R.loc[g.index, "q_fdr_within"] = stats.false_discovery_control(pw, method="bh")
R["strong"] = (R.r.abs() >= 0.6) | (R.r_within_lot.abs() >= 0.6)
R.round(4).to_csv(OUT / "correlation_screen.csv", index=False, encoding="utf-8-sig")

HP = pd.read_csv(FA / "heldout_predictions.csv")


def cvr(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return np.corrcoef(y[m], p[m])[0, 1]


M = []
for t, lab in [("brix", "brix"), ("hardness", "hardness")]:
    for fam in ["NIR", "US", "NIR+US"]:
        h = (
            HP[
                (HP.target == t)
                & (HP.family == fam)
                & (HP.scheme == "LOO")
                & HP.model.str.startswith("NESTED")
            ]
            .set_index("apple")
            .loc[APPLES]
        )
        y, p = h.actual.to_numpy(), h.predicted.to_numpy()
        M.append(
            dict(
                output=lab,
                inputs=fam,
                calibration="SNV, angle sorting, size teacher (final analysis 11-14)",
                n=N,
                cv_r=cvr(y, p),
                cv_R2=r2(y, p),
                within_lot_r=_within_r(y, p, B),
                source="final_analysis_20261114 heldout",
            )
        )

y = V.brix.to_numpy(float)
DMZ = np.column_stack([core.TEACH, V.dry_matter.to_numpy(float)])
pool_dm = {
    "NIR mean SNV, guided by size+DM": (NIR, DMZ),
    "NIR mean SNV, guided by DM": (NIR, DMZ[:, 2:]),
    "NIR extreme angles (680), guided by size+DM": (ext(S680), DMZ),
    "NIR mean SNV": (NIR, False),
    "NIR mean SNV, guided by size": (NIR, True),
    "NIR extreme angles (680)": (ext(S680), False),
}
G = np.arange(N)
P_dm, _, ch_dm = nested(y, None, G, pool=pool_dm)
M.append(
    dict(
        output="brix",
        inputs="NIR",
        calibration="+ dry matter as training-only teacher (McGlone 2003; Vapnik 2009)",
        n=N,
        cv_r=cvr(y, P_dm),
        cv_R2=r2(y, P_dm),
        within_lot_r=_within_r(y, P_dm, B),
        source="nested LOO, this script",
    )
)

pool_epo = {"NIR EPO mean": (NIR_EPO, False), "NIR EPO mean, guided by size": (NIR_EPO, True)}
for t in ["brix", "hardness"]:
    yt = V[t].to_numpy(float)
    Pp, _, _ = nested(yt, None, G, pool=pool_epo)
    M.append(
        dict(
            output=t,
            inputs="NIR",
            calibration="size direction removed from spectra (EPO, Roger 2003)",
            n=N,
            cv_r=cvr(yt, Pp),
            cv_R2=r2(yt, Pp),
            within_lot_r=_within_r(yt, Pp, B),
            source="nested LOO, this script",
        )
    )

ydm = V.dry_matter.to_numpy(float)
P_dmx, _, _ = nested(ydm, "NIR", G)
M.append(
    dict(
        output="dry_matter",
        inputs="NIR",
        calibration="as final analysis",
        n=N,
        cv_r=cvr(ydm, P_dmx),
        cv_R2=r2(ydm, P_dmx),
        within_lot_r=_within_r(ydm, P_dmx, B),
        source="nested LOO, this script",
    )
)

sm = V.loc[SENS_APPLES, "sens_mealy"].to_numpy(float)
for fam in ["NIR", "US", "NIR+US"]:
    for t in ["puncture2", "hardness"]:
        yt = V[t].to_numpy(float)
        if t == "hardness":
            h = HP[
                (HP.target == "hardness")
                & (HP.family == fam)
                & (HP.scheme == "LOO")
                & HP.model.str.startswith("NESTED")
            ].set_index("apple")
            pr = h.predicted.reindex(SENS_APPLES).to_numpy()
        else:
            okm = ~np.isnan(yt)
            yy = np.where(okm, yt, np.nanmean(yt))
            Pp, _, _ = nested(yy, fam, G)
            pr = pd.Series(Pp, index=APPLES).loc[SENS_APPLES].to_numpy()
        M.append(
            dict(
                output="mealiness (panel, 15)",
                inputs=fam,
                calibration=f"via instrumental proxy: {t} model on 70 apples",
                n=15,
                cv_r=cvr(sm, pr),
                cv_R2=np.nan,
                within_lot_r=_within_r(sm, pr, B[np.isin(APPLES, SENS_APPLES)]),
                spearman=stats.spearmanr(sm, pr)[0],
                source="LOO predictions vs panel",
            )
        )

for t in ["puncture2", "hardness", "dry_matter", "brix", "intact_share", "cell_count"]:
    v = V.loc[SENS_APPLES, t].to_numpy(float)
    M.append(
        dict(
            output="mealiness (panel, 15)",
            inputs=f"measured {t}",
            calibration="none (reference)",
            n=15,
            cv_r=np.corrcoef(sm, v)[0, 1],
            cv_R2=np.nan,
            within_lot_r=_within_r(sm, v, B[np.isin(APPLES, SENS_APPLES)]),
            spearman=stats.spearmanr(sm, v)[0],
            source="measured",
        )
    )


def honest_proxy(family, proxies):
    pr_meas, pr_model = [], []
    Zall = V[proxies].to_numpy(float)
    Zall = np.where(np.isnan(Zall), np.nanmean(Zall, 0), Zall)
    for i, a in enumerate(SENS_APPLES):
        tr = [b for b in SENS_APPLES if b != a]
        Zt = V.loc[tr, proxies].to_numpy(float)
        yt = V.loc[tr, "sens_mealy"].to_numpy(float)
        Zt = np.where(np.isnan(Zt), np.nanmean(Zall, 0), Zt)
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), Zt]), yt, rcond=None)[0]
        proxy = coef[0] + Zall @ coef[1:]
        k = list(APPLES).index(a)
        pr_meas.append(proxy[k])
        if family is not None:
            groups = (APPLES == a).astype(int)
            Pp, _, _ = nested(proxy, family, groups)
            pr_model.append(Pp[k])
    return np.array(pr_meas), (np.array(pr_model) if family is not None else None)


SB_ = B[np.isin(APPLES, SENS_APPLES)]
for proxies in (["dry_matter", "puncture2"], ["dry_matter"]):
    pm, _ = honest_proxy(None, proxies)
    M.append(
        dict(
            output="mealiness (panel, 15)",
            inputs="measured " + "+".join(proxies),
            calibration="regression calibration, LOO weights",
            n=15,
            cv_r=cvr(sm, pm),
            cv_R2=r2(sm, pm),
            within_lot_r=_within_r(sm, pm, SB_),
            spearman=stats.spearmanr(sm, pm)[0],
            source="LOO",
        )
    )
    for fam in ["NIR", "NIR+US"]:
        _, pmod = honest_proxy(fam, proxies)
        M.append(
            dict(
                output="mealiness (panel, 15)",
                inputs=fam,
                calibration="via regression-calibrated proxy "
                + "+".join(proxies)
                + " (Carroll 2006)",
                n=15,
                cv_r=cvr(sm, pmod),
                cv_R2=r2(sm, pmod),
                within_lot_r=_within_r(sm, pmod, SB_),
                spearman=stats.spearmanr(sm, pmod)[0],
                source="honest LOO (calibration and model both without the held-out apple)",
            )
        )
MT = pd.DataFrame(M)
MT["cv_r_disattenuated"] = [
    (
        r
        / np.sqrt(
            REL.get(
                {"brix": "brix", "hardness": "hardness", "dry_matter": "dry_matter"}.get(
                    o, "sens_mealy"
                ),
                np.nan,
            )
        )
        if o in ("brix", "hardness", "dry_matter", "mealiness (panel, 15)")
        else np.nan
    )
    for o, r in zip(MT.output, MT.cv_r)
]
MT.round(4).to_csv(OUT / "model_correlations.csv", index=False, encoding="utf-8-sig")
TAB.round(5).to_csv(OUT / "data_table.csv", encoding="utf-8-sig")
json.dump(dict(chosen_dm=[list(c) for c in ch_dm[:5]]), open(OUT / "run_info.json", "w"), indent=1)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print("RELIABILITIES")
print(
    pd.Series({k: v for k, v in REL.items() if not k.startswith(("nir", "us"))})
    .round(3)
    .to_string()
)
print(
    "NIR reliability range",
    np.nanmin([REL[f"nir_{c}"] for c in CH]).round(3),
    np.nanmax([REL[f"nir_{c}"] for c in CH]).round(3),
)
print("\nMODELS")
print(MT.round(3).to_string(index=False))
print("\nSTRONG PAIRS (|r| >= 0.6 raw or within lot)")
cols = [
    "family",
    "x",
    "y",
    "n",
    "r",
    "ci_lo",
    "ci_hi",
    "q_fdr",
    "r_within_lot",
    "q_fdr_within",
    "r_disattenuated",
]
St = R[R.strong].assign(a=lambda d: -d.r.abs()).sort_values(["family", "y", "a"])
print(St[cols].round(3).to_string(index=False))
print("\nTOP per family x output (by |r|)")
print(
    R.loc[R.groupby(["family", "y"]).r.apply(lambda s: s.abs().idxmax())][cols]
    .round(3)
    .to_string(index=False)
)
