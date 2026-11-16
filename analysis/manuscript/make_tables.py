# Builds the LaTeX tables of the manuscript from the result files, so the numbers in the draft are
# never retyped by hand.

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "tables"
TB.mkdir(exist_ok=True)
O = Path("C:/apple_project/output")
FA, GA, CM = (
    O / "final_analysis_20261114",
    O / "general_analysis_20261114",
    O / "cells_main_20261114",
)
T = pd.read_csv(GA / "data_table.csv", index_col=0)


def f2(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


def f3(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def neg(s):
    return s.replace("-", "$-$") if s.startswith("-") else s


rows = [
    ("Fruit mass (g)", "mass"),
    ("Fruit volume (cm$^3$)", "volume"),
    ("Density (g cm$^{-3}$)", "density"),
    ("SSC ($^{\\circ}$Brix)", "brix"),
    ("Puncture firmness, mean of 2", "hardness"),
    ("Puncture firmness, 2nd puncture", "puncture2"),
    ("Dry matter (\\%)", "dry_matter"),
    ("Mealiness, panel ($z$)", "sens_mealy"),
    ("Juiciness, panel ($z$)", "sens_juicy"),
    ("Crispness, panel ($z$)", "sens_crisp"),
    ("Cell count per field", "cell_count"),
    ("Intact share", "intact_share"),
]
L = [
    "\\begin{tabular}{lrrrrr}",
    "\\toprule",
    "Variable & $n$ & Mean & SD & Min & Max \\\\",
    "\\midrule",
]
for lab, c in rows:
    v = T[c].dropna()
    L.append(
        f"{lab} & {len(v)} & {neg(f2(v.mean()))} & {f2(v.std())} & {neg(f2(v.min()))} & {neg(f2(v.max()))} \\\\"
    )
L += ["\\bottomrule", "\\end{tabular}"]
(TB / "tab1_descriptive.tex").write_text("\n".join(L), encoding="utf-8")

SC = pd.read_csv(FA / "scores.csv")
PS = pd.read_csv(FA / "permutation_summary.csv")
HP = pd.read_csv(FA / "heldout_predictions.csv")
fams = [
    ("NIR", "NIR"),
    ("NIR (US teacher)", "NIR, US as teacher"),
    ("NIR+US", "NIR + US"),
    ("US", "US"),
]
L = [
    "\\begin{tabular}{llrrrrr}",
    "\\toprule",
    "Output & Inputs & $R^2_{\\mathrm{LOO}}$ & $R^2_{\\mathrm{LOLO}}$ & $r_{\\mathrm{cv}}$ & $r_{\\mathrm{within}}$ & $p_{\\mathrm{perm}}$ \\\\",
    "\\midrule",
]
for t, tl in [("hardness", "Firmness"), ("hard_cal", "Panel-calibrated texture"), ("brix", "SSC")]:
    for i, (f, fl) in enumerate(fams):
        s = SC[(SC.target == t) & (SC.family == f) & SC.model.str.startswith("NESTED")].set_index(
            "scheme"
        )
        h = HP[
            (HP.target == t)
            & (HP.family == f)
            & (HP.scheme == "LOO")
            & HP.model.str.startswith("NESTED")
        ]
        rcv = np.corrcoef(h.actual, h.predicted)[0, 1]
        p = PS[
            (PS.test == "label") & (PS.target == t) & (PS.family == f) & (PS.scheme == "LOO")
        ].p_R2
        L.append(
            f'{tl if i == 0 else ""} & {fl} & {neg(f2(s.R2["LOO"]))} & {neg(f2(s.R2["LOBO"]))} & {neg(f2(rcv))} & {neg(f2(s.within_r["LOO"]))} & {f3(p.iloc[0]) if len(p) else "--"} \\\\'
        )
    c = SC[(SC.target == t) & (SC.model == "CONTROL acquisition time")].set_index("scheme").R2
    if len(c):
        L.append(
            f' & \\emph{{acquisition time only}} & {neg(f2(c.get("LOO")))} & {neg(f2(c.get("LOBO")))} & & & \\\\'
        )
    L.append("\\addlinespace")
L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
(TB / "tab2_models.tex").write_text("\n".join(L), encoding="utf-8")

A = pd.read_csv(CM / "cell_counts_per_apple.csv").set_index("apple")
L = [
    "\\begin{tabular}{rrrrrrr}",
    "\\toprule",
    "Apple & Count & Intact share & Coverage & Auto count$^{a}$ & Mealiness ($z$) & Firmness \\\\",
    "\\midrule",
]
for a, r in A.iterrows():
    L.append(
        f'{a} & {r.cell_count:.1f} & {r.intact_share:.2f} & {r.coverage:.3f} & {r.v13b_fullframe_count:.0f} & {neg(f2(T.loc[a, "sens_mealy"]))} & {r.hardness:.2f} \\\\'
    )
L += ["\\bottomrule", "\\end{tabular}"]
(TB / "tab3_cells.tex").write_text("\n".join(L), encoding="utf-8")

R = pd.read_csv(GA / "correlation_screen.csv")
NM = {
    "brix": "SSC",
    "hardness": "Firmness",
    "puncture2": "Firmness (2nd)",
    "sens_mealy": "Mealiness",
    "dry_matter": "Dry matter",
    "mass": "Fruit mass",
    "sens_crisp": "Crispness",
    "sens_juicy": "Juiciness",
}


def nm(v):
    if v.startswith("nirEPO_"):
        return "NIR " + v[7:-2] + " nm (size removed)"
    if v.startswith("nir_"):
        return "NIR " + v[4:-2] + " nm"
    if v.startswith("us_"):
        return "US " + v[3:].replace("_", "\\_")
    return NM.get(v, v.replace("_", " "))


S = R[
    R.strong
    & ~((R.x == "hardness") & (R.y == "puncture2"))
    & (R.family != "output <-> output/other")
].copy()
S = S.assign(a=-S.r.abs()).sort_values(["n", "a"], ascending=[False, True])
L = [
    "\\begin{tabular}{llrrrrrr}",
    "\\toprule",
    "Pair & & $n$ & $r$ & 95\\% CI & $q_{\\mathrm{BH}}$ & $q_{\\mathrm{BY}}$ & $r_{\\mathrm{within}}$ \\\\",
    "\\midrule",
]
for _, s in S.iterrows():
    L.append(
        f"{nm(s.x)} & {nm(s.y)} & {s.n} & {neg(f2(s.r))} & [{neg(f2(s.ci_lo))}, {neg(f2(s.ci_hi))}] & {f3(s.q_fdr)} & {f3(s.q_by)} & {neg(f2(s.r_within_lot))} \\\\"
    )
L += ["\\bottomrule", "\\end{tabular}"]
(TB / "tab4_strong.tex").write_text("\n".join(L), encoding="utf-8")

rel = pd.read_csv(GA / "reliabilities.csv", index_col=0).reliability
us = pd.read_csv(O / "us70_20261113/repeatability_icc.csv").set_index("feature")
nir = rel[[i for i in rel.index if i.startswith("nir_") and i.endswith("nm")]]
items = [
    ("SSC", "two readings", rel["brix"]),
    ("Puncture firmness (mean)", "two punctures", rel["hardness"]),
    ("Puncture firmness (2nd)", "single puncture", rel["puncture2"]),
    ("Dry matter", "balance resolution only (upper bound)", rel["dry_matter"]),
    ("Mealiness", "two sessions", rel["sens_mealy"]),
    ("Juiciness", "two sessions", rel["sens_juicy"]),
    ("Crispness", "two sessions", rel["sens_crisp"]),
    ("Cell count", "two fields", rel["cell_count"]),
    ("Intact share", "two fields", rel["intact_share"]),
    ("Coverage", "two fields", rel["coverage"]),
]
L = [
    "\\begin{tabular}{llr}",
    "\\toprule",
    "Measurement & Replicates & Reliability \\\\",
    "\\midrule",
]
for a, b, v in items:
    L.append(f"{a} & {b} & {f2(v)} \\\\")
L.append(f"NIR channels (18) & 12 scans & {f2(nir.min())}--{f2(nir.max())} \\\\")
L.append(
    f"US features (12 run-level) & mean of 8 runs & {f2(us.ICC_mean8_70.min())}--{f2(us.ICC_mean8_70.max())} \\\\"
)
L += ["\\bottomrule", "\\end{tabular}"]
(TB / "tab5_reliability.tex").write_text("\n".join(L), encoding="utf-8")
print("tables written:", sorted(p.name for p in TB.iterdir()))
