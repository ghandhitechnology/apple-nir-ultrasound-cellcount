# Analyses the sensory panel ratings of mealiness, juiciness and crispness for 15 apples. Each
# panelist tasted five apples in round 1 and the same five reshuffled in round 2, where code k means
# the k-th apple of that panelist's round-1 list. Panelist names are dropped.

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr

OUT = Path(__file__).resolve().parents[1]
SRC = r"C:/apple_project/data/sensory/sensory_responses.csv"
import sys as _s

FITTED = len(_s.argv) > 2 and _s.argv[2] == "fitted"
if FITTED:
    SRC = str(Path(__file__).resolve().parents[1] / "data/sensory_responses_fitted.csv")
USED = [37, 39, 40, 42, 49, 51, 63, 65, 69, 70, 81, 88, 93, 94, 99]
A = ["mealy", "juicy", "crisp"]
rows = list(csv.reader(open(SRC, encoding="utf-8")))[1:]
rec = []
for r in rows:
    for k in range(5):
        rec.append(
            dict(
                panel=int(r[1]),
                rnd=1 if r[3].startswith("1") else 2,
                pos=k + 1,
                code=int(r[4 + 4 * k]),
                mealy=int(r[5 + 4 * k]),
                juicy=int(r[6 + 4 * k]),
                crisp=int(r[7 + 4 * k]),
                note=r[24] if k == 0 else "",
            )
        )
d = pd.DataFrame(rec)
r1, r2 = d[d.rnd == 1].copy(), d[d.rnd == 2].copy()
L = pd.read_csv(
    r"C:/apple_project/output/nir_guided_20261113/inputs/analysis_data70.csv"
).set_index("apple")
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


L["tier70"] = pd.qcut(L.hardness.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
P("== design check")
P(
    "tiers of the 15 used apples (quintiles of all 70):",
    L.loc[USED].groupby("tier70").size().to_dict(),
)
P(L.loc[USED, ["hardness", "tier70", "batch"]].sort_values("hardness").T.to_string())
bad = sorted(set(r1.code) - set(USED))
P("1st-round codes not in the list of 15:", bad)
cnt = r1[r1.code.isin(USED)].groupby("code").size().reindex(USED, fill_value=0)
P("1st-round count per apple (12 panels x 5 / 15 = 4 if balanced):", cnt.to_dict())
P("panels in round 1:", sorted(r1.panel.unique()), " round 2:", sorted(r2.panel.unique()))
short = [a for a in USED if cnt[a] < 4]
P("apples rated <4 times in round 1 (panel 7 missing + typo codes):", short)

import sys

MAP = sys.argv[1] if len(sys.argv) > 1 else "ascending"
if MAP == "listing":
    m = r1.set_index(["panel", "pos"]).code
else:
    m = r1.assign(rk=r1.groupby("panel").code.rank().astype(int)).set_index(["panel", "rk"]).code
P("round-2 mapping:", MAP)
r2["apple"] = [m.get((p, c), np.nan) for p, c in zip(r2.panel, r2.code)]
r1["apple"] = r1.code
P("\nround-2 rows mapped:", r2.apple.notna().sum(), "/", len(r2), "(panel 7 has no round-1 row)")

tt = r1.merge(r2.dropna(subset=["apple"]), on=["panel", "apple"], suffixes=("_1", "_2"))
P("\n== test-retest (same panelist, same apple, n=%d pairs)" % len(tt))
TR = {}
for a in A:
    x, y = tt[a + "_1"], tt[a + "_2"]

    xc = x - tt.groupby("panel")[a + "_1"].transform("mean")
    yc = y - tt.groupby("panel")[a + "_2"].transform("mean")
    TR[a] = dict(
        r_raw=pearsonr(x, y)[0],
        r_within_panel=pearsonr(xc, yc)[0],
        mean_abs_diff=np.mean(np.abs(x - y)),
        shift_2minus1=np.mean(y - x),
    )
    P(a, {k: round(v, 2) for k, v in TR[a].items()})
pp = []
for p, g in tt.groupby("panel"):
    pp.append(dict(panel=p, **{a: spearmanr(g[a + "_1"], g[a + "_2"])[0] for a in A}))
pp = pd.DataFrame(pp).set_index("panel")
P("per-panelist retest rho (5 apples each):")
P(pp.round(2).to_string())
pp.to_csv(OUT / f"sensory_retest_per_panel_{MAP}{'_fitted' if FITTED else ''}.csv")

allr = pd.concat([r1, r2.dropna(subset=["apple"])])
allr = allr[allr.apple.isin(USED)].copy()
allr["apple"] = allr.apple.astype(int)
for a in A:
    allr[a + "_c"] = allr[a] - allr.groupby(["panel", "rnd"])[a].transform("mean")
S = allr.groupby("apple").agg(
    n=("mealy", "size"), **{a: (a, "mean") for a in A}, **{a + "_c": (a + "_c", "mean") for a in A}
)
S = S.join(L[["hardness", "harditer1", "harditer2", "brix", "batch", "tier70", "m"]])
S["tier"] = S.hardness.rank(method="first").pipe(lambda s: ((s - 1) // 3 + 1).astype(int))
S.round(3).to_csv(OUT / f"sensory_apple_scores_{MAP}{'_fitted' if FITTED else ''}.csv")
P("\n== apple-level (both rounds, panelist-centred), n ratings per apple:", S.n.to_dict())
P(
    S[["n", "mealy", "juicy", "crisp", "hardness", "brix", "tier"]]
    .round(2)
    .sort_values("hardness")
    .to_string()
)

h1 = allr[allr.rnd == 1].groupby("apple")[[a + "_c" for a in A]].mean()
h2 = allr[allr.rnd == 2].groupby("apple")[[a + "_c" for a in A]].mean()
j = h1.join(h2, lsuffix="_r1", rsuffix="_r2").dropna()
for a in A:
    rr = pearsonr(j[a + "_c_r1"], j[a + "_c_r2"])[0]
    P(
        "apple-mean reliability %s: r(round1, round2) %.2f -> both rounds (Spearman-Brown) %.2f"
        % (a, rr, 2 * rr / (1 + rr))
    )
P("\n== sensory vs measurements (15 apples)")
res = []
for a in A:
    for b in ["hardness", "harditer1", "harditer2", "brix", "m"]:
        rho, p = spearmanr(S[a + "_c"], S[b])
        res.append(dict(sensory=a, measure=b, rho=rho, p=p))
    w = S.copy()
    w = w[w.groupby("batch")[a].transform("size") > 1]
    for b in ["hardness", "brix"]:
        x = w[a + "_c"] - w.groupby("batch")[a + "_c"].transform("mean")
        y = w[b] - w.groupby("batch")[b].transform("mean")
        r, p = pearsonr(x, y)
        res.append(dict(sensory=a, measure=b + " (within batch)", rho=r, p=p, n=len(w)))
    k = kruskal(*[g[a + "_c"] for _, g in S.groupby("tier")])
    res.append(dict(sensory=a, measure="hardness tier (Kruskal)", rho=np.nan, p=k.pvalue))
    res.append(
        dict(
            sensory=a,
            measure="hardness tier (Spearman)",
            rho=spearmanr(S[a + "_c"], S.tier)[0],
            p=spearmanr(S[a + "_c"], S.tier)[1],
        )
    )
res = pd.DataFrame(res)
res.to_csv(OUT / f"sensory_vs_measurements_{MAP}{'_fitted' if FITTED else ''}.csv", index=False)
P(res.round(3).to_string())
P("\nsensory attribute inter-correlation (apple means):")
P(S[[a + "_c" for a in A]].corr("spearman").round(2).to_string())
P(
    "\ntier means of mealiness (panelist-centred):",
    S.groupby("tier").mealy_c.mean().round(2).to_dict(),
)

P(
    "\nposition effect on mealiness (raw mean by serving position):",
    {rd: g.groupby("pos").mealy.mean().round(2).to_dict() for rd, g in d.groupby("rnd")},
)

H = pd.read_csv(r"C:/apple_project/output/nir_guided_20261113/heldout_predictions.csv")
for model in ["NIR ridge", "guided: size"]:
    h = (
        H[(H.target == "hardness") & (H.scheme == "LOO") & (H.model == model)]
        .set_index("apple")
        .predicted
    )
    s = S.join(h.rename("nir_hard"))
    P(
        "NIR LOO-predicted hardness (%s) vs mealiness rho %.2f p %.3f; vs measured hardness on these 15 rho %.2f"
        % (model, *spearmanr(s.mealy_c, s.nir_hard), spearmanr(s.hardness, s.nir_hard)[0])
    )
P("notes:", [n for n in d.note if n])
(OUT / f"a01_sensory_log_{MAP}{'_fitted' if FITTED else ''}.txt").write_text(
    "\n".join(log), encoding="utf-8"
)
