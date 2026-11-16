# Turns the reviewed cell counts of the 15 sensory apples (two photos each) into per-photo and
# per-apple tables. It checks how well the two photos, the blind re-check and the automatic counter
# agree. It then relates the cell measures to sensory texture, firmness, Brix and the NIR model.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = Path(r"D:/rando-testing/cell_count_2026-11-14_main/results")
V13 = Path(r"D:/cellcount_v13b_1112_work/main_20261114/v13b/summary.csv")
FA = Path(r"C:/apple_project/output/final_analysis_20261114")
sys.path.insert(0, str(FA / "scripts"))
from core import SENS_APPLES, sensory_ratings, texture_index

SQ = 1096 * 1096
INTERIOR = {
    "r2c2": (1646, 1920, 806, 1080),
    "r2c3": (1920, 2194, 806, 1080),
    "r3c2": (1646, 1920, 1080, 1354),
    "r3c3": (1920, 2194, 1080, 1354),
}
rng = np.random.default_rng(0)


def perm_p(x, y, n=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r0 = spearmanr(x, y)[0]
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return r0, (np.sum(np.abs(null) >= abs(r0) - 1e-12) + 1) / (n + 1), int(ok.sum())


def demean(v, g):
    v = pd.Series(np.asarray(v, float))
    return (v - v.groupby(np.asarray(g)).transform("mean")).to_numpy()


rows = []
for d in sorted(p for p in RES.iterdir() if p.is_dir() and p.name[:2].isdigit()):
    img = d.name
    apple, field = map(int, img.split("-"))
    r = dict(image=img, apple=apple, field=field)
    ob = d / "objects.csv"
    if not ob.exists():
        rows.append(r)
        continue
    O = pd.read_csv(ob)
    r["count"] = len(O)
    cl = d / "object_classes_main.csv"
    if cl.exists():
        C = pd.read_csv(cl)
        r["labelled"] = len(C)
        r["n_intact"] = int((C["class"].str.lower() == "intact").sum())
        r["n_broken"] = int((C["class"].str.lower() == "broken").sum())
        r["n_low_evidence"] = (
            int(C["low_evidence"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            if "low_evidence" in C
            else np.nan
        )
    r["coverage"] = O.area_px.sum() / SQ
    r["median_area_px"] = O.area_px.median()
    r["median_eq_diam_um"] = O.eq_diam_um_meta.median() if "eq_diam_um_meta" in O else np.nan
    for f, tag in (("review_main.json", "counter"), ("revise_main.json", "revise")):
        p = d / f
        if p.exists():
            J = json.load(open(p, encoding="utf-8"))
            r[f"{tag}_count"] = J.get("count")
            r[f"{tag}_conf"] = J.get("confidence")
            r[f"{tag}_cls_conf"] = J.get("classification_confidence")
    S = json.load(open(d / "summary.json"))
    r["summary_count"] = S.get("count")
    r["n_clump_flag"] = S.get("n_clump")
    k = d / "check" / "check.json"
    if k.exists():
        K = json.load(open(k, encoding="utf-8"))
        r["check_json"] = json.dumps(K)[:2000]

    r["final_interior"] = int(
        sum(
            ((O.x >= x0) & (O.x < x1) & (O.y >= y0) & (O.y < y1)).sum()
            for x0, x1, y0, y1 in INTERIOR.values()
        )
    )
    pts = d / "check" / "points.csv"
    if pts.exists():
        P = pd.read_csv(pts)
        r["checker_interior"] = len(P)
        if "class" in P:
            r["checker_intact_share"] = (P["class"].str.lower() == "intact").mean()
        if cl.exists():
            Oc = O.merge(C[["id", "class"]], on="id", how="left", suffixes=("", "_m"))
            inn = np.zeros(len(Oc), bool)
            for x0, x1, y0, y1 in INTERIOR.values():
                inn |= ((Oc.x >= x0) & (Oc.x < x1) & (Oc.y >= y0) & (Oc.y < y1)).to_numpy()
            r["counter_interior_intact_share"] = (
                Oc.loc[inn, "class_m"].str.lower() == "intact"
            ).mean()
    rows.append(r)
I = pd.DataFrame(rows)
I["intact_share"] = I.n_intact / I["count"]
if V13.exists():
    v = pd.read_csv(V13).set_index("image")
    I["v13b_fullframe_count"] = I.image.map(v["count"])
    I["v13b_coverage"] = I.image.map(v["coverage"])
OUT = ROOT
OUT.mkdir(exist_ok=True)
I.drop(columns=["check_json"], errors="ignore").to_csv(
    OUT / "per_image.csv", index=False, encoding="utf-8-sig"
)

lines = []
ok = I.dropna(subset=["count"])
if "checker_interior" in I:
    k = I.dropna(subset=["checker_interior"])
    ratio = k.checker_interior / k.final_interior
    lines.append(
        f"checker/final in 4 interior squares: median {ratio.median():.2f}, range {ratio.min():.2f}-{ratio.max():.2f}, "
        f"within 10 %: {(abs(ratio - 1) <= .10).sum()}/{len(k)}; Spearman over images {spearmanr(k.checker_interior, k.final_interior)[0]:.2f}"
    )
    if "checker_intact_share" in k:
        lines.append(
            f'intact share in the 4 squares, checker vs counter: Spearman {spearmanr(k.checker_intact_share, k.counter_interior_intact_share, nan_policy="omit")[0]:.2f}; '
            f"mean {k.checker_intact_share.mean():.2f} vs {k.counter_interior_intact_share.mean():.2f}"
        )
if "v13b_fullframe_count" in I:
    vv = ok.dropna(subset=["v13b_fullframe_count"])
    lines.append(
        f'agent count vs v13b full-frame count (images): Spearman {spearmanr(vv["count"], vv.v13b_fullframe_count)[0]:.2f} (n={len(vv)})'
    )
w = ok.pivot(index="apple", columns="field", values="count")
lines.append(
    f'field 1 vs field 2 count (apples): Spearman {spearmanr(w[1], w[2], nan_policy="omit")[0]:.2f}'
)
if "intact_share" in ok:
    w2 = ok.pivot(index="apple", columns="field", values="intact_share")
    lines.append(
        f'field 1 vs field 2 intact share (apples): Spearman {spearmanr(w2[1], w2[2], nan_policy="omit")[0]:.2f}'
    )
    w3 = ok.pivot(index="apple", columns="field", values="coverage")
    lines.append(
        f'field 1 vs field 2 coverage (apples): Spearman {spearmanr(w3[1], w3[2], nan_policy="omit")[0]:.2f}'
    )

A = ok.groupby("apple").agg(
    fields=("image", "size"),
    count=("count", "mean"),
    n_intact=("n_intact", "sum"),
    n_broken=("n_broken", "sum"),
    coverage=("coverage", "mean"),
    median_area_px=("median_area_px", "mean"),
)
A["intact_share"] = A.n_intact / (A.n_intact + A.n_broken)
if "v13b_fullframe_count" in ok:
    A["v13b_fullframe_count"] = ok.groupby("apple").v13b_fullframe_count.mean()
T = pd.read_csv(FA / "targets.csv").set_index("apple")
allr = sensory_ratings()
sens = (
    allr.groupby("apple")[["mealy_c", "juicy_c", "crisp_c"]].mean().rename(columns=lambda c: c[:-2])
)
sens["texture"] = texture_index(allr)
HP = pd.read_csv(FA / "heldout_predictions.csv")
nir = HP[(HP.family == "NIR") & (HP.scheme == "LOO") & HP.model.str.startswith("NESTED")]
A = A.join(T[["batch", "hardness", "puncture2", "hard_cal", "brix", "soft_flag"]]).join(sens)
for t in ("hardness", "brix"):
    A[f"nir_pred_{t}"] = nir[nir.target == t].set_index("apple").predicted
A.to_csv(OUT / "per_apple.csv", encoding="utf-8-sig")

cell_vars = ["count", "intact_share", "coverage", "median_area_px"] + (
    ["v13b_fullframe_count"] if "v13b_fullframe_count" in A else []
)
tgt = ["texture", "mealy", "juicy", "crisp", "hardness", "puncture2", "brix", "nir_pred_hardness"]
R = []
for cv in cell_vars:
    for t in tgt:
        r0, p, n = perm_p(A[cv], A[t])
        rw = spearmanr(demean(A[cv], A.batch), demean(A[t], A.batch))[0]
        R.append(dict(cell=cv, target=t, rho=r0, p_perm=p, n=n, rho_within_batch=rw))
R = pd.DataFrame(R)
R.to_csv(OUT / "correlations.csv", index=False)
n_tests = len(R)
lines.append(
    f"{n_tests} cell x target tests; expected false positives at p<0.05 about {0.05 * n_tests:.1f}"
)
open(OUT / "agreement.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
pd.set_option("display.width", 220)
print("\n".join(lines))
print(A.round(3).to_string())
print(R.round(3).to_string(index=False))
