# Exports the cell counts as CSV, one table per photo and one per apple with the sensory survey
# means joined. It computes the Spearman correlation of every cell measure with every sensory item
# across the 15 apples and draws the scatter plots. Run analyze.py first.

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
FA = Path(r"C:/apple_project/output/final_analysis_20261114")
sys.path.insert(0, str(FA / "scripts"))
from core import sensory_ratings, texture_index

WF = json.load(open(OUT / "workflow_results.json", encoding="utf-8"))
I = pd.read_csv(OUT / "per_image.csv")

img = pd.DataFrame(
    dict(
        image=I.image,
        apple=I.apple,
        field=I.field,
        cell_count=I["count"],
        n_intact=I.n_intact,
        n_broken=I.n_broken,
        intact_share=(I.n_intact / I["count"]).round(3),
        coverage=I.coverage.round(4),
        median_area_px=I.median_area_px,
        detector_count=[WF[i]["count"]["detector_count"] for i in I.image],
        revised=[WF[i].get("revise") is not None for i in I.image],
        confidence=[
            (
                (WF[i]["revise"] or WF[i]["count"])["confidence"]
                if WF[i].get("revise")
                else WF[i]["count"]["confidence"]
            )
            for i in I.image
        ],
        check_checker_4sq=I.checker_interior,
        check_final_4sq=I.final_interior,
        v13b_fullframe_count=I.v13b_fullframe_count,
        v13b_coverage=I.v13b_coverage,
    )
)
img.to_csv(OUT / "cell_counts_per_image.csv", index=False, encoding="utf-8-sig")

g = img.groupby("apple")
ap = pd.DataFrame(
    dict(
        cell_count=g.cell_count.mean(),
        n_intact=g.n_intact.sum(),
        n_broken=g.n_broken.sum(),
        coverage=g.coverage.mean().round(4),
        v13b_fullframe_count=g.v13b_fullframe_count.mean(),
    )
)
ap["intact_share"] = (ap.n_intact / (ap.n_intact + ap.n_broken)).round(3)
allr = sensory_ratings()
raw = allr.groupby("apple")[["mealy", "juicy", "crisp"]].mean().round(2).add_prefix("survey_raw_")
cen = allr.groupby("apple")[["mealy_c", "juicy_c", "crisp_c"]].mean().round(3)
cen.columns = ["survey_mealy", "survey_juicy", "survey_crisp"]
n_ratings = allr.groupby("apple").size().rename("survey_n_ratings")
T = pd.read_csv(FA / "targets.csv").set_index("apple")[["batch", "hardness", "puncture2", "brix"]]
ap = (
    ap.join(raw)
    .join(cen)
    .join(texture_index(allr).round(3).rename("survey_texture"))
    .join(n_ratings)
    .join(T)
)
ap.to_csv(OUT / "cell_counts_per_apple.csv", encoding="utf-8-sig")

CELL = {
    "cell_count": "세포 수 (사진당)",
    "intact_share": "온전한 세포 비율",
    "coverage": "피도",
    "v13b_fullframe_count": "자동 계수 (사진 전체)",
}
SENS = {
    "survey_mealy": "푸석함",
    "survey_juicy": "다즙성",
    "survey_crisp": "아삭함",
    "survey_texture": "식감 점수",
}
rows = []
for c in CELL:
    for s in SENS:
        r, p = spearmanr(ap[c], ap[s])
        rows.append(dict(cell=c, sensory=s, rho=round(r, 3), p=round(p, 3), n=len(ap)))
C = pd.DataFrame(rows)
C.to_csv(OUT / "cells_vs_sensory.csv", index=False, encoding="utf-8-sig")

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8.5})
fig, axs = plt.subplots(len(CELL), len(SENS), figsize=(11, 10))
for i, c in enumerate(CELL):
    for j, s in enumerate(SENS):
        a = axs[i, j]
        a.scatter(ap[s], ap[c], s=22, color="#2a78d6")
        for k, (x, y) in enumerate(zip(ap[s], ap[c])):
            a.annotate(
                str(ap.index[k]),
                (x, y),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6.5,
                color="#555",
            )
        rr = C[(C.cell == c) & (C.sensory == s)].iloc[0]
        a.set_title(f"ρ = {rr.rho:.2f}, p = {rr.p:.2f}", fontsize=8.5, loc="left")
        if i == len(CELL) - 1:
            a.set_xlabel(SENS[s] + " (설문, 패널 중심화)")
        if j == 0:
            a.set_ylabel(CELL[c])
plt.tight_layout()
plt.savefig(OUT / "cells_vs_sensory.png", dpi=150)
plt.close()

pd.set_option("display.width", 220)
print(ap.drop(columns=["n_intact", "n_broken"]).to_string())
print(C.pivot(index="cell", columns="sensory", values="rho").to_string())
print(C.pivot(index="cell", columns="sensory", values="p").to_string())
