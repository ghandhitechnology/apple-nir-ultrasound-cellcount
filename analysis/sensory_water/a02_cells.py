# Compares the automatic cell counts of the pilot apples (1-30) with their hardness, Brix and
# texture notes. These apples were not part of the sensory panel.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[1]
C = pd.read_csv(
    r"C:/apple_project/research_archive/fullframe_v13/v13b/per_apple.csv", encoding="utf-8-sig"
)
I = pd.read_csv(
    r"C:/apple_project/research_archive/fullframe_v13/v13b/per_image.csv", encoding="utf-8-sig"
)
A = pd.read_csv(r"D:/cellcount_wrapup/reanalysis/retrain_1108/apples30.csv")
C["apple"] = C.apple.astype(int)
I["apple"] = I.apple.astype(int)
d = C.set_index("apple").join(
    A.set_index("fruit_id")[
        ["hardness", "hardness_sd", "brix", "texture_note", "total_count", "cells_per_field"]
    ],
    rsuffix="_old",
)
d["clump_frac"] = d.clumps / (d.total_cells + d.clumps)
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


P("apples", len(d), "fields per apple", d.fields.value_counts().to_dict())

w = I.groupby("apple")["count"].agg(["min", "max", "mean", "size"])
w = w[w["size"] >= 2]
P(
    "two photos of the same apple: median |difference| / mean = %.0f%%"
    % (100 * np.median((w["max"] - w["min"]) / w["mean"]))
)
I2 = I.sort_values("image").groupby("apple")["count"].apply(list)
a = np.array([v[:2] for v in I2 if len(v) >= 2])
P("photo 1 vs photo 2 rho %.2f (n %d)" % (spearmanr(a[:, 0], a[:, 1])[0], len(a)))
P("v13b vs old count (apples30 total_count) rho %.2f" % spearmanr(d.total_cells, d.total_count)[0])
res = []
for x in ["cells_per_field", "coverage", "clump_frac", "cells_per_field_old"]:
    for y in ["hardness", "brix"]:
        r, p = spearmanr(d[x], d[y])
        res.append(dict(cell=x, measure=y, rho=r, p=p, n=len(d)))
res = pd.DataFrame(res)
P(res.round(3).to_string())
res.to_csv(OUT / "cells_vs_measurements.csv", index=False)
notes = d.dropna(subset=["texture_note"])
mealy_word = notes.texture_note.str.contains("푸석")
P("\napples with a texture note:", len(notes))
P(
    notes[["texture_note", "cells_per_field", "coverage", "hardness", "brix"]]
    .assign(mealy_word=mealy_word)
    .to_string()
)
P(
    "note says 푸석 (mealy): mean cells/field %.0f vs others %.0f"
    % (notes[mealy_word].cells_per_field.mean(), notes[~mealy_word].cells_per_field.mean())
)
d.round(4).to_csv(OUT / "cells_per_apple_joined.csv")
(OUT / "a02_cells_log.txt").write_text("\n".join(log), encoding="utf-8")
