# Fits the final models on all 70 apples with the same procedure that was cross-validated. Every
# model is linear, so it is saved to final_models.json as a feature block with its means, scales,
# weights and intercept.

import json

from core import *

TG = pd.read_csv(ROOT / "targets.csv").set_index("apple").loc[APPLES]
TARGETS = {
    "hardness": TG.hardness.to_numpy(float),
    "hard_cal": TG.hard_cal.to_numpy(float),
    "brix": TG.brix.to_numpy(float),
}
ALL = np.ones(N, bool)
FEATURE_OF = {}
for n in members("NIR+US"):
    FEATURE_OF[n] = n.replace(", guided by size", "")
FEATURE_OF["NIR mean SNV + US features, guided by size"] = (
    "NIR mean SNV + US features + energy ratios (26)"
)
FEATURE_OF["US features + energy ratios, guided by size"] = "US features + energy ratios (26)"


def final(tname, fam):
    y = TARGETS[tname]
    pool = members(fam)
    names = list(pool)
    inner = {m: fit_member(pool[m][0], y, ALL, pool[m][1]) for m in names}
    best = None
    import itertools

    for k in (1, 2, 3):
        for c in itertools.combinations(names, k):
            s = r2(y, np.mean([inner[m][1] for m in c], 0))
            if best is None or s > best[0]:
                best = (s, c)
    mem = []
    for m in best[1]:
        lf = inner[m][2]
        mem.append(
            dict(
                member=m,
                features=FEATURE_OF.get(m, m),
                alpha=float(lf["alpha"]),
                mu=lf["mu"].tolist(),
                sd=lf["sd"].tolist(),
                w=lf["w"].tolist(),
                c0=float(lf["c0"]),
            )
        )
    fitted = np.mean([inner[m][0](pool[m][0]) for m in best[1]], 0)
    return (
        dict(
            target=tname,
            family=fam,
            members=mem,
            inner_loo_R2_all70=float(best[0]),
            in_sample_R2=float(r2(y, fitted)),
        ),
        fitted,
    )


models, fits = {}, {}
for t in TARGETS:
    for fam in ["NIR", "NIR+US"]:
        m, fv = final(t, fam)
        key = f"{t}__{fam}"
        models[key] = m
        fits[key] = fv
        print(
            key,
            [x["member"] for x in m["members"]],
            "inner LOO %.3f in-sample %.3f" % (m["inner_loo_R2_all70"], m["in_sample_R2"]),
        )
cal = json.load(open(ROOT / "sensory_calibration.json"))
meta = dict(
    created="2026-11-14",
    apples="31-100 (70)",
    channels=CH,
    sensory_calibration=dict(formula="texture = a + c * min(puncture2, knee)", **cal["chosen"]),
    note="hard_cal predictions are on the sensory texture scale (0 = average of the 15 panel apples; higher = firmer, juicier, crisper, less mealy).",
)
json.dump(dict(meta=meta, models=models), open(ROOT / "final_models.json", "w"), indent=1)
pd.DataFrame({k: v for k, v in fits.items()}, index=APPLES).to_csv(
    ROOT / "final_models_in_sample_predictions.csv"
)
