# Chance-level tests for the final models. Shuffling labels within a batch tests whether the
# procedure ranks apples inside a batch better than chance. Shuffling ultrasound rows within a batch
# tests whether ultrasound adds anything specific to the apple beyond session drift.

import json
import sys
import time

from core import *
from core import _within_r
from joblib import Parallel, delayed

TG = pd.read_csv(ROOT / "targets.csv").set_index("apple").loc[APPLES]
TARGETS = {
    "hardness": TG.hardness.to_numpy(float),
    "hard_cal": TG.hard_cal.to_numpy(float),
    "brix": TG.brix.to_numpy(float),
}
NL, NU = int(sys.argv[1]) if len(sys.argv) > 1 else 50, (
    int(sys.argv[2]) if len(sys.argv) > 2 else 100
)
SINGLE = {
    "hardness": "NIR mean SNV + US transfer bins (32)",
    "hard_cal": "NIR mean SNV + US transfer bins (32)",
    "brix": "NIR mean SNV + US features, guided by size",
}
LOO = np.arange(N)


def stat(y, P):
    return dict(R2=r2(y, P), within_r=_within_r(y, P, B))


def label_null(seed, tname, fam):
    rng = np.random.default_rng(seed)
    y = TARGETS[tname][within_batch_perm(rng)]
    P, _, _ = nested(y, fam, LOO)
    return dict(kind="label", target=tname, family=fam, scheme="LOO", seed=seed, **stat(y, P))


def us_null(seed, tname, fam, scheme):
    rng = np.random.default_rng(10_000 + seed)
    perm = within_batch_perm(rng) if seed >= 0 else None
    y = TARGETS[tname]
    groups = LOO if scheme == "LOO" else B
    if fam == "single":
        pool = {SINGLE[tname]: members("NIR+US", perm)[SINGLE[tname]]}
        P, _, _ = nested(y, fam, groups, max_k=1, pool=pool)
    else:
        P, _, _ = nested(y, fam, groups, pool=members(fam, perm))
    return dict(
        kind="us_rows" if seed >= 0 else "observed",
        target=tname,
        family=fam,
        scheme=scheme,
        seed=seed,
        **stat(y, P),
    )


t0 = time.time()
jobs = [
    delayed(label_null)(s, t, f)
    for t in TARGETS
    for f in ["NIR", "NIR+US", "NIR (US teacher)"]
    for s in range(NL)
]
jobs += [
    delayed(us_null)(s, t, f, sc)
    for t in TARGETS
    for f in ["US", "NIR+US", "NIR (US teacher)", "single"]
    for sc in ["LOO", "LOBO"]
    for s in [-1] + list(range(NU))
]
res = pd.DataFrame(Parallel(n_jobs=12, verbose=2)(jobs))
res.to_csv(ROOT / "permutation_raw.csv", index=False)
print("done %.0f s" % (time.time() - t0))

obs_scores = pd.read_csv(ROOT / "scores.csv")
out = []
for (kind, t, f, sc), g in res[res.kind != "observed"].groupby(
    ["kind", "target", "family", "scheme"]
):
    if kind == "label":
        o = obs_scores[
            (obs_scores.target == t)
            & (obs_scores.family == f)
            & (obs_scores.scheme == sc)
            & obs_scores.model.str.startswith("NESTED")
        ].iloc[0]
        oR2, ow = o.R2, o.within_r
    else:
        o = res[
            (res.kind == "observed") & (res.target == t) & (res.family == f) & (res.scheme == sc)
        ].iloc[0]
        oR2, ow = o.R2, o.within_r
    out.append(
        dict(
            test=kind,
            target=t,
            family=f if f != "single" else "single: " + SINGLE[t],
            scheme=sc,
            n=len(g),
            observed_R2=oR2,
            null_R2_median=g.R2.median(),
            null_R2_95=g.R2.quantile(0.95),
            p_R2=(1 + (g.R2 >= oR2).sum()) / (1 + len(g)),
            observed_within_r=ow,
            null_within_r_95=g.within_r.quantile(0.95),
            p_within_r=(1 + (g.within_r >= ow).sum()) / (1 + len(g)),
        )
    )
S = pd.DataFrame(out)
S.to_csv(ROOT / "permutation_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(S.round(3).to_string(index=False))
