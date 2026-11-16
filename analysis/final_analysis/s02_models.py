# Scores every target against every input family (NIR, ultrasound, both) with nested
# leave-one-apple-out and leave-one-batch-out cross-validation. It also scores each single model, a
# control that uses only session information, and the agreement with the sensory panel.

import json
import time
from collections import Counter

from core import *
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

TG = pd.read_csv(ROOT / "targets.csv").set_index("apple").loc[APPLES]
CAL = json.load(open(ROOT / "sensory_calibration.json"))
allr = sensory_ratings()
T = texture_index(allr).loc[SENS_APPLES]
SENS_G = allr.groupby("apple")[["mealy_c", "juicy_c", "crisp_c"]].mean().loc[SENS_APPLES]
sidx = np.array([list(APPLES).index(a) for a in SENS_APPLES])
TARGETS = {
    "hardness": TG.hardness.to_numpy(float),
    "hard_cal": TG.hard_cal.to_numpy(float),
    "brix": TG.brix.to_numpy(float),
}
FAMILIES = ["NIR", "US", "NIR+US", "NIR (US teacher)"]
soft = (TG.puncture2.to_numpy() < CAL["chosen"]["knee"]).astype(int)


def sens_scores(p):
    q = p[sidx]
    return dict(
        rho_texture=spearmanr(q, T)[0],
        rho_mealy=spearmanr(q, SENS_G.mealy_c)[0],
        rho_juicy=spearmanr(q, SENS_G.juicy_c)[0],
        rho_crisp=spearmanr(q, SENS_G.crisp_c)[0],
    )


def run(tname, fam, scheme):
    y = TARGETS[tname]
    groups = np.arange(N) if scheme == "LOO" else B
    P, Ps, ch = nested(y, fam, groups)
    return tname, fam, scheme, P, Ps, ch


t0 = time.time()
jobs = [(t, f, s) for t in TARGETS for f in FAMILIES for s in ["LOO", "LOBO"]]
out = Parallel(n_jobs=12)(delayed(run)(*j) for j in jobs)
rows, preds, choices = [], [], []
for tname, fam, scheme, P, Ps, ch in out:
    y = TARGETS[tname]
    rows.append(
        dict(
            target=tname,
            family=fam,
            scheme=scheme,
            model="NESTED best average of 1-3 members",
            **score(y, P),
            **sens_scores(P),
            auc_soft=roc_auc_score(soft, -P) if tname != "brix" else np.nan
        )
    )
    preds.append(
        pd.DataFrame(
            dict(
                apple=APPLES,
                batch=B,
                target=tname,
                family=fam,
                scheme=scheme,
                model="NESTED",
                actual=y,
                predicted=P,
            )
        )
    )
    cnt = Counter(m for c in ch for m in c)
    n_us = sum(any(m.startswith("US") or "+ US" in m for m in c) for c in ch)
    choices.append(
        dict(
            target=tname,
            family=fam,
            scheme=scheme,
            folds=len(ch),
            folds_using_US=n_us,
            top_combo=Counter(ch).most_common(1)[0][0],
            top_combo_share=Counter(ch).most_common(1)[0][1] / len(ch),
            member_use=dict(cnt.most_common(8)),
        )
    )
    if fam == "NIR+US":
        for m, p in Ps.items():
            rows.append(
                dict(
                    target=tname,
                    family="member",
                    scheme=scheme,
                    model=m,
                    **score(y, p),
                    **sens_scores(p),
                    auc_soft=roc_auc_score(soft, -p) if tname != "brix" else np.nan
                )
            )
            preds.append(
                pd.DataFrame(
                    dict(
                        apple=APPLES,
                        batch=B,
                        target=tname,
                        family="member",
                        scheme=scheme,
                        model=m,
                        actual=y,
                        predicted=p,
                    )
                )
            )
print("nested runs %.0f s" % (time.time() - t0), flush=True)

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ctrl = {
    "CONTROL acquisition time": D.nir_time.astype("int64").to_numpy()[:, None] / 1e9,
    "CONTROL batch identity": pd.get_dummies(B).to_numpy(float),
}
for tname, y in TARGETS.items():
    for cn, X in ctrl.items():
        for scheme, cv in [("LOO", LeaveOneOut()), ("LOBO", LeaveOneGroupOut())]:
            p = cross_val_predict(
                make_pipeline(StandardScaler(), RidgeCV(AL)), X, y, cv=cv, groups=B
            )
            rows.append(
                dict(
                    target=tname,
                    family="control",
                    scheme=scheme,
                    model=cn,
                    **score(y, p),
                    **sens_scores(p),
                    auc_soft=roc_auc_score(soft, -p) if tname != "brix" else np.nan
                )
            )

h1 = TG.puncture1.to_numpy()
h2 = TG.puncture2.to_numpy()
t = T.to_numpy()
W = [0, 0.25, 0.5, 0.75, 1.0]
K = [np.inf, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]


def cal_without(i):
    rows_ = [j for j in range(15) if j != i]

    def err(w, k):
        x = np.minimum(w * h1 + (1 - w) * h2, k)[sidx]
        e = []
        for r in rows_:
            m = [j for j in rows_ if j != r]
            c, a = np.polyfit(x[m], t[m], 1)
            e.append((t[r] - a - c * x[r]) ** 2)
        return np.mean(e)

    _, w, k = min((err(w, k), w, k) for w in W for k in K)
    x = np.minimum(w * h1 + (1 - w) * h2, k)
    c, a = np.polyfit(x[sidx][rows_], t[rows_], 1)
    return a + c * x, a, c


def honest_one(i, fam):
    y, a, c = cal_without(i)
    te = np.zeros(N, bool)
    te[sidx[i]] = True
    P, _, _ = nested(y, fam, np.where(te, 1, 0))
    return (P[sidx[i]] - a) / c


hon = {}
for fam in FAMILIES:
    pr = Parallel(n_jobs=12)(delayed(honest_one)(i, fam) for i in range(15))
    hon[fam] = dict(rho_texture=float(spearmanr(pr, t)[0]))
print(
    "honest sensory agreement (calibration refitted without the held-out apple):", hon, flush=True
)

R = pd.DataFrame(rows)
R.to_csv(ROOT / "scores.csv", index=False)
pd.concat(preds).to_csv(ROOT / "heldout_predictions.csv", index=False)
pd.DataFrame(choices).to_csv(ROOT / "nested_choices.csv", index=False)
json.dump(hon, open(ROOT / "honest_sensory_agreement_hard_cal.json", "w"), indent=2)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
nest = R[R.model.str.startswith("NESTED")].pivot_table(
    index=["target", "family"],
    columns="scheme",
    values=["R2", "within_r", "rho_texture", "auc_soft"],
)
print(nest.round(3).to_string())
print(
    pd.DataFrame(choices)[
        ["target", "family", "scheme", "folds_using_US", "folds", "top_combo", "top_combo_share"]
    ].to_string()
)
mem = R[(R.family == "member") & (R.scheme == "LOO")].sort_values(
    ["target", "R2"], ascending=[True, False]
)
print(mem[["target", "model", "R2", "within_r", "rho_texture"]].round(3).to_string(index=False))
print(
    R[R.family == "control"][["target", "model", "scheme", "R2", "rho_texture", "auc_soft"]]
    .round(3)
    .to_string(index=False)
)
