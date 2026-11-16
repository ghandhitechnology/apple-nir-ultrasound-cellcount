# Extra checks on the held-out predictions. It tests how well soft apples are detected inside each
# batch and how well predictions agree with the sensory panel after batch differences are removed.

import json

from core import *
from core import _within_r
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

TG = pd.read_csv(ROOT / "targets.csv").set_index("apple").loc[APPLES]
HP = pd.read_csv(ROOT / "heldout_predictions.csv")
cal = json.load(open(ROOT / "sensory_calibration.json"))["chosen"]
soft = (TG.puncture2.to_numpy() < cal["knee"]).astype(int)
allr = sensory_ratings()
T = texture_index(allr).loc[SENS_APPLES]
sidx = np.array([list(APPLES).index(a) for a in SENS_APPLES])


def demean(v):
    return v - pd.Series(v).groupby(B).transform("mean").to_numpy()


rows = []
for (t, fam, sc, m), g in HP.groupby(["target", "family", "scheme", "model"]):
    if fam == "member":
        continue
    p = g.set_index("apple").loc[APPLES].predicted.to_numpy()
    r = dict(target=t, family=fam, scheme=sc)
    if t != "brix":
        r["auc_soft"] = roc_auc_score(soft, -p)

        num = den = 0
        for gv in np.unique(B):
            m_ = B == gv
            s_ = soft[m_]
            q = p[m_]
            if 0 < s_.sum() < len(s_):
                pos, neg = q[s_ == 1], q[s_ == 0]
                num += sum((pp < nn) + 0.5 * (pp == nn) for pp in pos for nn in neg)
                den += len(pos) * len(neg)
        r["auc_soft_within_batch"] = num / den
    dp = demean(p)[sidx]
    dT = T - pd.Series(T.to_numpy()).groupby(B[sidx]).transform("mean").to_numpy()
    r["rho_texture"] = spearmanr(p[sidx], T)[0]
    r["rho_texture_within_batch"] = spearmanr(dp, dT)[0]
    rows.append(r)

tt = D.nir_time.astype("int64").to_numpy() / 1e9
r = dict(
    target="hardness",
    family="control: acquisition time (raw)",
    scheme="-",
    auc_soft=roc_auc_score(soft, -tt),
)
num = den = 0
for gv in np.unique(B):
    m_ = B == gv
    s_ = soft[m_]
    q = tt[m_]
    if 0 < s_.sum() < len(s_):
        pos, neg = q[s_ == 1], q[s_ == 0]
        num += sum((pp < nn) for pp in pos for nn in neg)
        den += len(pos) * len(neg)
r["auc_soft_within_batch"] = num / den
rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(ROOT / "diagnostics.csv", index=False)
print(
    "soft apples (puncture2 < %.1f): %d of 70; per batch %s"
    % (cal["knee"], soft.sum(), pd.Series(soft).groupby(B).sum().to_dict())
)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
