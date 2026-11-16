# Calibrates the hardness label against the sensory panel. It fits a capped blend of the two
# puncture readings to the panel texture score of the 15 tasted apples, choosing the blend weight
# and the cap by leave-one-apple-out error. The fitted map is then applied to all 70 apples.

import json

from core import *
from scipy.stats import pearsonr, spearmanr

allr = sensory_ratings()
T = texture_index(allr).loc[SENS_APPLES]
h1 = L.harditer1.to_numpy(float)
h2 = L.harditer2.to_numpy(float)
hm = L.hardness.to_numpy(float)
h1 = np.where(np.isfinite(h1), h1, hm)
h2 = np.where(np.isfinite(h2), h2, hm)
W = [0, 0.25, 0.5, 0.75, 1.0]
K = [np.inf, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
idx = np.array([list(APPLES).index(a) for a in SENS_APPLES])
t = T.to_numpy()


def hw(w):
    return w * h1 + (1 - w) * h2


def fit_map(x, tt):
    c, a = np.polyfit(x, tt, 1)
    return a, c


def loo_err(w, k, rows):
    x = np.minimum(hw(w)[idx], k)
    e = []
    for i in rows:
        m = np.array([j for j in rows if j != i])
        a, c = fit_map(x[m], t[m])
        e.append((t[i] - a - c * x[i]) ** 2)
    return np.mean(e)


rows_all = list(range(15))
grid = pd.DataFrame(
    [
        dict(
            w_puncture1=w,
            knee=k,
            loo_mse=loo_err(w, k, rows_all),
            r_in_sample=(
                pearsonr(np.minimum(hw(w)[idx], k), t)[0]
                if np.std(np.minimum(hw(w)[idx], k)) > 0
                else np.nan
            ),
        )
        for w in W
        for k in K
    ]
)
grid.to_csv(ROOT / "sensory_calibration_grid.csv", index=False)
best = grid.sort_values("loo_mse").iloc[0]
w0, k0 = best.w_puncture1, best.knee


def choose(rows):
    g = [(loo_err(w, k, rows), w, k) for w in W for k in K]
    return min(g)[1:]


stab = pd.DataFrame(
    [
        dict(
            left_out=SENS_APPLES[i],
            w=choose([j for j in rows_all if j != i])[0],
            k=choose([j for j in rows_all if j != i])[1],
        )
        for i in rows_all
    ]
)
x = np.minimum(hw(w0), k0)
a, c = fit_map(x[idx], t)
L_out = pd.DataFrame(
    dict(
        apple=APPLES,
        batch=B,
        hardness=hm,
        puncture1=h1,
        puncture2=h2,
        h_weighted=hw(w0),
        hard_cal=a + c * x,
        brix=L.brix.to_numpy(float),
        soft_flag=(hw(w0) < k0).astype(int) if np.isfinite(k0) else np.nan,
    )
)
L_out = L_out.merge(T.rename("texture").reset_index(), on="apple", how="left")
L_out.to_csv(ROOT / "targets.csv", index=False)

ref = dict(
    mean_of_punctures=(0.5, np.inf), puncture2_only=(0.0, np.inf), puncture1_only=(1.0, np.inf)
)
summ = dict(
    chosen=dict(w_puncture1=w0, knee=k0, a=a, c=c, loo_mse=best.loo_mse),
    reference_loo_mse={n: float(loo_err(w, k, rows_all)) for n, (w, k) in ref.items()},
    T_variance=float(np.var(t)),
    choice_stability=stab.groupby(["w", "k"]).size().rename("n").reset_index().to_dict("records"),
    spearman_T={
        n: float(spearmanr(v[idx], t)[0])
        for n, v in dict(
            hardness=hm, puncture1=h1, puncture2=h2, h_weighted=hw(w0), hard_cal=a + c * x
        ).items()
    },
    soft_apples=int(np.sum(hw(w0) < k0)) if np.isfinite(k0) else None,
    texture_split_half=float(
        np.corrcoef(
            texture_index(allr, 1).loc[SENS_APPLES], texture_index(allr, 2).loc[SENS_APPLES]
        )[0, 1]
    ),
)
json.dump(summ, open(ROOT / "sensory_calibration.json", "w"), indent=2, default=float)
print(json.dumps(summ, indent=1, default=float))
print(grid.sort_values("loo_mse").head(10).round(3).to_string(index=False))
