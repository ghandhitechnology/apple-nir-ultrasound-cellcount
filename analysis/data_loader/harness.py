# Shared data loader and scoring used by every analysis script. It reads the labels, NIR spectra and
# ultrasound features of the 70 main-study apples and defines the cross-validation folds and scores.
# Results from different scripts are comparable because they all go through this file.

import json
import os
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HUNT = Path(__file__).resolve().parent
RAND = Path("C:/apple_project")
RES = HUNT / "results"
RES.mkdir(exist_ok=True)
CH = [
    f"{w}nm"
    for w in [
        410,
        435,
        460,
        485,
        510,
        535,
        560,
        585,
        610,
        645,
        680,
        705,
        730,
        760,
        810,
        860,
        900,
        940,
    ]
]
ALPHAS = np.logspace(-2, 4, 25)


class Data:
    pass


def snv(M):
    M = np.asarray(M, float)
    return (M - M.mean(1, keepdims=True)) / M.std(1, ddof=1, keepdims=True)


def load_all():
    D = Data()
    lab = (
        pd.read_csv(RAND / "datasets/본실험_031-100/analysis_data70.csv")
        .set_index("apple")
        .drop(columns="mass(after)")
    )
    w = pd.read_csv(
        RAND / "output/nir_guided_20261113/inputs/labels_with_dry_mass.csv", index_col=0
    )
    lab["dry_fraction"] = (w["mass(after)"] / w["mass(before)"]).reindex(lab.index)
    lab["dmean"] = (lab.d1 + lab.d2) / 2
    D.labels = lab
    D.y = lab.hardness.to_numpy(float)
    D.batch = lab.batch.to_numpy()
    D.apples = lab.index.to_numpy()

    f = pd.read_csv(RAND / "datasets/본실험_031-100/nir/nir_files_completed.csv")
    f = f[f.apple.isin(lab.index)].copy()
    f["apple"] = f.apple.astype(int)
    f["time"] = pd.to_datetime(f.time)
    D.nir_files = f
    D.nir_raw = f.groupby("apple")[CH].mean().loc[lab.index]
    D.nir_snv = pd.DataFrame(snv(D.nir_raw), index=lab.index, columns=CH)
    D.nir_angle_raw = f.groupby(["apple", "angle"])[CH].mean().unstack("angle")
    D.nir_time = f.groupby("apple").time.min().loc[lab.index]

    U = RAND / "output/us70_20261113"
    D.us_apple = pd.read_csv(U / "features_apple_level_primary.csv", index_col=0).loc[lab.index]
    runs = pd.read_csv(U / "features_run_level.csv")
    D.us_runs_all = runs
    D.us_runs = runs[runs.set == "primary"].copy()
    nz = pd.read_csv(U / "noise_floor_run_level.csv")
    D.us_noise = nz
    tb = np.load(U / "transfer_bins.npz")
    D.us_bins = {k: tb[k] for k in tb.files}
    D.us_bins_row_index = runs.index.to_numpy()
    D.us_raw_paths = RAND / "datasets/본실험_031-100/ultrasound/primary_recordings.csv"
    return D


def _within_r(y, p, batch):
    yb = y - pd.Series(y).groupby(batch).transform("mean").to_numpy()
    pb = p - pd.Series(p).groupby(batch).transform("mean").to_numpy()
    return float(np.corrcoef(yb, pb)[0, 1]) if pb.std() > 0 else np.nan


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def evaluate(est, X, y=None, D=None, schemes=("LOO", "LOBO"), n_jobs=4, batch=None):
    if D is None:
        D = _CACHE.setdefault("D", load_all())
    y = D.y if y is None else np.asarray(y, float)
    batch = D.batch if batch is None else batch
    X = np.asarray(X, float)
    out = {"n": len(y), "p": X.shape[1]}
    for s in schemes:
        cv = LeaveOneOut() if s == "LOO" else LeaveOneGroupOut()
        p = cross_val_predict(clone(est), X, y, cv=cv, groups=batch, n_jobs=n_jobs)
        out[f"{s}_R2"] = r2(y, p)
        out[f"{s}_RMSE"] = float(np.sqrt(np.mean((y - p) ** 2)))
        out[f"{s}_within_r"] = _within_r(y, p, batch)
        out[f"{s}_pred"] = p
    return out


_CACHE = {}


def log_result(worker, hid, desc, res, inputs="", uses_size=False, notes=""):
    row = dict(
        time=time.strftime("%H:%M:%S"),
        worker=worker,
        hypothesis=hid,
        description=desc,
        inputs=inputs,
        uses_size=uses_size,
        n=res["n"],
        p=res["p"],
        **{k: v for k, v in res.items() if k.endswith(("R2", "RMSE", "within_r"))},
        notes=notes,
    )
    fn = RES / f"{worker}.csv"
    pd.DataFrame([row]).to_csv(fn, mode="a", header=not fn.exists(), index=False)
    pr = {k: list(map(float, v)) for k, v in res.items() if k.endswith("_pred")}
    with open(RES / f"{worker}_preds.jsonl", "a") as fh:
        fh.write(json.dumps(dict(hypothesis=hid, description=desc, **pr)) + "\n")
    return row


def baseline():
    D = load_all()
    e = make_pipeline(StandardScaler(), RidgeCV(ALPHAS))
    return evaluate(e, D.nir_snv.to_numpy(), D=D)


if __name__ == "__main__":
    D = load_all()
    print(
        "labels",
        D.labels.shape,
        "nir files",
        D.nir_files.shape,
        "nir angle",
        D.nir_angle_raw.shape,
        "us apple",
        D.us_apple.shape,
        "us runs",
        D.us_runs.shape,
    )
    print(
        "max |nir_snv - analysis table|:",
        float(np.abs(D.nir_snv.to_numpy() - D.labels[CH].to_numpy()).max()),
    )
    b = baseline()
    print({k: round(v, 3) for k, v in b.items() if not k.endswith("_pred")})
    lb = pd.Series(D.y).groupby(D.batch).transform("mean").to_numpy()
    m = np.array([np.mean(D.y[(D.batch == D.batch[i]) & (np.arange(70) != i)]) for i in range(70)])
    print("oracle: own-batch mean of the other 9 apples, LOO R2 = %.3f" % r2(D.y, m))
