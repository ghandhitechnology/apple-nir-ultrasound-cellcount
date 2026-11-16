# Independent audit of the 40 kHz ultrasound recordings, reading the raw files only. It recomputes
# envelope and energy-ratio measures and tests them against the labels. The measures are exploratory
# proxies, not calibrated tissue attenuation.

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert, sosfiltfilt
from scipy.stats import rankdata, spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "output/us70_20261113/scripts"))
from core import EDGES, FS, NB, SOS1, N, fit_sweep, process_run, t
from s04_models import BASE, folds_for, run_model

RNG = np.random.default_rng(20261114)


def metrics(A, B, T):
    bins, f = process_run(A, B, T)
    a = hilbert(sosfiltfilt(SOS1, A - A.mean()))
    b = hilbert(sosfiltfilt(SOS1, B - B.mean()))
    ff, seg, *_ = fit_sweep(b)
    bi = np.digitize(ff, EDGES) - 1
    ok = (seg >= 0) & (bi >= 0) & (bi < NB)
    ratios = []
    for q in range(NB):
        m = ok & (bi == q)
        ratios.append(np.sum(abs(a[m]) ** 2) / np.sum(abs(b[m]) ** 2))
    f["envelope_dB"] = 10 * np.log10(np.mean(ratios))
    f["local_gap_dB"] = f["level_rms_dB"] - f["envelope_dB"]
    full = np.sum(a[ok] * b[ok].conj()) / np.sum(abs(b[ok]) ** 2)
    f["broad_coherent_dB"] = 20 * np.log10(abs(full))
    f["broad_energy_dB"] = 10 * np.log10(np.sum(abs(a[ok]) ** 2) / np.sum(abs(b[ok]) ** 2))
    f["broad_gap_dB"] = f["broad_coherent_dB"] - f["broad_energy_dB"]
    f["A_min"], f["A_max"] = float(A.min()), float(A.max())
    f["extreme_sample_fraction"] = np.mean((A == A.min()) | (A == A.max()))
    return f


def simulate():
    rows = []

    def phase(x):
        k = np.floor(x / 0.01)
        u = x - k * 0.01
        return 2 * np.pi * (k * 400 + 38000 * u + 0.5 * 400000 * u * u)

    for start in [0.0, 0.0023, 0.0067]:
        B = 5.4 * np.cos(phase(t + start))
        for delay in np.arange(0, 0.002501, 0.00005):
            A = 0.054 * np.cos(phase(t + start - delay))
            f = metrics(A, B, t)
            rows.append(
                dict(
                    start_ms=start * 1000,
                    delay_ms=delay * 1000,
                    local_bias_dB=f["level_rms_dB"] + 40,
                    broad_bias_dB=f["broad_coherent_dB"] + 40,
                    legacy_psd_bias_dB=20 * np.log10(f["leg_H"]) + 40,
                    envelope_bias_dB=f["envelope_dB"] + 40,
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "synthetic.csv", index=False)
    print(
        "SYNTHETIC",
        d.loc[(d.start_ms == 2.3) & d.delay_ms.round(2).isin([0, 0.25, 1, 1.5, 2, 2.5])]
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    return d


def extract():
    manifest = pd.read_csv(
        ROOT / "datasets/\ubcf8\uc2e4\ud5d8_031-100/ultrasound/primary_recordings.csv",
        keep_default_na=False,
    )
    rows = []
    provenance = []
    for j in manifest.itertuples():
        p = ROOT / j.source.replace("\\", "/")
        if j.archive_member:
            with zipfile.ZipFile(p) as z:
                raw = z.read(j.archive_member)
        else:
            raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == j.sha256, (j.apple, "hash mismatch")
        data = pd.read_csv(io.BytesIO(raw))
        assert data.shape == (N, 3 * j.runs)
        air = {int(x) for x in j.air_runs.split(";")}
        for k in range(1, j.runs + 1):
            T = data[f"Time (s) Run #{k}"].to_numpy()
            A = data[f"Voltage, Ch A (V) Run #{k}"].to_numpy()
            B = data[f"Voltage, Ch B (V) Run #{k}"].to_numpy()
            assert np.isfinite(A).all() and np.isfinite(B).all()
            f = metrics(A, B, T)
            f.update(
                apple=j.apple,
                run=k,
                condition="air" if k in air else "apple",
                condition_status=j.condition_status,
                dt_us=np.median(np.diff(T)) * 1e6,
            )
            rows.append(f)
        provenance.append(
            dict(
                apple=j.apple,
                sha256=sha,
                runs=j.runs,
                source=j.source,
                archive_member=j.archive_member,
            )
        )
        if len(provenance) % 10 == 0:
            print("EXTRACTED", len(provenance), "files", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "raw_features.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "provenance.csv", index=False)
    old = pd.read_csv(ROOT / "output/us70_20261113/features_run_level.csv").query("set=='primary'")
    cmp = d.merge(old, on=["apple", "run"], suffixes=("_new", "_old"))
    delta = float(abs(cmp.level_rms_dB_new - cmp.level_rms_dB_old).max())
    print("REPRODUCTION_MAX_ABS_DB", delta, flush=True)
    return d, delta


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def analyse(r):
    app = r.query("condition=='apple'").groupby("apple").median(numeric_only=True)
    d = pd.read_csv(ROOT / "output/us70_20261113/model_matrix_primary.csv", index_col=0)
    for c in ["envelope_dB", "local_gap_dB", "broad_coherent_dB", "broad_gap_dB"]:
        d[c] = app[c]
    d.to_csv(OUT / "apple_metrics.csv")
    feats = [
        "level_rms_dB",
        "envelope_dB",
        "broad_coherent_dB",
        "local_gap_dB",
        "gd_ms",
        "acq_order",
    ]
    corr = []
    g = d.batch.to_numpy()
    perms = np.tile(np.arange(len(d)), (4999, 1))
    for i in range(len(perms)):
        for v in np.unique(g):
            ix = np.flatnonzero(g == v)
            perms[i, ix] = RNG.permutation(ix)
    for f in feats:
        for target in ["hardness", "brix"]:
            rho, p = spearmanr(d[f], d[target])
            x = pd.Series(rankdata(d[f]), index=d.index)
            y = pd.Series(rankdata(d[target]), index=d.index)
            x = (x - x.groupby(d.batch).transform("mean")).to_numpy()
            y = (y - y.groupby(d.batch).transform("mean")).to_numpy()
            rw = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
            null = (x[perms] @ y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
            pw = (1 + np.sum(abs(null) >= abs(rw))) / 5000
            corr.append(
                dict(
                    feature=f,
                    target=target,
                    rho=rho,
                    p=p,
                    rank_corr_batch_adjusted=rw,
                    p_within_batch_perm=pw,
                )
            )
    c = pd.DataFrame(corr)
    c["q_BH"] = bh(c.p_within_batch_perm)
    c.to_csv(OUT / "correlations.csv", index=False)
    print("CORRELATIONS\n", c.round(4).to_string(index=False), flush=True)
    predrows = []
    scores = []
    specs = {
        "NIR+geom": BASE,
        "NIR+geom+local": BASE + ["level_rms_dB"],
        "NIR+geom+envelope": BASE + ["envelope_dB"],
        "NIR+geom+envelope+delay": BASE + ["envelope_dB", "gd_ms"],
    }
    for target in ["hardness", "brix"]:
        for name, cols in specs.items():
            pred, _ = run_model(d, target, {"fixed": cols}, folds_for(d, "LOBO"))
            y = d[target].to_numpy()
            e = (y - pred) ** 2
            if name == "NIR+geom":
                base = e
            ix = RNG.integers(0, len(y), (5000, len(y)))
            bd = np.sqrt(e[ix].mean(1)) - np.sqrt(base[ix].mean(1))

            groups = np.unique(g)
            draws = RNG.choice(groups, (5000, len(groups)))
            be = np.array([e[g == v].mean() for v in groups])
            b0 = np.array([base[g == v].mean() for v in groups])
            gi = np.searchsorted(groups, draws)
            cluster = np.sqrt(be[gi].mean(1)) - np.sqrt(b0[gi].mean(1))
            scores.append(
                dict(
                    target=target,
                    model=name,
                    RMSE=np.sqrt(e.mean()),
                    delta_RMSE=np.sqrt(e.mean()) - np.sqrt(base.mean()),
                    apple_boot_lo=np.quantile(bd, 0.025),
                    apple_boot_hi=np.quantile(bd, 0.975),
                    batch_boot_lo=np.quantile(cluster, 0.025),
                    batch_boot_hi=np.quantile(cluster, 0.975),
                )
            )
            predrows.extend(
                dict(
                    apple=int(a),
                    batch=int(b),
                    target=target,
                    model=name,
                    actual=float(v),
                    predicted=float(p),
                )
                for a, b, v, p in zip(d.index, g, y, pred)
            )
    s = pd.DataFrame(scores)
    s.to_csv(OUT / "nested_lobo_scores.csv", index=False)
    pd.DataFrame(predrows).to_csv(OUT / "heldout_predictions.csv", index=False)
    print("LOBO\n", s.round(4).to_string(index=False), flush=True)
    return d, c, s


if __name__ == "__main__":
    synth = simulate()
    raw, delta = extract()
    d, c, s = analyse(raw)
    summary = dict(
        n_apples=int(raw.apple.nunique()),
        n_runs=len(raw),
        condition_counts=raw.condition.value_counts().to_dict(),
        reproduction_max_abs_dB=delta,
        primary_apple_gaps=raw.query("condition=='apple'")[["local_gap_dB", "broad_gap_dB"]]
        .describe()
        .to_dict(),
        primary_air_qc=raw.query("condition=='air'")[
            ["A_min", "A_max", "A_crest", "extreme_sample_fraction", "sat_frac"]
        ]
        .describe()
        .to_dict(),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", flush=True)
