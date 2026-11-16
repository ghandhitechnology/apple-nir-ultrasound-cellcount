# Loads every ultrasound recording of the 70 main-study apples and the six superseded originals and
# applies the shared transform. It writes run-level features, per-bin transfer arrays and a record
# of which file each run came from. Raw files are only read.

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import NB, N, process_run

R = Path("C:/apple_project")
OUT = Path(__file__).resolve().parents[1]
US = R / "datasets/본실험_031-100/ultrasound"
ZIPS = {
    "orig": R / "datasets/본실험_031-100/ultrasound/A31-A70_original.zip",
    "add": R / "datasets/ultrasound_intake_20261113/additional_ultrasound_original.zip",
    "rep": R / "datasets/ultrasound_replacements_20261113/replacement_original.zip",
}
stamp = {}
for k, z in ZIPS.items():
    for i in zipfile.ZipFile(z).infolist():
        stamp.setdefault(i.filename, "%04d-%02d-%02d %02d:%02d:%02d" % i.date_time)

prim = pd.read_csv(US / "primary_recordings.csv", keep_default_na=False)
prim["src"] = prim.source.str.replace(chr(92), "/", regex=False)
jobs = []
for r in prim.itertuples():
    member = r.archive_member or None
    name = member or Path(r.src).name
    jobs.append(
        dict(
            set="primary",
            apple=int(r.apple),
            file=name,
            path=r.src,
            member=member,
            n_runs=int(r.runs),
            air_runs=[int(x) for x in r.air_runs.split(";")],
            cond_status=r.condition_status,
        )
    )
for a in [31, 41, 48, 65, 69, 70]:
    jobs.append(
        dict(
            set="superseded",
            apple=a,
            file=f"A{a}.csv",
            path=str(ZIPS["orig"].relative_to(R)).replace(chr(92), "/"),
            member=f"A{a}.csv",
            n_runs=10,
            air_runs=[1, 2],
            cond_status="confirmed",
        )
    )

rows, bins, prov = [], [], []
for j in jobs:
    p = R / j["path"]
    raw = zipfile.ZipFile(p).read(j["member"]) if j["member"] else p.read_bytes()
    d = pd.read_csv(io.BytesIO(raw))
    assert d.shape == (N, 3 * j["n_runs"]), (j["file"], d.shape)
    prov.append(
        dict(
            set=j["set"],
            apple=j["apple"],
            file=j["file"],
            n_runs=j["n_runs"],
            condition_status=j["cond_status"],
            zip_time=stamp[j["file"]],
        )
    )
    for k in range(1, j["n_runs"] + 1):
        T = d[f"Time (s) Run #{k}"].to_numpy(float)
        A = d[f"Voltage, Ch A (V) Run #{k}"].to_numpy(float)
        B = d[f"Voltage, Ch B (V) Run #{k}"].to_numpy(float)
        assert np.isfinite(A).all() and np.isfinite(B).all()
        b, f = process_run(A, B, T)
        cond = "air" if k in j["air_runs"] else "apple"
        apple_idx = k - len(j["air_runs"])
        rows.append(
            dict(
                set=j["set"],
                apple=j["apple"],
                file=j["file"],
                run=k,
                condition=cond,
                apple_capture=apple_idx if cond == "apple" else 0,
                n_apple_runs=j["n_runs"] - len(j["air_runs"]),
                assumed_spot=((apple_idx - 1) % 4 + 1) if cond == "apple" else 0,
                condition_status=j["cond_status"],
                **f,
            )
        )
        bins.append(b)
    print(j["set"], j["file"], flush=True)

Rr = pd.DataFrame(rows)
pv = pd.DataFrame(prov).sort_values("zip_time")
pv["acq_order"] = np.arange(len(pv))
pv.sort_values(["set", "apple"]).to_csv(OUT / "file_provenance.csv", index=False)
air = (
    Rr[Rr.condition == "air"]
    .groupby(["set", "apple"])[
        ["gd_ms", "level_rms_dB", "level_med_dB", "B_rms_V", "leg_H", "leg_centroid"]
    ]
    .mean()
    .add_prefix("air_")
)
Rr = Rr.join(air, on=["set", "apple"])
Rr["gd_rel_ms"] = Rr.gd_ms - Rr.air_gd_ms
Rr["insertion_vs_air_dB"] = Rr.air_level_rms_dB - Rr.level_rms_dB
Rr = Rr.merge(pv[["set", "apple", "zip_time", "acq_order"]], on=["set", "apple"])
Rr.to_csv(OUT / "features_run_level.csv", index=False)
np.savez_compressed(
    OUT / "transfer_bins.npz",
    H=np.array([b["H"] for b in bins]),
    Hp=np.array([b["Hp"] for b in bins]),
    Amed=np.array([b["Amed"] for b in bins]),
    Amax=np.array([b["Amax"] for b in bins]),
    H3=np.array([b["H3"] for b in bins]),
)
print(Rr.groupby(["set", "condition"]).size())
