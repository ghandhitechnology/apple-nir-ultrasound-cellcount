# Draws the summary figure of the general analysis. Panel (a) shows the cross-validated correlation
# of each input with each output, and panel (b) shows the strongest single-variable correlations
# with 95% confidence intervals.

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
M = pd.read_csv(OUT / "model_correlations.csv")
R = pd.read_csv(OUT / "correlation_screen.csv")
D = pd.read_csv(OUT / "mealiness_direct_models.csv")
plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))


def pick(o, i, c):
    s = M[(M.output == o) & (M.inputs == i) & M.calibration.str.contains(c, regex=False)]
    return abs(float(s.cv_r.iloc[0])), (float(s.within_lot_r.iloc[0]) if len(s) else np.nan)


rows = [
    ("당도 ← NIR", *pick("brix", "NIR", "final analysis")),
    ("당도 ← NIR (건물 함량 보조)", *pick("brix", "NIR", "dry matter")),
    ("당도 ← NIR+초음파", *pick("brix", "NIR+US", "final analysis")),
    ("당도 ← 초음파", *pick("brix", "US", "final analysis")),
    ("경도 ← NIR", *pick("hardness", "NIR", "final analysis")),
    ("경도 ← NIR+초음파", *pick("hardness", "NIR+US", "final analysis")),
    ("경도 ← 초음파", *pick("hardness", "US", "final analysis")),
    (
        "푸석함 ← NIR 560 nm (선택도 교차검증 안)",
        float(D[D.model.str.startswith("NIR best")].cv_r.iloc[0]),
        np.nan,
    ),
    ("푸석함 ← NIR (경도 대리지표)", *pick("mealiness (panel, 15)", "NIR", "puncture2 model")),
    (
        "푸석함 ← NIR+초음파 (경도 대리지표)",
        *pick("mealiness (panel, 15)", "NIR+US", "puncture2 model"),
    ),
    (
        "푸석함 ← 초음파 (특징 선택)",
        abs(float(D[D.model.str.startswith("US best")].cv_r.iloc[0])),
        np.nan,
    ),
]
y = np.arange(len(rows))[::-1]
cols = [
    "#1baf7a" if r[0].startswith("푸석함") else "#2a78d6" if r[0].startswith("당도") else "#eb6834"
    for r in rows
]
ax[0].barh(y, [r[1] for r in rows], color=cols, height=0.62)
for yy, r in zip(y, rows):
    ax[0].text(r[1] + 0.01, yy, f"{r[1]:.2f}", va="center", fontsize=8)
ax[0].axvline(0.6, color="#555", ls="--", lw=0.8)
ax[0].set_yticks(y)
ax[0].set_yticklabels([r[0] for r in rows])
ax[0].set_xlim(0, 0.85)
ax[0].set_xlabel("교차검증 예측 상관 |r| (사과 제외 예측 vs 실측)")
ax[0].set_title("(a) 입력 → 출력 모델", loc="left", fontweight="bold")

keep = [
    ("dry_matter", "brix", "건물 함량 – 당도"),
    ("dry_matter", "sens_mealy", "건물 함량 – 푸석함"),
    ("brix", "sens_mealy", "당도 – 푸석함"),
    ("nir_560nm", "sens_mealy", "NIR 560 nm – 푸석함"),
    ("nir_760nm", "sens_mealy", "NIR 760 nm – 푸석함"),
    ("nir_680nm", "mass", "NIR 680 nm – 과일 질량"),
    ("nir_645nm", "brix", "NIR 645 nm – 당도"),
    ("nir_730nm", "puncture2", "NIR 730 nm – 경도(2회차)"),
    ("puncture2", "sens_mealy", "경도(2회차) – 푸석함"),
    ("us_shape8_7", "sens_mealy", "초음파 shape8_7 – 푸석함"),
    ("us_msc_weighted", "brix", "초음파 msc – 당도"),
]
rr = []
for x_, y_, lab in keep:
    s = R[((R.x == x_) & (R.y == y_)) | ((R.x == y_) & (R.y == x_))].iloc[0]
    rr.append((lab, s.r, s.ci_lo, s.ci_hi, s.n, s.r_within_lot))
yb = np.arange(len(rr))[::-1]
for yy, (lab, r, lo, hi, n, rw) in zip(yb, rr):
    c = "#2a78d6" if n >= 60 else "#c8423b"
    ax[1].plot([lo, hi], [yy, yy], color=c, lw=1.6)
    ax[1].plot([r], [yy], "o", color=c, ms=6)
    ax[1].plot([rw], [yy], "x", color="#333", ms=5)
    ax[1].text(1.02, yy, f"{r:+.2f}  (n={int(n)})", va="center", fontsize=7.8)
ax[1].axvline(0, color="#999", lw=0.6)
ax[1].axvline(0.6, color="#555", ls="--", lw=0.7)
ax[1].axvline(-0.6, color="#555", ls="--", lw=0.7)
ax[1].set_yticks(yb)
ax[1].set_yticklabels([r[0] for r in rr])
ax[1].set_xlim(-1, 1.32)
ax[1].set_xticks([-1, -0.6, 0, 0.6, 1])
ax[1].set_xlabel("상관 r과 95 % 신뢰구간   ●: 전체   ×: 측정 묶음 안")
ax[1].set_title(
    "(b) 변수 사이 상관 (파랑 사과 70개, 빨강 관능 사과 15개)", loc="left", fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "summary.png", dpi=170)
plt.close()
print("ok")
