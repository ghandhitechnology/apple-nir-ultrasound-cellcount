# Draws the figures of the final report, with Korean labels.

import json

import matplotlib
from core import *
from core import _within_r

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BLUE, ORANGE, AQUA, GREY, INK, INK2, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#b4b3ad",
    "#0b0b0b",
    "#52514e",
    "#c8423b",
)
plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    }
)
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)
TG = pd.read_csv(ROOT / "targets.csv").set_index("apple").loc[APPLES]
cal = json.load(open(ROOT / "sensory_calibration.json"))
ch = cal["chosen"]
S = pd.read_csv(ROOT / "scores.csv")
HP = pd.read_csv(ROOT / "heldout_predictions.csv")
allr = sensory_ratings()
T = texture_index(allr).loc[SENS_APPLES]
st = TG.loc[SENS_APPLES]
cols = plt.cm.viridis(np.linspace(0.05, 0.9, 7))
BCOL = {g: cols[i] for i, g in enumerate(np.unique(B))}

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, x, title in [
    (ax[0], st.hardness, "기존 정답: 찌르기 2회 평균"),
    (ax[1], st.puncture2, "관능 보정: 두 번째 찌르기, 8.5에서 포화"),
]:
    a.scatter(
        x,
        T,
        s=42,
        color=[BCOL[TG.batch[i]] for i in SENS_APPLES],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for i in SENS_APPLES:
        a.annotate(
            str(i),
            (x[i], T[i]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
        )
    a.set_xlabel("경도")
    a.set_ylabel("관능 식감 점수 (높을수록 단단·다즙·아삭, 덜 푸석)")
    a.set_title(title, fontsize=10)
xx = np.linspace(5.5, 12, 200)
c, a0 = np.polyfit(st.hardness, T, 1)
ax[0].plot(xx, a0 + c * xx, color=GREY, lw=1.2)
ax[1].plot(xx, ch["a"] + ch["c"] * np.minimum(xx, ch["knee"]), color=ORANGE, lw=1.6)
ax[1].axvline(ch["knee"], color=ORANGE, lw=0.8, ls=":")
ref = cal["reference_loo_mse"]
var = cal["T_variance"]
ax[0].text(
    0.03,
    0.97,
    "LOO 오차 %.2f (점수 분산 %.2f)" % (ref["mean_of_punctures"], var),
    transform=ax[0].transAxes,
    va="top",
    fontsize=9,
)
ax[1].text(
    0.03, 0.97, "LOO 오차 %.2f" % ch["loo_mse"], transform=ax[1].transAxes, va="top", fontsize=9
)
plt.tight_layout()
plt.savefig(FIG / "fig1_sensory_calibration.png")
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
for a, (t, lab) in zip(
    ax, [("hardness", "경도"), ("hard_cal", "관능 보정 식감 점수"), ("brix", "당도 (°Brix)")]
):
    g = (
        HP[(HP.target == t) & (HP.family == "NIR") & (HP.scheme == "LOO")]
        .set_index("apple")
        .loc[APPLES]
    )
    for gv in np.unique(B):
        m = B == gv
        a.scatter(
            g.actual[m],
            g.predicted[m],
            s=24,
            color=BCOL[gv],
            edgecolor="white",
            linewidth=0.6,
            label=f"{10*gv+1}–{10*gv+10}",
        )
    lo, hi = min(g.actual.min(), g.predicted.min()), max(g.actual.max(), g.predicted.max())
    pad = 0.05 * (hi - lo)
    a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=1, zorder=0)
    a.set_xlim(lo - pad, hi + pad)
    a.set_ylim(lo - pad, hi + pad)
    s1 = S[(S.target == t) & (S.family == "NIR") & S.model.str.startswith("NESTED")].set_index(
        "scheme"
    )
    a.text(
        0.04,
        0.96,
        "LOO R² %.2f\n새 묶음(LOBO) R² %.2f\n묶음 안 상관 %.2f"
        % (s1.R2["LOO"], s1.R2["LOBO"], s1.within_r["LOO"]),
        transform=a.transAxes,
        va="top",
        fontsize=9,
        linespacing=1.5,
    )
    a.set_xlabel("실측 " + lab)
    a.set_ylabel("NIR 예측 (사과 제외)")
    a.set_title(lab)
ax[2].legend(title="사과 번호", fontsize=7.5, title_fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(FIG / "fig3_final_nir_predictions.png")
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
for a, (t, fam, lab) in zip(
    ax,
    [
        ("hardness", "NIR", "NIR → 경도"),
        ("hard_cal", "NIR", "NIR → 관능 보정 식감"),
        ("hard_cal", "NIR+US", "NIR+초음파 → 관능 보정 식감"),
    ],
):
    g = (
        HP[(HP.target == t) & (HP.family == fam) & (HP.scheme == "LOO")]
        .set_index("apple")
        .loc[SENS_APPLES]
    )
    a.scatter(
        g.predicted,
        T,
        s=40,
        color=[BCOL[TG.batch[i]] for i in SENS_APPLES],
        edgecolor="white",
        linewidth=0.7,
    )
    for i in SENS_APPLES:
        a.annotate(
            str(i),
            (g.predicted[i], T[i]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=INK2,
        )
    a.set_title(lab, fontsize=10)
    a.set_xlabel("예측값 (사과 제외)")
    a.set_ylabel("관능 식감 점수")
    a.text(
        0.04,
        0.96,
        "ρ = %.2f" % spearmanr(g.predicted, T)[0],
        transform=a.transAxes,
        va="top",
        fontsize=9.5,
    )
plt.tight_layout()
plt.savefig(FIG / "fig5_sensory_agreement.png")
plt.close()
print("figs 1, 3, 5 written")

P = ROOT / "permutation_summary.csv"
if P.exists():
    PS = pd.read_csv(P)
    raw = pd.read_csv(ROOT / "permutation_raw.csv")
    fams = ["NIR", "NIR (US teacher)", "NIR+US", "US"]
    FL = {
        "NIR": "NIR",
        "NIR (US teacher)": "NIR (초음파는 학습에만)",
        "NIR+US": "NIR+초음파",
        "US": "초음파만",
    }
    FC = {"NIR": BLUE, "NIR (US teacher)": AQUA, "NIR+US": ORANGE, "US": RED}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for a, (t, lab) in zip(
        ax, [("hardness", "경도"), ("hard_cal", "관능 보정 식감"), ("brix", "당도")]
    ):
        for j, sc in enumerate(["LOO", "LOBO"]):
            for i, f in enumerate(fams):
                v = S[
                    (S.target == t)
                    & (S.family == f)
                    & (S.scheme == sc)
                    & S.model.str.startswith("NESTED")
                ].R2.iloc[0]
                x = j * 5 + i
                a.bar(
                    x,
                    v,
                    color=FC[f],
                    width=0.8,
                    alpha=1 if sc == "LOO" else 0.55,
                    label=FL[f] if j == 0 else None,
                )
                a.text(
                    x,
                    v + 0.008 if v >= 0 else v - 0.008,
                    "%.2f" % v,
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5,
                    color=INK,
                )
        for j, sc in enumerate(["LOO", "LOBO"]):
            c_ = S[
                (S.target == t) & (S.model == "CONTROL acquisition time") & (S.scheme == sc)
            ].R2.iloc[0]
            a.plot([j * 5 - 0.5, j * 5 + 3.5], [c_, c_], color=INK2, lw=1, ls="--")
        a.set_xticks([1.5, 6.5])
        a.set_xticklabels(["사과 하나씩 제외 (LOO)", "묶음 하나씩 제외 (LOBO)"])
        a.axhline(0, color=INK2, lw=0.6)
        a.set_title(lab)
        a.set_ylabel("R²")
        lo_, hi_ = a.get_ylim()
        a.set_ylim(min(lo_, -0.05) - 0.05, hi_ + 0.03)
    ax[0].legend(fontsize=8, loc="upper right")
    fig.text(
        0.99,
        -0.01,
        "점선: 측정 시각만 쓴 대조군 (사과 정보 없이 측정 회차만 아는 경우)",
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK2,
    )
    plt.tight_layout()
    plt.savefig(FIG / "fig2_summary.png")
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for a, (t, lab) in zip(
        ax, [("hardness", "경도"), ("hard_cal", "관능 보정 식감"), ("brix", "당도")]
    ):
        for i, f in enumerate(["NIR+US", "NIR (US teacher)", "US"]):
            g = raw[
                (raw.kind == "us_rows")
                & (raw.target == t)
                & (raw.family == f)
                & (raw.scheme == "LOO")
            ].R2
            o = raw[
                (raw.kind == "observed")
                & (raw.target == t)
                & (raw.family == f)
                & (raw.scheme == "LOO")
            ].R2.iloc[0]
            a.violinplot([g], positions=[i], widths=0.7, showextrema=False)
            a.scatter([i], [o], color=ORANGE, zorder=3, s=40, marker="D")
        a.set_xticks([0, 1, 2])
        a.set_xticklabels(["NIR+초음파", "NIR\n(초음파 학습만)", "초음파만"], fontsize=8.5)
        a.set_title(lab)
        a.set_ylabel("LOO R²")
    ax[0].text(
        0.02,
        0.02,
        "◆ 실제 초음파   분포: 초음파 행을 묶음 안에서 섞은 100회",
        transform=ax[0].transAxes,
        fontsize=7.8,
        color=INK2,
    )
    plt.tight_layout()
    plt.savefig(FIG / "fig4_ultrasound_null.png")
    plt.close()
    print("figs 2, 4 written")
