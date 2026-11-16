# Redraws four report figures at print size: the sensory calibration of the hardness label, the R2
# summary by input family, NIR cross-validated predictions against measured values, and held-out
# predictions against the sensory texture score. It reads only the saved results and writes PNG, PDF
# and SVG files to figs_paper/.

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as Outline
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs_paper"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
NEUTRAL = "#b4b3ad"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
SOFT = "#76746f"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WIDTH_IN = 7.0

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "font.size": 7,
        "axes.unicode_minus": False,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": SOFT,
        "axes.labelsize": 7,
        "axes.labelpad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
    }
)

PREDICTIONS = pd.read_csv(ROOT / "heldout_predictions.csv")
SCORES = pd.read_csv(ROOT / "scores.csv")
TARGETS = pd.read_csv(ROOT / "targets.csv").set_index("apple")
TASTED = TARGETS[TARGETS.texture.notna()]
CALIBRATION = json.loads((ROOT / "sensory_calibration.json").read_text(encoding="utf-8"))


def heldout(target, family):
    rows = PREDICTIONS[
        (PREDICTIONS.target == target)
        & (PREDICTIONS.family == family)
        & (PREDICTIONS.scheme == "LOO")
    ]
    return rows.set_index("apple")


def panel_title(ax, letter, text):
    ax.text(0, 1.04, letter, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK2)
    ax.text(0.07, 1.04, text, transform=ax.transAxes, fontsize=7.5, color=INK2)


def stat_rows(ax, rows, x_label, x_value, y_top, step, align="bottom"):
    ax.add_patch(
        Rectangle(
            (x_label - 0.03, y_top - (len(rows) - 1) * step - 0.025),
            x_value - x_label + 0.05,
            len(rows) * step + 0.03,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )
    for index, (label, value) in enumerate(rows):
        y = y_top - index * step
        style = dict(transform=ax.transAxes, va=align, fontsize=6.5, zorder=5)
        ax.text(x_label, y, label, ha="left", color=MUTED, **style)
        ax.text(x_value, y, value, ha="right", color=INK2, **style)


def save(fig, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def predicted_vs_measured():
    panels = [
        ("hardness", "경도", "실측 경도", 1.0),
        ("hard_cal", "관능 보정 식감 점수", "실측 식감 점수", 0.5),
        ("brix", "당도", "실측 당도 (°Brix)", 2.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.72))
    fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.155, wspace=0.27)
    for ax, letter, (target, title, x_label, tick_step) in zip(axes, "abc", panels):
        data = heldout(target, "NIR").loc[TARGETS.index]
        low = min(data.actual.min(), data.predicted.min())
        high = max(data.actual.max(), data.predicted.max())
        pad = 0.06 * (high - low)
        low, high = low - pad, high + pad
        ax.plot([low, high], [low, high], color=AXIS, linewidth=0.9, zorder=1)
        ax.scatter(
            data.actual,
            data.predicted,
            s=17,
            color=BLUE,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.set_xlabel(x_label)
        panel_title(ax, letter, title)

        nested = SCORES[
            (SCORES.target == target)
            & (SCORES.family == "NIR")
            & SCORES.model.str.startswith("NESTED")
        ].set_index("scheme")
        stat_rows(
            ax,
            [
                ("LOO R²", f"{nested.R2['LOO']:.2f}"),
                ("LOBO R²", f"{nested.R2['LOBO']:.2f}"),
                ("묶음 안 r", f"{nested.within_r['LOO']:.2f}"),
            ],
            x_label=0.655,
            x_value=0.975,
            y_top=0.185,
            step=0.072,
        )
        corner = low + 0.90 * (high - low)
        if letter == "a":
            ax.text(
                corner,
                corner + 0.035 * (high - low),
                "예측 = 실측",
                rotation=45,
                rotation_mode="anchor",
                ha="right",
                va="bottom",
                fontsize=6,
                color=MUTED,
            )

        if target == "hard_cal":
            ceiling = data.actual.max()
            at_ceiling = data[data.actual >= ceiling - 1e-9]
            ax.annotate(
                f"실측 상한\n사과 {len(at_ceiling)}개",
                xy=(ceiling, at_ceiling.predicted.min() - 0.03),
                xytext=(ceiling - 0.02, at_ceiling.predicted.min() - 0.33),
                ha="center",
                va="top",
                fontsize=6,
                color=MUTED,
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=1, shrinkB=2),
            )
    axes[0].set_ylabel("NIR 예측값 (교차검증)")
    save(fig, "nir_cv_predicted_vs_measured")


def place_labels(ax, xs, ys, names, fontsize=5.8, marker_radius=3.2, avoid=()):
    fig = ax.figure
    fig.canvas.draw()
    scale = 72.0 / fig.dpi
    points = ax.transData.transform(np.column_stack([xs, ys])) * scale
    blocked = [ax.transData.transform(np.asarray(line)) * scale for line in avoid]
    box = ax.get_window_extent()
    left, bottom, right, top = box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale
    height = fontsize * 0.95
    widths = [0.52 * fontsize * len(name) for name in names]
    near = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    directions = [(dx, dy, reach) for reach in (1.0, 2.4) for dx, dy in near]

    def rectangle(index, direction):
        dx, dy, reach = direction
        gap = (marker_radius + 1.0) * reach
        x, y = points[index]
        width = widths[index]
        x0 = x + gap * 0.75 if dx > 0 else x - gap * 0.75 - width if dx < 0 else x - width / 2
        y0 = y + gap * 0.45 if dy > 0 else y - gap * 0.45 - height if dy < 0 else y - height / 2
        if dx == 0:
            y0 = y + gap if dy > 0 else y - gap - height
        if dy == 0:
            x0 = x + gap if dx > 0 else x - gap - width
        return x0, y0, x0 + width, y0 + height

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def cost(index, direction, chosen):
        rect = rectangle(index, direction)
        total = 0.02 * directions.index(direction) + 3.0 * (direction[2] - 1.0)
        for other, (x, y) in enumerate(points):
            if other == index:
                continue
            r = marker_radius + 0.6
            total += 6.0 * overlap(rect, (x - r, y - r, x + r, y + r))
            if other in chosen:
                total += 8.0 * overlap(rect, rectangle(other, chosen[other]))
        for line in blocked:
            inside = (
                (line[:, 0] > rect[0] - 1.5)
                & (line[:, 0] < rect[2] + 1.5)
                & (line[:, 1] > rect[1] - 1.5)
                & (line[:, 1] < rect[3] + 1.5)
            )
            total += 60.0 * inside.sum()
        if rect[0] < left + 1 or rect[2] > right - 1 or rect[1] < bottom + 1 or rect[3] > top - 1:
            total += 500.0
        return total

    chosen = {}
    order = sorted(range(len(names)), key=lambda i: -sum(np.hypot(*(points - points[i]).T) < 22))
    for _ in range(4):
        for index in order:
            chosen[index] = min(directions, key=lambda d: cost(index, d, chosen))

    for index, name in enumerate(names):
        x0, y0, x1, y1 = rectangle(index, chosen[index])
        ax.annotate(
            name,
            xy=(xs[index], ys[index]),
            xytext=((x0 + x1) / 2 - points[index][0], (y0 + y1) / 2 - points[index][1]),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=MUTED,
            zorder=4,
            arrowprops=(
                dict(arrowstyle="-", color=MUTED, linewidth=0.4, shrinkA=0.5, shrinkB=3.4)
                if chosen[index][2] > 1
                else None
            ),
        )


def sensory_agreement():
    panels = [
        ("hardness", "NIR", "NIR → 경도", "예측 경도"),
        ("hard_cal", "NIR", "NIR → 관능 보정 식감", "예측 식감 점수"),
        ("hard_cal", "NIR+US", "NIR+초음파 → 관능 보정 식감", "예측 식감 점수"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.78), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.155, wspace=0.09)
    texture = TASTED.texture
    for ax, letter, (target, family, title, x_label) in zip(axes, "abc", panels):
        predicted = heldout(target, family).loc[TASTED.index].predicted
        ax.scatter(predicted, texture, s=30, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        span = predicted.max() - predicted.min()
        ax.set_xlim(predicted.min() - 0.09 * span, predicted.max() + 0.09 * span)
        ax.set_ylim(texture.min() - 0.3, texture.max() + 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_xlabel(x_label)
        panel_title(ax, letter, title)
        rho = spearmanr(predicted, texture)[0]
        ax.text(
            0.04,
            0.955,
            f"Spearman ρ = {rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=INK2,
        )
        place_labels(ax, predicted.to_numpy(), texture.to_numpy(), [str(a) for a in TASTED.index])
    axes[0].set_ylabel("관능 식감 점수")
    save(fig, "sensory_agreement")


def sensory_calibration():
    chosen = CALIBRATION["chosen"]
    texture = TASTED.texture
    knee = chosen["knee"]
    grid = np.linspace(5.0, 12.5, 400)
    slope, intercept = np.polyfit(TASTED.hardness, texture, 1)
    panels = [
        (
            TASTED.hardness,
            "기존 정답: 찌르기 2회 평균",
            "경도 (찌르기 2회 평균)",
            intercept + slope * grid,
            [
                ("LOO 오차", f"{CALIBRATION['reference_loo_mse']['mean_of_punctures']:.2f}"),
                ("점수 분산", f"{CALIBRATION['T_variance']:.2f}"),
            ],
        ),
        (
            TASTED.puncture2,
            f"관능 보정: 두 번째 찌르기, {knee:g}에서 포화",
            "경도 (두 번째 찌르기)",
            chosen["a"] + chosen["c"] * np.minimum(grid, knee),
            [("LOO 오차", f"{chosen['loo_mse']:.2f}")],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.95), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.992, top=0.91, bottom=0.145, wspace=0.07)
    for ax, letter, (hardness, title, x_label, fitted, rows) in zip(axes, "ab", panels):
        ax.plot(grid, fitted, color=MUTED, linewidth=1.0, solid_capstyle="round", zorder=2)
        ax.scatter(hardness, texture, s=30, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xlim(5.9, 12.1)
        ax.set_ylim(texture.min() - 0.35, texture.max() + 0.35)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_xlabel(x_label)
        ax.text(
            0, 1.04, letter, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK2
        )
        ax.text(0.04, 1.04, title, transform=ax.transAxes, fontsize=7.5, color=INK2)
        ax.add_patch(
            Rectangle(
                (0.015, 0.975 - 0.065 * len(rows)),
                0.26,
                0.065 * len(rows),
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        )
        for index, (label, value) in enumerate(rows):
            y = 0.955 - 0.065 * index
            style = dict(transform=ax.transAxes, va="top", fontsize=6.5, zorder=5)
            ax.text(0.03, y, label, ha="left", color=MUTED, **style)
            ax.text(0.20, y, value, ha="left", color=INK2, **style)
        place_labels(
            ax,
            hardness.to_numpy(),
            texture.to_numpy(),
            [str(a) for a in TASTED.index],
            avoid=[np.column_stack([grid, fitted])],
        )
    axes[1].axvline(knee, color=AXIS, linewidth=0.7, zorder=1)
    axes[1].text(
        knee + 0.06,
        texture.min() - 0.28,
        f"포화점 {knee:g}",
        ha="left",
        va="bottom",
        fontsize=6,
        color=MUTED,
    )
    axes[0].set_ylabel("관능 식감 점수")
    save(fig, "sensory_calibration")


def rounded_bar(ax, value, centre, thickness, color, radius_pt=1.6):
    box = ax.get_position()
    width_pt = box.width * ax.figure.get_figwidth() * 72
    height_pt = box.height * ax.figure.get_figheight() * 72
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    rx = min(radius_pt * abs(x_high - x_low) / width_pt, abs(value))
    ry = min(radius_pt * abs(y_high - y_low) / height_pt, thickness / 2)
    rx = rx if value >= 0 else -rx
    bottom, top = centre - thickness / 2, centre + thickness / 2
    vertices = [
        (0, bottom),
        (value - rx, bottom),
        (value, bottom),
        (value, bottom + ry),
        (value, top - ry),
        (value, top),
        (value - rx, top),
        (0, top),
        (0, bottom),
    ]
    codes = [
        Outline.MOVETO,
        Outline.LINETO,
        Outline.CURVE3,
        Outline.CURVE3,
        Outline.LINETO,
        Outline.CURVE3,
        Outline.CURVE3,
        Outline.LINETO,
        Outline.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Outline(vertices, codes), facecolor=color, edgecolor="none", zorder=3))


def score_summary():
    series = [
        ("NIR", "NIR", BLUE),
        ("NIR (US teacher)", "NIR (초음파는 학습에만)", ORANGE),
        ("NIR+US", "NIR + 초음파", AQUA),
        ("US", "초음파만", YELLOW),
        ("control", "대조: 측정 시각만", NEUTRAL),
    ]
    schemes = [("LOO", "사과 하나씩 제외 (LOO)"), ("LOBO", "묶음 하나씩 제외 (LOBO)")]
    targets = [("hardness", "경도"), ("hard_cal", "관능 보정 식감"), ("brix", "당도")]
    group_height = len(series) + 1.7
    thickness = 0.74

    def r2(target, family, scheme):
        rows = SCORES[(SCORES.target == target) & (SCORES.scheme == scheme)]
        if family == "control":
            return rows[rows.model == "CONTROL acquisition time"].R2.iloc[0]
        return rows[(rows.family == family) & rows.model.str.startswith("NESTED")].R2.iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.62), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.215, right=0.992, top=0.915, bottom=0.135, wspace=0.07)
    positions = []
    labels = []
    for group, (scheme, heading) in enumerate(schemes):
        base = group * group_height
        axes[0].text(
            -0.035,
            base,
            heading,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.5,
            color=INK2,
        )
        for row, (_, name, _) in enumerate(series):
            positions.append(base + 1 + row)
            labels.append(name)

    for ax, letter, (target, title) in zip(axes, "abc", targets):
        ax.set_xlim(-0.27, 0.58)
        ax.set_ylim(positions[-1] + 0.75, -0.6)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=AXIS, linewidth=0.7, zorder=2)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_xlabel("R²")
        ax.text(
            0, 1.035, letter, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK2
        )
        ax.text(0.07, 1.035, title, transform=ax.transAxes, fontsize=7.5, color=INK2)
        for group, (scheme, _) in enumerate(schemes):
            for row, (family, _, color) in enumerate(series):
                value = r2(target, family, scheme)
                centre = group * group_height + 1 + row
                rounded_bar(ax, value, centre, thickness, color)
                ax.annotate(
                    f"{value:.2f}",
                    xy=(value, centre),
                    xytext=(2.5 if value >= 0 else -2.5, -0.2),
                    textcoords="offset points",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=5.8,
                    color=SOFT,
                )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis="y", pad=5)
    save(fig, "score_summary")


sensory_calibration()
score_summary()
predicted_vs_measured()
sensory_agreement()
print("written to", OUT)
