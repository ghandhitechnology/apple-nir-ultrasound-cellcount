# Draws the journal figures for the manuscript draft, typeset in Latin Modern through matplotlib's
# PGF backend. Run it as: python figs.py [fig3 fig4 ...]; it writes a vector PDF and a 600 dpi PNG
# preview for each figure.

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["PATH"] = r"C:\texlive\2026\bin\windows;" + os.environ["PATH"]
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)
RAND = Path("C:/apple_project")
O = RAND / "output"
FA, GA, CM = (
    O / "final_analysis_20261114",
    O / "general_analysis_20261114",
    O / "cells_main_20261114",
)
CW = Path("D:/rando-testing/cell_count_2026-11-14_main")

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "text.usetex": True,
        "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{textcomp}",
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.titlesize": 8,
    }
)
K, G1, G2, ACC, ACC2 = "black", "0.45", "0.75", "#1f4e79", "#a6261c"
W2, W1 = 7.0, 3.4


def LOT(b):
    return np.asarray(b) - 2


SENS = [37, 39, 40, 42, 49, 51, 63, 65, 69, 70, 81, 88, 93, 94, 99]
T = pd.read_csv(GA / "data_table.csv", index_col=0)


def plab(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, f"({s})", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)


def save(fig, name):
    p = FIG / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    subprocess.run(
        ["pdftocairo", "-png", "-r", "300", "-singlefile", str(p), str(FIG / name)], check=True
    )
    print("saved", name)


def fit_line(ax, x, y, **kw):
    m = ~(np.isnan(x) | np.isnan(y))
    b = np.polyfit(x[m], y[m], 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(
        xx,
        np.polyval(b, xx),
        color=kw.pop("color", G1),
        lw=kw.pop("lw", 0.7),
        ls=kw.pop("ls", "-"),
        zorder=1,
    )


def rtxt(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    r = np.corrcoef(x[m], y[m])[0, 1]
    return r, int(m.sum())


def fig3():
    fig = plt.figure(figsize=(W2, 1.85))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.95, 1.0], wspace=0.5)
    ax = [fig.add_subplot(gs[i]) for i in (0, 1, 2, 4)]
    lot = LOT(T.batch)
    rng = np.random.default_rng(1)
    for a, col, lab in [
        (ax[0], "brix", r"SSC ($^{\circ}$Brix)"),
        (ax[1], "hardness", "Puncture firmness"),
    ]:
        for L_ in range(1, 8):
            v = T.loc[lot == L_, col].to_numpy()
            a.plot(
                L_ + rng.uniform(-0.18, 0.18, len(v)), v, "o", mfc="none", mec=G1, mew=0.5, ms=2.6
            )
            a.errorbar(
                L_ + 0.33,
                np.nanmean(v),
                yerr=np.nanstd(v, ddof=1),
                fmt="s",
                color=K,
                ms=2.4,
                lw=0.6,
                capsize=1.2,
                capthick=0.6,
            )
        a.set_xticks(range(1, 8))
        a.set_xlabel("Measurement lot")
        a.set_ylabel(lab)
        a.set_xlim(0.4, 7.8)
    f = stats.f_oneway(*[T.loc[lot == L_, "hardness"].dropna() for L_ in range(1, 8)])
    x, y = T.dry_matter.to_numpy(float), T.brix.to_numpy(float)
    ax[2].plot(x, y, "o", mfc="none", mec=K, mew=0.5, ms=2.8)
    fit_line(ax[2], x, y, color=K)
    r, n = rtxt(x, y)
    ax[2].text(
        0.95,
        0.05,
        rf"$r = {r:.2f}$" + "\n" + rf"$n = {n}$",
        transform=ax[2].transAxes,
        va="bottom",
        ha="right",
        linespacing=1.3,
    )
    ax[2].set_xlabel(r"Dry matter (\%)")
    ax[2].set_ylabel(r"SSC ($^{\circ}$Brix)")
    rel = pd.read_csv(GA / "reliabilities.csv", index_col=0).reliability
    nir = rel[[i for i in rel.index if i.startswith("nir_") and i.endswith("nm")]]
    us = pd.read_csv(O / "us70_20261113/repeatability_icc.csv").set_index("feature").ICC_mean8_70
    items = [
        ("SSC (2 readings)", rel["brix"]),
        ("Firmness (2 punctures)", rel["hardness"]),
        ("Firmness, 2nd puncture", rel["puncture2"]),
        (r"Dry matter$^{\dagger}$", rel["dry_matter"]),
        ("Mealiness (panel)", rel["sens_mealy"]),
        ("Juiciness (panel)", rel["sens_juicy"]),
        ("Crispness (panel)", rel["sens_crisp"]),
        ("Cell count (2 fields)", rel["cell_count"]),
        ("Intact share (2 fields)", rel["intact_share"]),
    ]
    yy = np.arange(len(items))[::-1]
    ax[3].plot([v for _, v in items], yy, "o", color=K, ms=3)
    ax[3].hlines(-1.2, nir.min(), nir.max(), color=K, lw=1.6)
    ax[3].plot([nir.median()], [-1.2], "|", color="white", ms=3)
    ax[3].hlines(-2.4, us.min(), us.max(), color=G1, lw=1.6)
    ax[3].set_yticks(list(yy) + [-1.2, -2.4])
    ax[3].set_yticklabels(
        [s for s, _ in items] + ["NIR channels (12 scans)", "US features (8 runs)"], fontsize=6.5
    )
    ax[3].set_xlim(0, 1.02)
    ax[3].set_xlabel("Reliability of the apple value")
    ax[3].tick_params(axis="y", length=0)
    ax[3].set_ylim(-3.1, len(items) - 0.4)
    ax[3].axvline(0.7, color=G2, lw=0.5, ls=":")
    for a, s in zip(ax, "abcd"):
        plab(a, s, x=-0.30 if s != "d" else -1.25)
    save(fig, "fig3_reference")
    return dict(anova_hardness_lot_F=f.statistic, p=f.pvalue)


def fig4():
    sys.path.insert(0, str(FA / "scripts"))
    import core

    allr = core.sensory_ratings()
    for a in ["mealy", "juicy", "crisp"]:
        g = allr.groupby(["panel", "rnd"])[a]
        allr[a + "_z"] = ((allr[a] - g.transform("mean")) / g.transform("std")).fillna(0)
    r1 = allr[allr.rnd == 1].groupby("apple")[["mealy_z", "juicy_z", "crisp_z"]].mean()
    r2 = allr[allr.rnd == 2].groupby("apple")[["mealy_z", "juicy_z", "crisp_z"]].mean()
    tex = core.texture_index(allr).loc[SENS]
    tg = pd.read_csv(FA / "targets.csv").set_index("apple").loc[SENS]
    cal = json.load(open(FA / "sensory_calibration.json"))["chosen"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), gridspec_kw=dict(wspace=0.42))
    for c, mk, lab in [
        ("mealy_z", "o", "Mealiness"),
        ("juicy_z", "s", "Juiciness"),
        ("crisp_z", "^", "Crispness"),
    ]:
        r = np.corrcoef(r1.loc[SENS, c], r2.loc[SENS, c])[0, 1]
        ax[0].plot(
            r1.loc[SENS, c],
            r2.loc[SENS, c],
            mk,
            mfc="none" if c != "mealy_z" else K,
            mec=K,
            mew=0.5,
            ms=3,
            label=rf"{lab} ($r={r:.2f}$)",
        )
    lim = [-1.6, 2.3]
    ax[0].plot(lim, lim, ls="--", color=G2, lw=0.6, zorder=0)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel("Session 1 score ($z$)")
    ax[0].set_ylabel("Session 2 score ($z$)")
    ax[0].legend(loc="upper left", fontsize=6.3, handletextpad=0.3)
    for a, x, name in [
        (ax[1], tg.hardness.to_numpy(), "Firmness, mean of 2 punctures"),
        (ax[2], tg.puncture2.to_numpy(), "Firmness, 2nd puncture"),
    ]:
        a.plot(x, tex, "o", mfc="none", mec=K, mew=0.5, ms=3)
        a.set_xlabel(name)
        a.set_ylabel("Texture score")
    fit_line(ax[1], tg.hardness.to_numpy(), tex.to_numpy(), color=G1, ls="--")
    xx = np.linspace(6, 12, 200)
    ax[2].plot(xx, cal["a"] + cal["c"] * np.minimum(xx, cal["knee"]), color=K, lw=0.8)
    ax[2].axvline(cal["knee"], color=G2, lw=0.5, ls=":")
    ref = json.load(open(FA / "sensory_calibration.json"))["reference_loo_mse"]
    ax[1].text(
        0.04,
        0.95,
        rf'LOO MSE $= {ref["mean_of_punctures"]:.2f}$',
        transform=ax[1].transAxes,
        va="top",
        fontsize=7,
    )
    ax[2].text(
        0.04,
        0.95,
        rf'LOO MSE $= {cal["loo_mse"]:.2f}$',
        transform=ax[2].transAxes,
        va="top",
        fontsize=7,
    )
    ax[2].text(
        cal["knee"] + 0.1, ax[2].get_ylim()[0] + 0.1, f'{cal["knee"]:.1f}', fontsize=6.5, color=G1
    )
    for a, s in zip(ax, "abc"):
        plab(a, s, x=-0.24)
    save(fig, "fig4_sensory")


def fig5():
    HP = pd.read_csv(FA / "heldout_predictions.csv")
    SC = pd.read_csv(FA / "scores.csv")
    nir = [c for c in T.columns if c.startswith("nir_") and c.endswith("nm")]
    wl = np.array([int(c[4:-2]) for c in nir])
    fig, ax = plt.subplots(
        1, 3, figsize=(W2, 2.05), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.38)
    )
    q = pd.qcut(T.hardness, 3, labels=["soft", "mid", "firm"])
    for lab, ls, c in [("soft", "-", K), ("mid", ":", G1), ("firm", "--", K)]:
        X = T.loc[q == lab, nir].to_numpy(float)
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax[0].fill_between(wl, m - se, m + se, color=G2, lw=0, alpha=0.6)
        ax[0].plot(wl, m, ls=ls, color=c, marker="o", ms=1.8, label=f"{lab} tertile ($n={len(X)}$)")
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].set_ylabel("SNV of log signal")
    ax[0].legend(loc="lower right", fontsize=6.3)
    for a, t, lab in [
        (ax[1], "hardness", "puncture firmness"),
        (ax[2], "brix", r"SSC ($^{\circ}$Brix)"),
    ]:
        h = HP[
            (HP.target == t)
            & (HP.family == "NIR")
            & (HP.scheme == "LOO")
            & HP.model.str.startswith("NESTED")
        ]
        a.plot(h.actual, h.predicted, "o", mfc="none", mec=K, mew=0.5, ms=2.8)
        lo, hi = min(h.actual.min(), h.predicted.min()), max(h.actual.max(), h.predicted.max())
        pad = 0.05 * (hi - lo)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=G2, lw=0.6, zorder=0)
        a.set_xlim(lo - pad, hi + pad)
        a.set_ylim(lo - pad, hi + pad)
        s = SC[
            (SC.target == t) & (SC.family == "NIR") & SC.model.str.startswith("NESTED")
        ].set_index("scheme")
        a.text(
            0.04,
            0.96,
            "\n".join(
                [
                    rf'$R^2_{{\mathrm{{LOO}}}} = {s.R2["LOO"]:.2f}$',
                    rf'$R^2_{{\mathrm{{LOLO}}}} = {s.R2["LOBO"]:.2f}$',
                    rf'$r_{{\mathrm{{within}}}} = {s.within_r["LOO"]:.2f}$',
                ]
            ),
            transform=a.transAxes,
            va="top",
            fontsize=6.8,
            linespacing=1.3,
        )
        a.set_xlabel("Measured " + lab)
        a.set_ylabel("Predicted (held out)")
    for a, s in zip(ax, "abc"):
        plab(a, s, x=-0.22)
    save(fig, "fig5_nir")


def fig6():
    SC = pd.read_csv(FA / "scores.csv")
    fams = [
        ("NIR", "NIR"),
        ("NIR (US teacher)", "NIR, US as teacher"),
        ("NIR+US", "NIR + US"),
        ("US", "US"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=False, gridspec_kw=dict(wspace=0.3))
    for a, (t, lab) in zip(
        ax, [("hardness", "Firmness"), ("hard_cal", "Panel-calibrated texture"), ("brix", "SSC")]
    ):
        for j, sc in enumerate(["LOO", "LOBO"]):
            for i, (f, _) in enumerate(fams):
                v = SC[
                    (SC.target == t)
                    & (SC.family == f)
                    & (SC.scheme == sc)
                    & SC.model.str.startswith("NESTED")
                ].R2.iloc[0]
                x = j * 5 + i
                a.bar(
                    x, v, width=0.78, color=["0.15", "0.45", "0.65", "0.9"][i], edgecolor=K, lw=0.5
                )
            c_ = SC[
                (SC.target == t) & (SC.model == "CONTROL acquisition time") & (SC.scheme == sc)
            ].R2
            if len(c_):
                a.plot([j * 5 - 0.5, j * 5 + 3.5], [c_.iloc[0]] * 2, ls="--", color=ACC2, lw=0.7)
        a.axhline(0, color=K, lw=0.5)
        a.set_xticks([1.5, 6.5])
        a.set_xticklabels(["LOO", "LOLO"])
        a.tick_params(axis="x", length=0)
        a.set_title(lab, fontsize=7.5, pad=3)
        a.set_ylabel(r"$R^2$ (held out)" if t == "hardness" else "")
        a.set_ylim(-0.2, 0.55)
    h = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=["0.15", "0.45", "0.65", "0.9"][i], edgecolor=K, lw=0.5
        )
        for i in range(4)
    ]
    h += [Line2D([], [], ls="--", color=ACC2, lw=0.7)]
    fig.legend(
        h,
        [s for _, s in fams] + ["acquisition time only"],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.1),
        fontsize=6.5,
    )
    for a, s in zip(ax, "abc"):
        plab(a, s, x=-0.2 if s == "a" else -0.1)
    save(fig, "fig6_models")


def fig7():
    PR = pd.read_csv(FA / "permutation_raw.csv")
    fig = plt.figure(figsize=(W2, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.75, 0.6, 1.0], wspace=0.5)
    ax = [fig.add_subplot(gs[i]) for i in (0, 1, 3)]
    pos = 0
    ticks = []
    for t, lab in [("hardness", "Firmness"), ("hard_cal", "Texture"), ("brix", "SSC")]:
        nul = PR[(PR.kind == "label") & (PR.target == t) & (PR.family == "NIR")].R2
        obs = pd.read_csv(FA / "scores.csv").query(
            "target == @t and family == 'NIR' and scheme == 'LOO'"
        )
        obs = obs[obs.model.str.startswith("NESTED")].R2.iloc[0]
        ax[0].boxplot(
            [nul],
            positions=[pos * 1.7],
            widths=0.6,
            showfliers=False,
            medianprops=dict(color=K, lw=0.7),
            boxprops=dict(lw=0.5),
            whiskerprops=dict(lw=0.5),
            capprops=dict(lw=0.5),
        )
        ax[0].plot([pos * 1.7], [obs], "D", color=K, ms=3.2)
        ticks.append(lab)
        pos += 1
    ax[0].set_xticks([0, 1.7, 3.4])
    ax[0].set_xticklabels(ticks, fontsize=6.5, rotation=30, ha="right", rotation_mode="anchor")
    ax[0].set_xlim(-0.65, 4.05)
    ax[0].set_ylabel(r"LOO $R^2$")
    ax[0].tick_params(axis="x", length=0)
    pos = 0
    ticks = []
    for t, tl in [("hardness", "Firmness"), ("brix", "SSC")]:
        for f, fl in [("US", "US"), ("NIR+US", "N+U"), ("NIR (US teacher)", "N(U)")]:
            nul = PR[
                (PR.kind == "us_rows") & (PR.target == t) & (PR.family == f) & (PR.scheme == "LOO")
            ].R2
            obs = PR[
                (PR.kind == "observed") & (PR.target == t) & (PR.family == f) & (PR.scheme == "LOO")
            ].R2.iloc[0]
            ax[1].boxplot(
                [nul],
                positions=[pos * 1.3],
                widths=0.6,
                showfliers=False,
                medianprops=dict(color=K, lw=0.7),
                boxprops=dict(lw=0.5),
                whiskerprops=dict(lw=0.5),
                capprops=dict(lw=0.5),
            )
            ax[1].plot([pos * 1.3], [obs], "D", color=K, ms=3.2)
            ticks.append(fl)
            pos += 1
        pos += 0.6
    ax[1].set_xticks([p * 1.3 for p in [0, 1, 2, 3.6, 4.6, 5.6]])
    ax[1].set_xticklabels(ticks, fontsize=5.8)
    ax[1].tick_params(axis="x", length=0)
    tr = mpl.transforms.blended_transform_factory(ax[1].transData, ax[1].transAxes)
    ax[1].text(1.3, -0.16, "Firmness", ha="center", va="top", fontsize=7, transform=tr)
    ax[1].text(4.6 * 1.3, -0.16, "SSC", ha="center", va="top", fontsize=7, transform=tr)
    ax[1].set_ylabel(r"LOO $R^2$")
    ax[1].set_ylim(-0.25, 0.6)
    icc = pd.read_csv(O / "us70_20261113/repeatability_icc.csv").set_index("feature")
    icc = icc.sort_values("ICC_mean8_70")
    yy = np.arange(len(icc))
    ax[2].hlines(yy, icc.ICC_single_70, icc.ICC_mean8_70, color=G2, lw=0.6)
    ax[2].plot(icc.ICC_single_70, yy, "o", mfc="none", mec=K, mew=0.5, ms=2.6, label="single run")
    ax[2].plot(icc.ICC_mean8_70, yy, "o", color=K, ms=2.6, label="mean of 8 runs")
    NAME = {
        "A_rms_V": "RMS amplitude",
        "level_med_dB": "median level",
        "insertion_vs_air_dB": "insertion loss vs air",
        "level_rms_dB": "RMS level",
        "sat_frac": "clipped fraction",
        "msc_weighted": "coherence (MSC)",
        "gd_ms": "group delay",
        "gd_rel_ms": "group delay, relative",
        "gd_hi_ms": "group delay, upper band",
        "roughness_dB": "spectral roughness",
        "tilt_dB": "spectral tilt",
        "gd_lo_ms": "group delay, lower band",
    }
    ax[2].set_yticks(yy)
    ax[2].set_yticklabels([NAME.get(s, s) for s in icc.index], fontsize=6.2)
    ax[2].tick_params(axis="y", length=0)
    ax[2].set_xlim(0, 1)
    ax[2].set_xlabel("ICC")
    ax[2].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=6,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    for a, s in zip(ax, "abc"):
        plab(a, s, x=-0.25 if s != "c" else -0.75)
    save(fig, "fig7_nulls_us")


def fig8():
    I = pd.read_csv(CM / "cell_counts_per_image.csv")
    A = pd.read_csv(CM / "cell_counts_per_apple.csv").set_index("apple")
    fig = plt.figure(figsize=(W2, 2.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.42)
    ax0 = fig.add_subplot(gs[0])
    img = "69-2"
    x0, y0, s = 1646, 806, 548
    im = np.asarray(Image.open(CW / f"full/{img}.jpg").convert("RGB"))[y0 : y0 + s, x0 : x0 + s]
    ax0.imshow(im, extent=(0, s, s, 0), interpolation="lanczos")
    O_ = pd.read_csv(CW / f"results/{img}/objects.csv")
    C_ = pd.read_csv(CW / f"results/{img}/object_classes_main.csv")
    O_ = O_.merge(C_[["id", "class"]], on="id", suffixes=("_p", ""))
    inn = O_[(O_.x >= x0) & (O_.x < x0 + s) & (O_.y >= y0) & (O_.y < y0 + s)]
    for cl, mk in [("intact", "o"), ("broken", "^")]:
        d = inn[inn["class"].str.lower() == cl]
        ax0.plot(
            d.x - x0,
            d.y - y0,
            mk,
            mfc="none",
            mec="white" if cl == "intact" else "#ffd84d",
            mew=0.6,
            ms=4.2,
            ls="none",
        )
    um = 0.37
    bar = 50 / um
    ax0.plot([s - 30 - bar, s - 30], [s - 28, s - 28], color="white", lw=1.6)
    ax0.text(
        s - 30 - bar / 2,
        s - 40,
        r"50~$\mu$m",
        color="white",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlim(0, s)
    ax0.set_ylim(s, 0)
    for sp in ax0.spines.values():
        sp.set_linewidth(0.5)
    ax = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
    w = I.pivot(index="apple", columns="field", values="cell_count")
    ax[0].plot(w[1], w[2], "o", mfc="none", mec=K, mew=0.5, ms=3)
    lim = [100, 320]
    ax[0].plot(lim, lim, "--", color=G2, lw=0.6, zorder=0)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    r = stats.spearmanr(w[1], w[2])[0]
    ax[0].text(0.05, 0.95, rf"$\rho = {r:.2f}$", transform=ax[0].transAxes, va="top")
    ax[0].set_xlabel("Count, field 1")
    ax[0].set_ylabel("Count, field 2")
    ax[1].plot(I.check_final_4sq, I.check_checker_4sq, "o", mfc="none", mec=K, mew=0.5, ms=3)
    lim = [15, 95]
    ax[1].plot(lim, lim, "--", color=G2, lw=0.6, zorder=0)
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ratio = (I.check_checker_4sq / I.check_final_4sq).median()
    ax[1].text(
        0.05,
        0.95,
        rf"$\rho = {stats.spearmanr(I.check_final_4sq, I.check_checker_4sq)[0]:.2f}$"
        + "\n"
        + rf"median ratio ${ratio:.2f}$",
        transform=ax[1].transAxes,
        va="top",
        ha="left",
        linespacing=1.3,
    )
    ax[1].set_xlabel("Final count (4 squares)")
    ax[1].set_ylabel("Blind recount (4 squares)")
    ax[2].plot(A.survey_mealy, A.intact_share, "o", mfc="none", mec=K, mew=0.5, ms=3)
    r = stats.spearmanr(A.survey_mealy, A.intact_share)[0]
    ax[2].text(
        0.95,
        0.95,
        rf"$\rho = {r:.2f}$" + "\n" + r"$n = 15$",
        transform=ax[2].transAxes,
        va="top",
        ha="right",
        linespacing=1.3,
    )
    ax[2].set_xlabel("Panel mealiness (centred)")
    ax[2].set_ylabel("Intact share")
    ax[2].set_ylim(0.18, 0.78)
    plab(ax0, "a", x=0.0)
    [plab(a, s, x=-0.3) for a, s in zip(ax, "bcd")]
    save(fig, "fig8_cells")


def fig9():
    R = pd.read_csv(GA / "correlation_screen.csv")
    keep = [
        ("dry_matter", "brix", "Dry matter -- SSC"),
        ("nir_645nm", "brix", "NIR 645 nm -- SSC"),
        ("us_msc_weighted", "brix", "US coherence -- SSC"),
        ("nir_730nm", "puncture2", "NIR 730 nm -- firmness (2nd)"),
        ("us_h3_dB_linear_bins", "hardness", "US harmonic level -- firmness"),
        ("nir_680nm", "mass", "NIR 680 nm -- fruit mass"),
        ("dry_matter", "sens_mealy", "Dry matter -- mealiness"),
        ("brix", "sens_mealy", "SSC -- mealiness"),
        ("puncture2", "sens_mealy", "Firmness (2nd) -- mealiness"),
        ("nir_560nm", "sens_mealy", "NIR 560 nm -- mealiness"),
        ("nir_760nm", "sens_mealy", "NIR 760 nm -- mealiness"),
        ("us_shape8_7", "sens_mealy", "US spectral shape 7 -- mealiness"),
        ("intact_share", "sens_mealy", "Intact share -- mealiness"),
        ("cell_count", "sens_mealy", "Cell count -- mealiness"),
    ]
    rows = []
    for x_, y_, lab in keep:
        s = R[((R.x == x_) & (R.y == y_)) | ((R.x == y_) & (R.y == x_))].iloc[0]
        rows.append((lab, s))
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 3.1))
    yy = np.arange(len(rows))[::-1]
    for y, (lab, s) in zip(yy, rows):
        ax.plot([s.ci_lo, s.ci_hi], [y, y], color=K if s.n >= 60 else G1, lw=0.8)
        ax.plot(
            [s.r],
            [y],
            "s" if s.n >= 60 else "o",
            color=K if s.n >= 60 else "white",
            mec=K,
            mew=0.6,
            ms=3.2,
        )
        ax.plot([s.r_within_lot], [y], "x", color=ACC2, ms=3.2, mew=0.6)
        ax.text(1.04, y, rf"${s.r:+.2f}$ ({int(s.n)})", va="center", fontsize=6.3)
    for v in (-0.6, 0.6):
        ax.axvline(v, color=G2, lw=0.5, ls=":")
    ax.axvline(0, color=K, lw=0.4)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"Pearson $r$ with 95\% CI")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", color=K, ls="", ms=3.2, label="$n = 69$--$70$"),
            Line2D([], [], marker="o", mfc="white", mec=K, ls="", ms=3.2, label="$n = 15$"),
            Line2D([], [], marker="x", color=ACC2, ls="", ms=3.2, label="within lot"),
        ],
        loc="lower left",
        fontsize=6.2,
        ncol=3,
        bbox_to_anchor=(-0.02, 1.0),
        columnspacing=0.8,
        handletextpad=0.2,
    )
    save(fig, "fig9_screen")


def fig10():
    D = pd.read_csv(GA / "mealiness_direct_models.csv")
    M = pd.read_csv(GA / "model_correlations.csv")
    S = T.loc[SENS]
    y = S.sens_mealy.to_numpy(float)
    fig = plt.figure(figsize=(W2, 1.95))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.95, 0.95], wspace=0.5)
    ax = [fig.add_subplot(gs[i]) for i in (0, 1, 2, 4)]
    for a, c, lab in [
        (ax[0], "dry_matter", r"Dry matter (\%)"),
        (ax[1], "nir_560nm", "NIR 560 nm (SNV)"),
    ]:
        x = S[c].to_numpy(float)
        a.plot(x, y, "o", mfc="none", mec=K, mew=0.5, ms=3)
        fit_line(a, x, y, color=G1, ls="--")
        r, n = rtxt(x, y)
        a.text(
            0.95,
            0.95 if r < 0 else 0.05,
            rf"$r = {r:.2f}$",
            transform=a.transAxes,
            va="top" if r < 0 else "bottom",
            ha="right",
        )
        a.set_xlabel(lab)
    ax[0].set_ylabel("Panel mealiness ($z$)")

    X = S[[c for c in T.columns if c.startswith("nir_") and c.endswith("nm")]].to_numpy(float)
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        j = int(np.argmax([abs(np.corrcoef(X[tr, k], y[tr])[0, 1]) for k in range(X.shape[1])]))
        p[i] = np.polyval(np.polyfit(X[tr, j], y[tr], 1), X[i, j])
    ax[2].plot(p, y, "o", mfc="none", mec=K, mew=0.5, ms=3)
    lim = [-1.0, 1.2]
    ax[2].plot(lim, lim, "--", color=G2, lw=0.6, zorder=0)
    ax[2].set_xlabel("LOO prediction")
    ax[2].text(
        0.95,
        0.05,
        rf"$r = {np.corrcoef(p, y)[0, 1]:.2f}$"
        + "\n"
        + rf"$p_{{\mathrm{{perm}}}} = {D.p_perm.iloc[0]:.3f}$",
        transform=ax[2].transAxes,
        va="bottom",
        ha="right",
        linespacing=1.3,
    )

    def m(o, i, c):
        s = M[(M.output == o) & (M.inputs == i) & M.calibration.str.contains(c, regex=False)]
        return abs(float(s.cv_r.iloc[0]))

    bars = [
        (
            "Firmness (measured)",
            m("mealiness (panel, 15)", "measured puncture2", "reference"),
            "meas",
        ),
        (
            "Dry matter (measured)",
            m("mealiness (panel, 15)", "measured dry_matter", "regression"),
            "meas",
        ),
        (
            "Dry matter + firmness",
            m("mealiness (panel, 15)", "measured dry_matter+puncture2", "regression"),
            "meas",
        ),
        ("NIR, firmness proxy", m("mealiness (panel, 15)", "NIR", "puncture2 model"), "inst"),
        ("NIR+US, firmness proxy", m("mealiness (panel, 15)", "NIR+US", "puncture2 model"), "inst"),
        (
            "NIR, dry-matter proxy",
            m("mealiness (panel, 15)", "NIR", "proxy dry_matter (Carroll"),
            "inst",
        ),
        ("NIR 560 nm, in-fold", float(D.cv_r.iloc[0]), "inst"),
        ("US feature, in-fold", abs(float(D.cv_r.iloc[2])), "inst"),
    ]
    yy = np.arange(len(bars))[::-1]
    ax[3].barh(
        yy,
        [b[1] for b in bars],
        color=["0.75" if b[2] == "meas" else "0.25" for b in bars],
        edgecolor=K,
        lw=0.4,
        height=0.65,
    )
    ax[3].set_yticks(yy)
    ax[3].set_yticklabels([b[0] for b in bars], fontsize=6.0)
    ax[3].tick_params(axis="y", length=0)
    ax[3].axvline(0.6, color=G1, lw=0.5, ls=":")
    ax[3].set_xlim(0, 0.8)
    ax[3].set_xlabel(r"$|r|$, held out ($n = 15$)")
    for a, s in zip(ax, "abcd"):
        plab(a, s, x=-0.3 if s != "d" else -1.15)
    save(fig, "fig10_mealiness")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10"]
    info = {}
    for f in todo:
        out = globals()[f]()
        if out:
            info[f] = out
    if info:
        json.dump(info, open(FIG / "fig_info.json", "w"), indent=1, default=float)
