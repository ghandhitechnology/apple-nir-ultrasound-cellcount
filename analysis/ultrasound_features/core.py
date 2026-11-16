# Shared ultrasound signal processing. It turns each recorded 40 kHz sweep into a transfer response
# per frequency bin and summary features such as level, group delay and saturation, for files with
# any number of runs. The transform is the frozen one from the earlier 40-apple analysis.

import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, hilbert, periodogram, sosfiltfilt, welch

FS = 1 / 3.1e-6
N = 6453
t = np.arange(N) / FS
SOS1 = butter(4, [30e3, 50e3], "band", fs=FS, output="sos")
SOS3 = butter(4, [105e3, 135e3], "band", fs=FS, output="sos")
EDGE = 150
GUARD = int(round(1.0e-3 * FS))
EDGES = np.arange(38.4e3, 41.6e3 + 1, 100.0)
FC = (EDGES[:-1] + EDGES[1:]) / 2
NB = len(FC)
DF = 100.0
SAT, WEAK = 0.19, 0.01


def fit_sweep(b):
    ph = np.unwrap(np.angle(b))
    f = np.gradient(ph) * FS / (2 * np.pi)
    valid = np.zeros(N, bool)
    valid[EDGE : N - EDGE] = True
    fs_ = pd.Series(f).rolling(15, center=True, min_periods=1).median().to_numpy()
    L = 40
    d = np.zeros(N)
    d[: N - L] = fs_[L:] - fs_[: N - L]
    bad = (d < -2500) & valid
    resets = []
    for reg in np.split(np.flatnonzero(bad), np.flatnonzero(np.diff(np.flatnonzero(bad)) > 1) + 1):
        if len(reg):
            resets.append(int(reg[np.argmin(d[reg])] + L // 2))
    bounds = [EDGE] + resets + [N - EDGE]
    ffit = np.full(N, np.nan)
    seg = np.full(N, -1)
    slopes = []
    for s, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        lo2 = lo + (GUARD if s > 0 else 0)
        if hi - lo2 < 200:
            continue
        idx = np.arange(lo2 + 20, hi - 20)
        p = np.polyfit(t[idx], f[idx], 1)
        slopes.append(p[0])
        ffit[lo2 : hi - 20] = np.polyval(p, t[lo2 : hi - 20])
        seg[lo2 : hi - 20] = s
    resid = np.nanstd(f[seg >= 0] - ffit[seg >= 0])
    return ffit, seg, resets, slopes, resid


def gd(Hv):
    c = Hv[1:] * np.conj(Hv[:-1])
    return -np.angle(np.nansum(c)) / (2 * np.pi * DF) * 1e3


def process_run(Araw, Braw, Traw):
    a = hilbert(sosfiltfilt(SOS1, Araw - Araw.mean()))
    b = hilbert(sosfiltfilt(SOS1, Braw - Braw.mean()))
    a3 = np.abs(hilbert(sosfiltfilt(SOS3, Araw - Araw.mean())))
    ffit, seg, resets, slopes, resid = fit_sweep(b)
    ok = seg >= 0
    bi = np.digitize(ffit, EDGES) - 1
    ok &= (bi >= 0) & (bi < NB)
    ab = a * np.conj(b)
    bb = np.abs(b) ** 2
    ea = np.abs(a)
    H = np.full(NB, np.nan, complex)
    Hp = np.full((3, NB), np.nan, complex)
    Amed = np.full(NB, np.nan)
    Amax = Amed.copy()
    H3 = Amed.copy()
    for q in range(NB):
        m = ok & (bi == q)
        if m.sum() < 20:
            continue
        H[q] = ab[m].sum() / bb[m].sum()
        Amax[q] = np.percentile(ea[m], 95)
        Amed[q] = np.median(ea[m])
        H3[q] = np.sum(a3[m] ** 2) / np.sum(ea[m] ** 2)
        for sgi, s in enumerate(np.unique(seg[m])[:3]):
            mm = m & (seg == s)
            if mm.sum() >= 20:
                Hp[sgi, q] = ab[mm].sum() / bb[mm].sum()
    d = 20 * np.log10(np.abs(H))
    sat = Amed > SAT
    weak = Amed < WEAK
    lo, hi = FC < 40e3, FC >= 40e3
    x = Araw - Araw.mean()
    p8 = 10 * np.log10(np.mean(np.abs(H.reshape(8, 4)) ** 2, 1))

    fsl = 1 / np.median(np.diff(Traw[:100]))
    f, pa = periodogram(Araw - Araw.mean(), fsl, window="hann")
    _, pb = periodogram(Braw - Braw.mean(), fsl, window="hann")
    band = (f >= 32000) & (f <= 48000)
    dfl = f[1] - f[0]

    fc_, C = coherence(Braw, Araw, fs=FS, nperseg=2048)
    _, pbw = welch(Braw, fs=FS, nperseg=2048)
    mc = (fc_ >= 38.4e3) & (fc_ <= 41.6e3)
    feat = dict(
        n_resets=len(resets),
        first_reset_ms=resets[0] / FS * 1e3 if resets else np.nan,
        period_ms=np.median(np.diff(resets)) / FS * 1e3 if len(resets) > 1 else np.nan,
        sweep_rate_Hz_per_ms=np.median(slopes) / 1e3 if slopes else np.nan,
        fit_resid_Hz=resid,
        B_env_median=np.median(np.abs(b)[EDGE:-EDGE]),
        A_env_median=np.median(ea[EDGE:-EDGE]),
        bins_covered=int(np.isfinite(H).sum()),
        level_med_dB=np.median(d),
        level_q25_dB=np.percentile(d, 25),
        level_q75_dB=np.percentile(d, 75),
        level_rms_dB=10 * np.log10(np.mean(np.abs(H) ** 2)),
        sat_frac=sat.mean(),
        weak_frac=weak.mean(),
        roughness_dB=np.subtract(*np.percentile(d, [75, 25])),
        tilt_dB=np.median(d[hi]) - np.median(d[lo]),
        gd_ms=gd(H),
        gd_lo_ms=gd(H[lo]),
        gd_hi_ms=gd(H[hi]),
        pass_rep_dB=(
            np.nanmedian(np.abs(20 * np.log10(np.abs(Hp[1] / Hp[0]))))
            if np.isfinite(Hp[1]).any()
            else np.nan
        ),
        h3_dB_linear_bins=(
            10 * np.log10(np.nanmedian(H3[~sat & ~weak])) if (~sat & ~weak).any() else np.nan
        ),
        A_rms_V=x.std(),
        A_crest=np.abs(x).max() / x.std(),
        B_rms_V=Braw.std(),
        leg_H=np.sqrt(pa[band].sum() * dfl) / np.sqrt(pb[band].sum() * dfl),
        leg_centroid=np.sum(pa[band] * f[band]) / pa[band].sum(),
        msc_weighted=np.sum(C[mc] * pbw[mc]) / np.sum(pbw[mc]),
        **{f"shape8_{j}": p8[j] for j in range(8)},
    )
    return dict(H=H, Hp=Hp, Amed=Amed, Amax=Amax, H3=H3), feat


def icc_unbalanced(df, col, group="apple"):
    df = df.dropna(subset=[col])
    g = df.groupby(group)[col]
    n = g.size()
    a = len(n)
    Ntot = n.sum()
    grand = df[col].mean()
    ssb = (n * (g.mean() - grand) ** 2).sum()
    ssw = ((df[col] - g.transform("mean")) ** 2).sum()
    msb = ssb / (a - 1)
    msw = ssw / (Ntot - a)
    k0 = (Ntot - (n**2).sum() / Ntot) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return icc, np.sqrt(msw), k0
