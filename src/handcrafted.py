"""Hand-crafted production features (spectral balance + dynamics / limiter signatures).

All features are computed on the full 30 s mono waveform. Selected after the EDA in eda/:
spectral tilt (sub-bass vs. air) and brick-wall-limiter signatures were the most
decade-discriminative; onset/transient-shape and macro-dynamics features were not.

Reads audio files only; extract_handcrafted.py stores the result.
"""
import numpy as np
import soundfile as sf
from scipy.signal import welch, butter, sosfilt
from scipy.stats import kurtosis

BANDS = {"sub_20_60": (20, 60), "bass_60_120": (60, 120), "low_120_250": (120, 250),
         "lowmid_250_500": (250, 500), "mid_500_2k": (500, 2000), "himid_2k_4k": (2000, 4000),
         "high_4k_8k": (4000, 8000), "air_8k_12k": (8000, 12000)}


def _db(v):
    return 20 * np.log10(np.asarray(v) + 1e-9)


def load_mono(path):
    """Read an audio file as float32 mono (channels averaged). Returns (samples, sample rate)."""
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x, sr


def spectral_features(x, sr):
    """Long-term spectrum (Welch) features: 8 band energies in dB relative to the energy above 20 Hz,
    two low-end ratios, spectral centroid and 85 % / 99 % roll-off."""
    f, P = welch(x, fs=sr, nperseg=4096, noverlap=2048)
    tot = P[f >= 20].sum() + 1e-12
    d = {k: 10 * np.log10(P[(f >= lo) & (f < hi)].sum() / tot + 1e-12) for k, (lo, hi) in BANDS.items()}
    d["below100_vs_rest_dB"] = 10 * np.log10(P[(f >= 20) & (f < 100)].sum() / (P[f >= 100].sum() + 1e-12) + 1e-12)
    d["tilt_60_120_over_20_60_dB"] = d["bass_60_120"] - d["sub_20_60"]
    pf = P / P.sum()
    c = np.cumsum(pf)
    d["centroid_hz"] = float((f * pf).sum())
    d["rolloff85_hz"] = float(f[np.searchsorted(c, 0.85)])
    d["rolloff99_hz"] = float(f[np.searchsorted(c, 0.99)])
    return d


def dynamics_features(x, sr):
    """Level and dynamics features: RMS, crest factor, limiter / clipping signatures, 400 ms block
    statistics and band-limited crest factors."""
    pk = np.abs(x).max() + 1e-9
    rms = np.sqrt((x ** 2).mean()) + 1e-9
    d = {"rms_dBFS": _db(rms), "crest_dB": _db(pk) - _db(rms)}
    # brick-wall limiter / clipping signatures
    d["near_peak_ratio_pct"] = 100 * np.mean(np.abs(x) > pk * 10 ** (-0.5 / 20))
    d["amp_kurtosis"] = kurtosis(x)
    s = (np.abs(x) > pk * 0.98).astype(np.int8)
    e = np.diff(np.r_[0, s, 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    d["max_flat_run_samples"] = float((en - st).max()) if len(st) else 0.0
    # micro-dynamics on 400 ms blocks
    n = int(0.4 * sr)
    b = x[: len(x) // n * n].reshape(-1, n)
    brms, bpk = _db(np.sqrt((b ** 2).mean(1))), _db(np.abs(b).max(1))
    d["block_crest_median_dB"] = float(np.median(bpk - brms))
    top = np.argsort(brms)[-len(brms) // 5:]
    d["DR_top20_dB"] = float(np.mean(bpk[top] - brms[top]))
    d["LRA_proxy_dB"] = float(np.percentile(brms, 95) - np.percentile(brms, 10))
    d["dyn_std_400ms_dB"] = float(brms.std())
    # band-limited crest
    for name, (lo, hi) in {"crest_low_60_250": (60, 250), "crest_high_2k_8k": (2000, 8000)}.items():
        xb = sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)
        d[name] = float(_db(np.abs(xb).max()) - _db(np.sqrt((xb ** 2).mean())))
    return d


def extract(path):
    """All hand-crafted features of one file as a dict (spectral features first, then dynamics)."""
    x, sr = load_mono(path)
    d = spectral_features(x, sr)
    d.update(dynamics_features(x, sr))
    return d
