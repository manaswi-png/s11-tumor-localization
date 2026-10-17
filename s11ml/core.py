"""
Shared code for the S11 tumor-localization project.

Everything that must be IDENTICAL between training and prediction lives here:
frequency axis, frequency bands, feature extraction, label mapping, rotation
augmentation and the physics (no-training) baseline. The training script and the
Streamlit app both import this file, so a prediction in the app goes through
exactly the same steps as the data the model was trained on.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Measurement geometry (from the dataset / original notebook)
# ----------------------------------------------------------------------------
N_FREQ = 1001                      # frequency points per scan
N_ANT = 72                         # antenna positions per scan (360 deg / 72 = 5 deg)
FREQ_GHZ = np.linspace(1.0, 8.0, N_FREQ)   # 1-8 GHz sweep (same as notebook)
ANT_STEP_DEG = 360.0 / N_ANT
QUARTER_TURN = N_ANT // 4          # 18 antenna positions = 90 degrees

# Frequency bands used by the original notebook: [low, high) in GHz
BANDS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

# ----------------------------------------------------------------------------
# Labels: 4 tumor positions (z is always -6.5 cm). Mapping kept from the notebook.
# ----------------------------------------------------------------------------
LOCATION_MAP = {(-2.25, 2.25): 0, (2.25, 2.25): 1, (-2.25, -2.25): 2, (2.25, -2.25): 3}
CLASS_COORDS = {v: k for k, v in LOCATION_MAP.items()}          # class -> (x, y) in cm
CLASS_NAMES = {
    0: "Upper-left (x=-2.25, y=+2.25)",
    1: "Upper-right (x=+2.25, y=+2.25)",
    2: "Lower-left (x=-2.25, y=-2.25)",
    3: "Lower-right (x=+2.25, y=-2.25)",
}
CLASS_SHORT = {0: "(-,+)", 1: "(+,+)", 2: "(-,-)", 3: "(+,-)"}

# Rolling the antenna axis by +18 positions moves the response peak by +90 deg of
# antenna index. Measured in this data: peak index ~0 for (-,-), ~18 for (-,+),
# ~36 for (+,+), ~54 for (+,-). So a +18 roll maps (x, y) -> (y, -x).
def rotate_class(c: int, quarter_turns: int) -> int:
    x, y = CLASS_COORDS[int(c)]
    for _ in range(quarter_turns % 4):
        x, y = y, -x
    return LOCATION_MAP[(x, y)]


def feature_names() -> list[str]:
    return [f"band{lo}-{hi}GHz_ant{a:02d}" for lo, hi in BANDS for a in range(N_ANT)]


# ----------------------------------------------------------------------------
# Loading and pairing
# ----------------------------------------------------------------------------
def load_metadata(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        md = pickle.load(f)
    df = pd.DataFrame(md)
    df.columns = [str(c) for c in df.columns]          # keys are np.str_ in the pickle
    return df


def load_s11(path: str | Path) -> np.ndarray:
    """Load the big (n_scans, 1001, 72) complex array. Call this ONCE."""
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[1:] != (N_FREQ, N_ANT):
        raise ValueError(f"Unexpected S11 shape {arr.shape}")
    return arr


# ----------------------------------------------------------------------------
# Feature extraction (the notebook's method, unchanged)
# ----------------------------------------------------------------------------
def delta_s11(tumor_scan: np.ndarray, reference_scan: np.ndarray) -> np.ndarray:
    """Tumor-induced change: dS11 = S11(tumor) - S11(healthy reference)."""
    return np.asarray(tumor_scan) - np.asarray(reference_scan)


def band_features(delta: np.ndarray) -> np.ndarray:
    """|dS11| averaged inside each frequency band, for each antenna.

    delta: (1001, 72) or (n, 1001, 72) complex  ->  (504,) or (n, 504) float
    Order: band-major (all 72 antennas of band 1, then band 2, ...).
    """
    d = np.asarray(delta)
    single = d.ndim == 2
    if single:
        d = d[None]
    if d.shape[1:] != (N_FREQ, N_ANT):
        raise ValueError(f"Expected (..., {N_FREQ}, {N_ANT}), got {d.shape}")
    mag = np.abs(d)
    feats = []
    for lo, hi in BANDS:
        mask = (FREQ_GHZ >= lo) & (FREQ_GHZ < hi)
        feats.append(mag[:, mask, :].mean(axis=1))      # (n, 72)
    out = np.concatenate(feats, axis=1)
    return out[0] if single else out


def roll_features(X: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate band features by k*90 deg: circular shift of the antenna axis inside each band.
    Equivalent to rolling dS11 along the antenna axis before band_features()."""
    X = np.asarray(X)
    n = X.shape[0]
    Xr = X.reshape(n, len(BANDS), N_ANT)
    return np.roll(Xr, QUARTER_TURN * quarter_turns, axis=2).reshape(n, -1)


def augment_rotations(X: np.ndarray, y: np.ndarray, groups=None):
    """Physics-based augmentation: add the 90/180/270-degree rotated copies of each sample.
    Used on TRAINING data only."""
    Xs, ys, gs = [X], [y], [groups] if groups is not None else None
    for k in (1, 2, 3):
        Xs.append(roll_features(X, k))
        ys.append(np.array([rotate_class(c, k) for c in y]))
        if groups is not None:
            gs.append(groups)
    Xa, ya = np.vstack(Xs), np.concatenate(ys)
    return (Xa, ya, np.concatenate(gs)) if groups is not None else (Xa, ya)


# ----------------------------------------------------------------------------
# Physics baseline (no machine learning)
# ----------------------------------------------------------------------------
def angular_profile(X_feat: np.ndarray) -> np.ndarray:
    """Average |dS11| per antenna across all bands -> (n, 72)."""
    X_feat = np.atleast_2d(X_feat)
    return X_feat.reshape(X_feat.shape[0], len(BANDS), N_ANT).mean(axis=1)


def peak_antenna(X_feat: np.ndarray) -> np.ndarray:
    """Antenna index with the largest (lightly smoothed) tumor response."""
    prof = angular_profile(X_feat)
    k = np.array([0.25, 0.5, 0.25])
    sm = sum(w * np.roll(prof, s, axis=1) for w, s in zip(k, (-1, 0, 1)))
    return sm.argmax(axis=1)


# class -> which multiple of 18 its peak sits at (relative to an offset learned on training data)
CLASS_QUARTER = {2: 0, 0: 1, 1: 2, 3: 3}          # (-,-)=0, (-,+)=1, (+,+)=2, (+,-)=3
QUARTER_CLASS = {v: k for k, v in CLASS_QUARTER.items()}


def fit_physics_offset(X_feat: np.ndarray, y: np.ndarray) -> float:
    """Learn the single angular offset (in antenna positions) from training data:
    circular mean of (peak index - 18*quarter(class))."""
    pk = peak_antenna(X_feat)
    off = pk - QUARTER_TURN * np.array([CLASS_QUARTER[int(c)] for c in y])
    ang = 2 * np.pi * off / N_ANT
    return float((np.angle(np.mean(np.exp(1j * ang))) * N_ANT / (2 * np.pi)) % N_ANT)


def physics_predict(X_feat: np.ndarray, offset: float) -> np.ndarray:
    pk = peak_antenna(X_feat)
    q = np.round(((pk - offset) % N_ANT) / QUARTER_TURN).astype(int) % 4
    return np.array([QUARTER_CLASS[int(v)] for v in q])


# ----------------------------------------------------------------------------
# Localization error
# ----------------------------------------------------------------------------
def localization_error_cm(y_true, y_pred) -> np.ndarray:
    """Euclidean distance (cm) between the true and predicted tumor positions.
    0 = correct, 4.5 = neighbouring quadrant, 6.36 = diagonal quadrant."""
    t = np.array([CLASS_COORDS[int(c)] for c in y_true])
    p = np.array([CLASS_COORDS[int(c)] for c in y_pred])
    return np.linalg.norm(t - p, axis=1)
