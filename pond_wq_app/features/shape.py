"""
features/shape.py
=================
Shape-only feature extractors (13-feature and 8-feature variants).
Both apply per-sample min-max normalization before extraction.
"""

import numpy as np


def extract_shape_only_features(a):
    """Exact 13-feature extractor used to train the Universal Shape XGBoost model."""
    a = np.array(a, dtype=float)
    # Per-sample min-max normalization
    if np.max(a) > np.min(a):
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        a = np.zeros_like(a)

    n = len(a)
    f = {}

    # Zero structure
    f["zero_percentage"] = np.sum(a == 0) / n
    f["non_zero_percentage"] = np.sum(a > 0) / n

    # Transitions
    f["zero_to_nonzero"] = sum(1 for i in range(1, n) if a[i-1] == 0 and a[i] > 0)
    f["nonzero_to_zero"] = sum(1 for i in range(1, n) if a[i-1] > 0 and a[i] == 0)

    # Consecutive zeros (fractional)
    consec = mx = 0
    for v in a:
        consec = consec + 1 if v == 0 else 0
        mx = max(mx, consec)
    f["max_consecutive_zeros"] = mx / n

    # Shape trends
    if n > 1:
        f["volatility"] = np.std(np.diff(a))
        f["mean_change"] = np.mean(np.abs(np.diff(a)))
    else:
        f["volatility"] = 0
        f["mean_change"] = 0

    # Peaks
    if n > 5:
        peaks = [i for i in range(1, n-1) if a[i] > a[i-1] and a[i] > a[i+1]]
        f["num_peaks"] = len(peaks)
    else:
        f["num_peaks"] = 0

    # Early vs Late ratio (fractional, relative only)
    early = a[:max(1, n//3)]
    late  = a[-max(1, n//3):]
    f["early_water_ratio"] = np.sum(early > 0) / len(early)
    f["late_water_ratio"]  = np.sum(late > 0) / len(late)
    f["late_to_early_ratio"] = (
        f["late_water_ratio"] / f["early_water_ratio"]
        if f["early_water_ratio"] > 0 else 0
    )

    # Shape flags
    if n > 10:
        seg = max(1, n // 5)
        segs = [np.mean(a[i*seg:(i+1)*seg]) for i in range(5)]
        f["is_monotonic_increasing"] = 1 if all(segs[i] <= segs[i+1] for i in range(4)) else 0
        f["has_u_shape"] = 1 if segs[2] < (segs[0] + segs[4]) / 2 * 0.7 else 0
    else:
        f["is_monotonic_increasing"] = 0
        f["has_u_shape"] = 0

    return f


def extract_shape_features_8(a):
    """8-feature shape extractor with per-sample min-max normalization."""
    a = np.array(a, dtype=float)
    if np.max(a) > np.min(a):
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        a = np.zeros_like(a)

    n = len(a)
    f = {}
    f["zero_pct"] = np.sum(a==0)/n
    f["nonzero_pct"] = np.sum(a>0)/n
    f["transitions"] = sum(1 for i in range(1,n) if (a[i]==0)!=(a[i-1]==0))

    consec = mx = 0
    for v in a:
        consec = consec+1 if v==0 else 0
        mx = max(mx, consec)
    f["max_consec_zero"] = mx/n

    if n>1:
        f["volatility"] = np.std(np.diff(a))
        f["mean_change"] = np.mean(np.abs(np.diff(a)))
    else:
        f["volatility"]=0
        f["mean_change"]=0

    early = a[:max(1,n//3)]
    late  = a[-max(1,n//3):]
    f["early_water"] = np.sum(early>0)/len(early)
    f["late_water"]  = np.sum(late>0)/len(late)
    return f
