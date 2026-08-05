"""
features/universal.py
=====================
Unified Generalized Feature Engine.
Uses fractional windowing and normalized time-axes to ensure
features are consistent across any time-series length (T).
"""

import numpy as np
from scipy import stats

from .utils import safex


def extract_universal_features(row, areacols):
    """
    Unified Generalized Feature Engine.
    Uses fractional windowing and normalized time-axes to ensure
    features are consistent across any time-series length (T).
    """
    a = row[areacols].values.astype(float)
    n = len(a)
    if n == 0: return {k: 0 for k in ["mean_area", "std_area", "overall_slope"]}

    f = {}
    # Time axis normalized to [0, 1] for all slope calculations
    t_norm = np.linspace(0, 1, n)

    # 1. Basic Stats (Scale Agnostic)
    f["mean_area"] = np.mean(a)
    f["std_area"] = np.std(a)
    f["max_area"] = np.max(a)
    f["min_area"] = np.min(a)
    f["cv_area"] = f["std_area"] / (f["mean_area"] + 1e-9)

    # 2. Water Presence (Fractional)
    f["zero_fraction"] = np.sum(a == 0) / n
    f["nonzero_fraction"] = np.sum(a > 0) / n

    # 3. Fractional Window Means (Quarters)
    for i in range(4):
        start, end = int(i*0.25*n), int((i+1)*0.25*n)
        segment = a[start:end]
        f[f"q{i+1}_mean"] = np.mean(segment) if len(segment) else 0

    # 4. Normalized Slopes (using t_norm)
    nz_mask = a > 0
    if np.sum(nz_mask) > 1:
        try:
            f["normalized_overall_slope"] = np.polyfit(t_norm[nz_mask], a[nz_mask], 1)[0]
        except:
            f["normalized_overall_slope"] = 0
    else:
        f["normalized_overall_slope"] = 0

    # 5. Rejuvenation (Fractional Location)
    if f["nonzero_fraction"] > 0:
        minidx = np.argmin(a)
        f["min_location_pct"] = minidx / n
        after = a[minidx+1:]
        t_after = t_norm[minidx+1:]
        if len(after) > 2:
            try:
                f["normalized_recovery_slope"] = np.polyfit(t_after, after, 1)[0]
            except:
                f["normalized_recovery_slope"] = 0
        else:
            f["normalized_recovery_slope"] = 0
    else:
        f["min_location_pct"] = 0
        f["normalized_recovery_slope"] = 0

    # 6. Zero Patterns (Fractional)
    consec = 0; mx = 0
    for v in a:
        if v == 0:
            consec += 1; mx = max(mx, consec)
        else:
            consec = 0
    f["max_consecutive_zeros_pct"] = mx / n
    f["zeros_at_end_pct"] = np.sum(a[-max(1, int(n*0.2)):] == 0) / max(1, int(n*0.2))

    # 7. Variability & Stability
    if n > 1:
        f["normalized_area_changes"] = np.sum(np.abs(np.diff(a))) / n
    else:
        f["normalized_area_changes"] = 0
    f["stability_index"] = 1 - (f["std_area"] / (f["mean_area"] + 1e-9))

    # 8. Distribution stats
    f["skewness"] = safex(stats.skew(a)) if n > 2 else 0
    f["kurtosis"] = safex(stats.kurtosis(a)) if n > 3 else 0

    # Sanitize
    for k in list(f.keys()):
        f[k] = safex(f[k])
    return f
