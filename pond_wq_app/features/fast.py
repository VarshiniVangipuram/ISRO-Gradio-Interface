"""
features/fast.py
================
Comprehensive 76-feature extractor for time-series data.
Used by the Single XGBoost training pipeline.
"""

import numpy as np
from scipy import stats

from .utils import safex


def extractfeaturesfast(row, areacols):
    """Extract 76 features from a time-series row."""
    a = row[areacols].values.astype(float)
    n = len(a)
    f = {}

    # Basic statistics 7
    f["mean_area"] = np.mean(a)
    f["median_area"] = np.median(a)
    f["std_area"] = np.std(a)
    f["max_area"] = np.max(a)
    f["min_area"] = np.min(a)
    f["range_area"] = f["max_area"] - f["min_area"]
    f["cv_area"] = f["std_area"] / f["mean_area"] if f["mean_area"] != 0 else 0

    # Water presence 4
    f["zero_count"] = np.sum(a == 0)
    f["zero_percentage"] = f["zero_count"] / n * 100
    f["nonzero_count"] = np.sum(a > 0)
    f["nonzero_percentage"] = f["nonzero_count"] / n * 100

    # Temporal quarters 4
    q1, q2, q3 = n//4, n//2, (3*n)//4
    f["quarter1_mean"] = np.mean(a[:q1]) if q1 else 0
    f["quarter2_mean"] = np.mean(a[q1:q2]) if q2 > q1 else 0
    f["quarter3_mean"] = np.mean(a[q2:q3]) if q3 > q2 else 0
    f["quarter4_mean"] = np.mean(a[q3:]) if n > q3 else 0

    # Trend 1 (nonzero only)
    f["overall_trend_slope"] = 0
    nz = a[a > 0]
    if np.sum(a > 0) > 1:
        try:
            f["overall_trend_slope"] = np.polyfit(np.arange(len(nz)), nz, 1)[0]
        except:
            f["overall_trend_slope"] = 0

    # Rejuvenation 5
    if f["nonzero_count"] > 0:
        minidx = np.argmin(a)
        before = a[:minidx] if minidx > 0 else np.array([0.0])
        after = a[minidx+1:] if minidx < n-1 else np.array([0.0])
        f["min_location_ratio"] = minidx / n
        f["before_min_mean"] = np.mean(before) if len(before) else 0
        f["after_min_mean"] = np.mean(after) if len(after) else 0
        f["recovery_ratio"] = (f["after_min_mean"] / (f["before_min_mean"] + 1e-9)) if f["after_min_mean"] > 0 else 0
        f["recovery_slope"] = np.polyfit(np.arange(len(after)), after, 1)[0] if len(after) >= 5 else 0
    else:
        f.update({"min_location_ratio":0,"before_min_mean":0,"after_min_mean":0,"recovery_ratio":0,"recovery_slope":0})

    # V-shape 4
    half = n//2
    mid = a[n//3:2*n//3] if n >= 3 else np.array([0.0])
    fh = np.mean(a[:half]) if half else 0
    sh = np.mean(a[half:]) if n else 0
    mm = np.mean(mid) if len(mid) else 0
    f["first_half_mean"] = fh
    f["second_half_mean"] = sh
    f["middle_section_mean"] = mm
    f["vshape_indicator"] = (fh + sh) / (2*mm + 1e-9) if mm != 0 else 0

    # Zero patterns 2
    consec = 0; mx = 0
    for v in a:
        if v == 0:
            consec += 1; mx = max(mx, consec)
        else:
            consec = 0
    f["max_consecutive_zeros"] = mx
    f["zeros_at_end"] = np.sum(a[-max(1, int(n*0.2)):] == 0)

    # Variability 2
    if n > 1:
        f["area_changes"] = np.sum(np.abs(np.diff(a)))
        f["mean_change"] = np.mean(np.abs(np.diff(a)))
    else:
        f["area_changes"] = 0
        f["mean_change"] = 0

    # Stability 1
    f["stability_index"] = 1 - (f["std_area"] / (f["mean_area"] + 1e-9)) if f["mean_area"] != 0 else 0

    # Peaks 2
    if n > 5:
        peaks = [i for i in range(1, n-1) if a[i] > a[i-1] and a[i] > a[i+1] and a[i] > 0]
        f["num_peaks"] = len(peaks)
        f["peak_distance"] = (peaks[-1] - peaks[0]) if len(peaks) >= 2 else 0
    else:
        f["num_peaks"] = 0
        f["peak_distance"] = 0

    # Advanced distribution 5
    f["skewness"] = safex(stats.skew(a)) if n > 2 else 0
    f["kurtosis"] = safex(stats.kurtosis(a)) if n > 3 else 0
    f["percentile25"] = np.percentile(a, 25)
    f["percentile75"] = np.percentile(a, 75)
    f["iqr"] = f["percentile75"] - f["percentile25"]

    # Start/end 2
    if n > 10:
        f10 = np.mean(a[:10])
        l10 = np.mean(a[-10:])
        f["start_end_ratio"] = l10 / (f10 + 1e-9) if l10 > 0 else 0
        f["start_end_diff"] = l10 - f10
    else:
        f["start_end_ratio"] = 0
        f["start_end_diff"] = 0

    # Returns 2
    if n > 1:
        r = np.diff(a) / (a[:-1] + 1e-9)
        f["volatility"] = np.std(r)
        f["mean_return"] = np.mean(r)
    else:
        f["volatility"] = 0
        f["mean_return"] = 0

    # Thirds 4
    d1 = a[:n//3]
    d2 = a[n//3:2*n//3]
    d3 = a[2*n//3:]
    f["decade1_mean"] = np.mean(d1) if len(d1) else 0
    f["decade2_mean"] = np.mean(d2) if len(d2) else 0
    f["decade3_mean"] = np.mean(d3) if len(d3) else 0
    f["decade_trend"] = (f["decade3_mean"] - f["decade1_mean"]) / (f["decade1_mean"] + 1e-9) if f["decade1_mean"] > 0 else 0

    # Zero span 3
    zp = np.where(a == 0)[0]
    if len(zp):
        f["first_zero_position"] = zp[0] / n
        f["last_zero_position"] = zp[-1] / n
        f["zero_span"] = (zp[-1] - zp[0]) / n if len(zp) > 1 else 0
    else:
        f["first_zero_position"] = 1
        f["last_zero_position"] = 1
        f["zero_span"] = 0

    # Discriminative (subset)
    early = a[:max(1, n//4)]
    ft = a[:max(1, n//3)]
    late = a[-max(1, n//4):]
    f["early_period_zeros"] = np.sum(early == 0)
    f["early_period_mean"] = np.mean(early)
    f["early_period_max"] = np.max(early) if len(early) else 0
    f["had_early_water"] = 1 if np.sum(ft > 0) > max(1, int(0.3*len(ft))) else 0
    f["early_water_ratio"] = np.sum(ft > 0) / (len(ft) + 1e-9)
    f["zero_to_nonzero_transitions"] = sum(1 for i in range(1, n) if a[i-1] == 0 and a[i] > 0)
    f["starts_from_zero"] = 1 if (n and a[0] == 0) else 0
    f["late_period_mean"] = np.mean(late)
    f["late_period_growing"] = 1 if (np.mean(late) > np.mean(a[:max(1, n//2)])) else 0

    # More discriminators (a few)
    f40 = a[:max(1, int(n*0.4))]
    f["first40pct_water_presence"] = np.sum(f40 > 0) / (len(f40) + 1e-9)
    f["max_area_location"] = np.argmax(a) / n if n else 0.5

    # sanitize all
    for k in list(f.keys()):
        f[k] = safex(f[k])
    return f
