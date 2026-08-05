"""
features/discriminative.py
==========================
Group-gated discriminative feature extractor for the Three-Stage ensemble
and the associated feature-frame builder.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .utils import safex


def extract_discriminative_features(row, areacols, groups):
    """
    Extract features from a time-series row, gated by feature-group flags.

    Parameters
    ----------
    row : pandas Series
        A single row from the unified DataFrame.
    areacols : list of str
        Column names representing the time-series values.
    groups : dict
        Keys: basic, water, quarters, trend, rejuvenation, vshape,
              zero_patterns, variability, peaks, advanced, discriminative.
        Values are booleans controlling whether each family is included.
    """
    areas = row[areacols].values.astype(float)
    n = len(areas)
    f = {}

    if groups["basic"]:
        f["mean_area"] = np.mean(areas)
        f["median_area"] = np.median(areas)
        f["std_area"] = np.std(areas)
        f["max_area"] = np.max(areas)
        f["min_area"] = np.min(areas)
        f["range_area"] = f["max_area"] - f["min_area"]
        f["cv_area"] = f["std_area"] / f["mean_area"] if f["mean_area"] != 0 else 0

    if groups["water"]:
        f["zero_count"] = np.sum(areas == 0)
        f["zero_percentage"] = f["zero_count"] / n * 100
        f["nonzero_count"] = np.sum(areas > 0)
        f["nonzero_percentage"] = f["nonzero_count"] / n * 100

    if groups["quarters"]:
        q1, q2, q3 = n // 4, n // 2, (3*n)//4
        f["quarter1_mean"] = np.mean(areas[:q1]) if q1 else 0
        f["quarter2_mean"] = np.mean(areas[q1:q2]) if q2 > q1 else 0
        f["quarter3_mean"] = np.mean(areas[q2:q3]) if q3 > q2 else 0
        f["quarter4_mean"] = np.mean(areas[q3:]) if n > q3 else 0

    if groups["trend"]:
        if np.sum(areas > 0) > 1:
            t = np.arange(n)
            m = areas > 0
            try:
                f["overall_trend_slope"] = np.polyfit(t[m], areas[m], 1)[0]
            except:
                f["overall_trend_slope"] = 0
        else:
            f["overall_trend_slope"] = 0

    if groups["rejuvenation"]:
        if np.sum(areas > 0) > 0:
            minidx = np.argmin(areas)
            before = areas[:minidx] if minidx > 0 else np.array([0.0])
            after = areas[minidx+1:] if minidx < n-1 else np.array([0.0])
            f["min_location_ratio"] = minidx / n
            f["before_min_mean"] = np.mean(before) if len(before) else 0
            f["after_min_mean"] = np.mean(after) if len(after) else 0
            f["recovery_ratio"] = (f["after_min_mean"] / (f["before_min_mean"] + 1e-9)) if f["after_min_mean"] > 0 else 0
            if len(after) >= 5:
                try:
                    f["recovery_slope"] = np.polyfit(np.arange(len(after)), after, 1)[0]
                except:
                    f["recovery_slope"] = 0
            else:
                f["recovery_slope"] = 0
        else:
            f.update({"min_location_ratio":0,"before_min_mean":0,"after_min_mean":0,"recovery_ratio":0,"recovery_slope":0})

    if groups["vshape"]:
        half = n // 2
        mid = areas[n//3:2*n//3] if n >= 3 else np.array([0.0])
        fh = np.mean(areas[:half]) if half else 0
        sh = np.mean(areas[half:]) if n else 0
        mm = np.mean(mid) if len(mid) else 0
        f["first_half_mean"] = fh
        f["second_half_mean"] = sh
        f["middle_section_mean"] = mm
        f["vshape_indicator"] = (fh + sh) / (2*mm + 1e-9) if mm != 0 else 0

    if groups["zero_patterns"]:
        consec = 0; mx = 0
        for v in areas:
            if v == 0:
                consec += 1; mx = max(mx, consec)
            else:
                consec = 0
        f["max_consecutive_zeros"] = mx
        f["zeros_at_end"] = np.sum(areas[-max(1, int(n*0.2)):] == 0)

    if groups["variability"]:
        if n > 1:
            f["area_changes"] = np.sum(np.abs(np.diff(areas)))
            f["mean_change"] = np.mean(np.abs(np.diff(areas)))
        else:
            f["area_changes"] = 0
            f["mean_change"] = 0

    if groups["peaks"]:
        if n > 5:
            peaks = [i for i in range(1, n-1) if areas[i] > areas[i-1] and areas[i] > areas[i+1] and areas[i] > 0]
            f["num_peaks"] = len(peaks)
            f["peak_distance"] = (peaks[-1] - peaks[0]) if len(peaks) >= 2 else 0
        else:
            f["num_peaks"] = 0
            f["peak_distance"] = 0

    if groups["advanced"]:
        f["skewness"] = stats.skew(areas) if n > 2 else 0
        f["kurtosis"] = stats.kurtosis(areas) if n > 3 else 0
        f["percentile25"] = np.percentile(areas, 25) if n else 0
        f["percentile75"] = np.percentile(areas, 75) if n else 0
        f["iqr"] = f["percentile75"] - f["percentile25"]

        if n > 10:
            first10 = np.mean(areas[:10])
            last10 = np.mean(areas[-10:])
            f["start_end_ratio"] = last10 / (first10 + 1e-9) if last10 > 0 else 0
            f["start_end_diff"] = last10 - first10
        else:
            f["start_end_ratio"] = 0
            f["start_end_diff"] = 0

        if n > 1:
            r = np.diff(areas) / (areas[:-1] + 1e-9)
            f["volatility"] = np.std(r)
            f["mean_return"] = np.mean(r)
        else:
            f["volatility"] = 0
            f["mean_return"] = 0

    if groups["discriminative"]:
        f["had_early_water"] = 1 if np.sum(areas[:max(1, n//3)] > 0) > max(1, int(0.3*max(1, n//3))) else 0
        f["starts_from_zero"] = 1 if (n and areas[0] == 0) else 0
        f["late_period_mean"] = np.mean(areas[-max(1, n//4):]) if n else 0
        f["late_period_growing"] = 1 if (np.mean(areas[-max(1, n//4):]) > np.mean(areas[:max(1, n//2)])) else 0

    # sanitize
    for k in list(f.keys()):
        v = float(f[k]) if np.isfinite(f[k]) else 0.0
        f[k] = 0.0 if (np.isnan(v) or np.isinf(v)) else v
    return f


def build_feature_frame_three_stage(df, areacols, labelcol, groups):
    """Build a feature DataFrame from rows using the discriminative extractor."""
    lst = []
    for _, row in df.iterrows():
        d = extract_discriminative_features(row, areacols, groups)
        d[labelcol] = row[labelcol]
        lst.append(d)
    fdf = pd.DataFrame(lst).replace([np.inf, -np.inf], 0).fillna(0)
    featurenames = [c for c in fdf.columns if c != labelcol]
    X = fdf[featurenames].values
    y = fdf[labelcol].values
    return fdf, featurenames, X, y
