"""
data/loader.py
==============
Robust CSV/Excel reading, label-column detection, time-series column
identification, resampling, and unified preprocessing.
"""

import os
import re

import numpy as np
import pandas as pd

from ..config import (
    DATASET_FOLDER,
    ALL_KNOWN_CLASSES,
)
from .drive_utils import download_drive_link


# -----------------------------
# CSV/XLSX reading
# -----------------------------
def smart_read_csv(path_or_buf):
    """Robust CSV reader that auto-detects separator and encoding."""
    is_stream = hasattr(path_or_buf, "seek")

    seps = [",", ";", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    # --- PHASE 1: Find best separator/encoding using a sample ---
    best_sep = ","
    best_enc = "utf-8"
    max_cols = 0

    for encoding in encodings:
        for sep in seps:
            try:
                if is_stream: path_or_buf.seek(0)
                df_trial = pd.read_csv(path_or_buf, sep=sep, encoding=encoding, nrows=20, engine="python")
                cols = df_trial.shape[1]
                if cols > max_cols:
                    max_cols = cols
                    best_sep = sep
                    best_enc = encoding
            except:
                continue
        if max_cols > 1:
            break

    # --- PHASE 2: Read the full file with best detected params ---
    try:
        if is_stream: path_or_buf.seek(0)
        df = pd.read_csv(path_or_buf, sep=best_sep, encoding=best_enc, engine="python", on_bad_lines='skip')
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        try:
            if is_stream: path_or_buf.seek(0)
            df = pd.read_csv(path_or_buf, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except:
            raise ValueError(f"Could not read CSV file even with robust detection. Error: {str(e)}")


def read_table_path(path_or_obj):
    """Read CSV/Excel data and preserve the source filename in DataFrame attrs."""
    if hasattr(path_or_obj, "name"):
        original_name = os.path.basename(str(path_or_obj.name))
        path_str = str(path_or_obj.name).lower()
    else:
        original_name = os.path.basename(str(path_or_obj).rstrip("/"))
        path_str = str(path_or_obj).lower()

    if path_str.startswith(("http://", "https://")):
        file_obj = download_drive_link(path_or_obj)
        original_name = os.path.basename(getattr(file_obj, "name", "google_drive_dataset.csv"))
        lower_name = original_name.lower()

        if lower_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_obj)
            df.columns = [str(c).strip() for c in df.columns]
        else:
            df = smart_read_csv(file_obj)

        df.attrs["source_name"] = original_name
        return df

    if path_str.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_obj)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = smart_read_csv(path_or_obj)

    if original_name:
        df.attrs["source_name"] = original_name
    return df


# -----------------------------
# Final-status-only label extraction
# -----------------------------
def normalize_status(s):
    """Standardize pond status label strings."""
    s = str(s).strip().lower()
    mapping = {
        "extint": "extinct",
        "disappeared": "extinct",
        "dried": "extinct",
        "newlyformed": "newly formed",
        "new": "newly formed",
        "persistant": "persistent",
        "permanent": "persistent",
        "stable": "persistent",
        "rejuv": "rejuvenated",
        "recovered": "rejuvenated",
        "revival": "rejuvenated",
    }
    return mapping.get(s, s)


def find_label_col_finalstatus_only(df):
    """Detect the label column.
    Falls back to ALL_KNOWN_CLASSES for detection so any valid dataset is accepted."""
    def norm(s):
        return re.sub(r"[^a-zA-Z0-9]", "", str(s).strip().lower())

    for c in df.columns:
        if norm(c) == "finalstatus":
            return c

    for c in df.columns:
        if norm(c) == "status":
            return c

    detection_set = set(ALL_KNOWN_CLASSES)
    for c in df.columns:
        try:
            sample_values = set(df[c].dropna().astype(str).str.lower().unique()[:100])
            hits = 0
            for v in sample_values:
                if normalize_status(v) in detection_set:
                    hits += 1
            if hits >= 2:
                return c
        except:
            continue

    return None


# -----------------------------
# Identify time-series columns (two strategies)
# -----------------------------
def pick_areacols_like_three_stage(df):
    """Detect columns matching the 'col_sum' pattern used by Three-Stage models."""
    areacols = []
    for col in df.columns:
        c = str(col).lower()
        if ("col" in c) and (("sum" in c) or ("su" in c)):
            areacols.append(col)
    def _key(x):
        try:
            return int(str(x).split("_")[1])
        except Exception:
            return 10**9
    return sorted(areacols, key=_key)


def pick_temporalcols_like_xgb(df):
    """Detect numeric columns excluding known metadata fields."""
    exclude = {
        "ID","id","FID","fid","polyarea","Areahecta","areahec",
        "status","remarks","finalstatus","final status","final_status","final-status","label",
        "predictedstatus","confidence","reviewpriority","priorityrank","originalstatus"
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def resample_series_to_length(a, T):
    """Resample a 1-D array to length T using linear interpolation."""
    a = np.asarray(a, dtype=float)
    if len(a) == T:
        return a
    if len(a) < 2:
        return np.pad(a, (0, max(0, T-len(a))), constant_values=0)[:T]
    x_old = np.linspace(0, 1, len(a))
    x_new = np.linspace(0, 1, T)
    return np.interp(x_new, x_old, a)


def preprocess_dataset_to_unified(df, path, T=None, require_label=True):
    """Preprocess a dataframe into the unified format.

    ``source_row_positions`` is retained in metadata so Operational predictions
    can be written back to the exact original rows, even after invalid labels or
    untrained classes are excluded from evaluation.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    source_row_positions = np.arange(len(df), dtype=int)

    labelcol = find_label_col_finalstatus_only(df)
    is_real_label = True

    if labelcol is None:
        if require_label:
            cols_found = ", ".join([f"'{c}'" for c in df.columns[:20]])
            nrows, ncols = df.shape
            raise ValueError(f"Missing label column. Found {ncols} columns: [{cols_found}] across {nrows} rows. "
                             f"Please ensure your CSV uses commas or semicolons properly and has a 'final_status' column.")
        else:
            labelcol = "_dummy_label"
            df[labelcol] = "extinct"
            is_real_label = False

    if is_real_label:
        df[labelcol] = df[labelcol].apply(normalize_status)
        df[labelcol] = df[labelcol].replace(["nan", "none", "", "null"], np.nan)
        valid_mask = df[labelcol].notna().to_numpy()
        source_row_positions = source_row_positions[valid_mask]
        df = df.loc[valid_mask].copy()
        if len(df) == 0:
            raise ValueError("No valid data remaining after removing rows with empty or NaN labels.")

    # ensure ID
    if "ID" not in df.columns and "id" not in df.columns:
        df["ID"] = np.arange(len(df))
    elif "id" in df.columns and "ID" not in df.columns:
        df = df.rename(columns={"id": "ID"})

    # Detect series columns
    series_cols = pick_areacols_like_three_stage(df)
    if len(series_cols) < 5:
        for c in df.columns:
            if c not in ["ID", labelcol]:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        series_cols = pick_temporalcols_like_xgb(df)

    if len(series_cols) < 5:
        raise ValueError("No usable time-series columns found (need >= 5).")

    # numeric coerce
    for c in series_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if T is None:
        T = len(series_cols)
    T = int(T)
    if T < 5:
        raise ValueError(f"Invalid unified time-series length T={T}; at least 5 is required.")

    ts_cols = [f"TS_{i:03d}" for i in range(T)]
    mat = []
    for _, row in df.iterrows():
        a = row[series_cols].values.astype(float)
        a = resample_series_to_length(a, T)
        mat.append(a)

    ts_df = pd.DataFrame(mat, columns=ts_cols)
    out = pd.concat([df[["ID"]].reset_index(drop=True), ts_df, df[[labelcol]].reset_index(drop=True)], axis=1)
    out = out.rename(columns={labelcol: "label"})

    meta = {
        "path": path,
        "T": T,
        "orig_labelcol": labelcol if is_real_label else None,
        "is_real_label": is_real_label,
        "orig_series_cols": list(series_cols),
        "source_row_positions": source_row_positions.tolist(),
    }
    return out, meta


def autoload_and_preprocess_all():
    """Scan the configured dataset folder, load all CSV/Excel files, and preprocess."""
    raw = []
    logs = []

    folder = DATASET_FOLDER
    if not os.path.exists(folder):
        logs.append(f"Google Drive folder not found: {folder}")
        fallback_folder = "datasets"
        if os.path.exists(fallback_folder):
            logs.append(f"Falling back to local folder: {fallback_folder}")
            folder = fallback_folder
        else:
            logs.append(f"Local fallback folder also not found: {fallback_folder}")
            return {}, "\n".join(logs)

    files = []
    try:
        for f in os.listdir(folder):
            if f.endswith(('.csv', '.xlsx', '.xls')):
                files.append(os.path.join(folder, f))
    except Exception as e:
        logs.append(f"Error reading folder {folder}: {e}")
        return {}, "\n".join(logs)

    if not files:
        logs.append(f"No CSV/Excel files found in folder: {folder}")
        return {}, "\n".join(logs)

    for p in files:
        try:
            raw.append((p, read_table_path(p)))
        except Exception as e:
            logs.append(f"ERROR reading {p}: {e}")

    if not raw:
        logs.append("No preloaded datasets could be read. Application will start with empty dataset list.")
        logs.append("You can upload your own datasets using the 'Upload Your Dataset' tab.")
        return {}, "\n".join(logs)

    # choose global T = max detected series length across datasets
    lengths = []
    for p, df in raw:
        try:
            df2 = df.copy()
            labelcol = find_label_col_finalstatus_only(df2)
            if labelcol is None:
                continue
            series_cols = pick_areacols_like_three_stage(df2)
            if len(series_cols) < 5:
                for c in df2.columns:
                    if c not in ["ID", "id", labelcol]:
                        df2[c] = pd.to_numeric(df2[c], errors="ignore")
                series_cols = pick_temporalcols_like_xgb(df2)
            if len(series_cols) >= 5:
                lengths.append(len(series_cols))
        except:
            pass

    T = int(max(lengths)) if lengths else 30
    logs.append(f"Unified time-series length T = {T}")

    datasets = {}
    for p, df in raw:
        name = os.path.basename(p)
        try:
            uni, meta = preprocess_dataset_to_unified(df, p, T=T)
            datasets[name] = {"df": uni, "meta": meta}
            logs.append(f"Loaded {name}: rows={len(uni)} | TS={T} | label='label'")
        except Exception as e:
            logs.append(f"{name}: ERROR {e}")

    return datasets, "\n".join(logs)
