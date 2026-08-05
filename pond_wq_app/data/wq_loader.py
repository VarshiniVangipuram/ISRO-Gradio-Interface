"""
data/wq_loader.py
=================
Water Quality dataset loading, column detection, and auto-loading.
"""

import os
import re

import numpy as np
import pandas as pd

from ..config import WQ_DATASET_FOLDER, WQ_SHEET_WQ, WQ_SHEET_WSA
from .drive_utils import download_drive_link


def wq_load_excel(file_path_or_obj, sheet_wq=1, sheet_wsa=0):
    """Load WQ (NDCI/NDTI) and WSA sheets from an Excel file."""
    try:
        df_wq  = pd.read_excel(file_path_or_obj, sheet_name=int(sheet_wq))
        df_wsa = pd.read_excel(file_path_or_obj, sheet_name=int(sheet_wsa))
        return df_wq, df_wsa, None
    except Exception as e:
        return None, None, str(e)


def wq_detect_columns(df_wq, df_wsa=None):
    """Auto-detect NDCI, NDTI, WSA column groups."""
    ndci_cols = [c for c in df_wq.columns if 'NDCI' in str(c).upper()]
    ndti_cols = [c for c in df_wq.columns if 'NDTI' in str(c).upper()]
    wsa_cols = []
    if df_wsa is not None:
        wsa_exclude = {'id', 'area_hecta', 'area', 'status', 'latitude', 'longitude',
                       'perimeter', 'shape_leng', 'shape_area', 's.no', 'village_id', 'village id'}
        wsa_cols = [c for c in df_wsa.columns
                    if str(c).lower().strip() not in wsa_exclude
                    and pd.api.types.is_numeric_dtype(df_wsa[c])
                    and re.match(r'^col_\d+', str(c).lower())]
    return ndci_cols, ndti_cols, wsa_cols


def wq_read_data(upload_file, drive_link, sheet_wq, sheet_wsa):
    """Unified loader for WQ data."""
    src = None
    if upload_file is not None:
        src = upload_file if isinstance(upload_file, str) else upload_file.name if hasattr(upload_file, 'name') else upload_file
    elif drive_link and str(drive_link).strip():
        link = drive_link.strip()
        if link.startswith("http"):
            src = download_drive_link(link)
        else:
            src = link
    if src is None:
        return None, None, "No data source provided."
    return wq_load_excel(src, sheet_wq, sheet_wsa)


def autoload_wq_datasets():
    """
    Autoload all WQ datasets from the configured folder.
    Returns a dictionary of dataset names to their data.
    """
    datasets = {}
    logs = []

    folder = WQ_DATASET_FOLDER if os.path.exists(WQ_DATASET_FOLDER) else "datasets"
    if not os.path.exists(folder):
        logs.append(f"WQ folder not found: {folder}")
        return {}, "\n".join(logs)

    files = []
    try:
        for f in os.listdir(folder):
            if f.endswith(('.xlsx', '.xls')):
                files.append(os.path.join(folder, f))
    except Exception as e:
        logs.append(f"Error reading WQ folder {folder}: {e}")
        return {}, "\n".join(logs)

    if not files:
        logs.append(f"No Excel files found in WQ folder: {folder}")
        return {}, "\n".join(logs)

    for p in files:
        try:
            name = os.path.basename(p)
            df_wq = pd.read_excel(p, sheet_name=WQ_SHEET_WQ)
            df_wsa = pd.read_excel(p, sheet_name=WQ_SHEET_WSA)
            ndci_cols, ndti_cols, wsa_cols = wq_detect_columns(df_wq, df_wsa)

            datasets[name] = {
                "df_wq": df_wq,
                "df_wsa": df_wsa,
                "ndci_cols": ndci_cols,
                "ndti_cols": ndti_cols,
                "wsa_cols": wsa_cols,
                "path": p
            }
            logs.append(f"Loaded WQ dataset {name}: WQ rows={len(df_wq)}, WSA rows={len(df_wsa)}, NDCI={len(ndci_cols)}, NDTI={len(ndti_cols)}, WSA={len(wsa_cols)}")
        except Exception as e:
            logs.append(f"ERROR loading WQ dataset {p}: {e}")

    return datasets, "\n".join(logs)
