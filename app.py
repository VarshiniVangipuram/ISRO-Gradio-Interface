#!/usr/bin/env python3
"""
app.py — Main entry point for the Pond Classification & Water Quality Gradio app.

This module:
1. Mounts Google Drive (Colab) or skips (local).
2. Discovers operational models from Drive.
3. Preloads and preprocesses datasets.
4. Builds the Gradio Blocks UI via ui.layout.build_demo().
5. Launches the app.
"""

import os
import sys

# Colab-specific commands (commented out for local execution)
# COLAB MAGIC: !pip -q install gradio pandas numpy scikit-learn imbalanced-learn joblib xgboost catboost openpyxl matplotlib seaborn scipy statsmodels
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Google Colab. Skipping Drive mount.")

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pond_wq_app.config import (
    DATASET_FOLDER,
    WQ_DATASET_FOLDER,
    WQ_SHEET_WQ,
    WQ_SHEET_WSA,
    ALL_KNOWN_CLASSES,
    OPERATIONAL_MODELS,
    WQ_OPERATIONAL_MODELS,
)
from pond_wq_app.registry import (
    discover_pond_operational_models,
    discover_wq_operational_models,
    get_operational_model_trained_classes,
)
from pond_wq_app.data.loader import autoload_and_preprocess_all
from pond_wq_app.data.wq_loader import autoload_wq_datasets
from pond_wq_app.ui.render import success_summary_html
from pond_wq_app.controllers import set_boot_ctx
from pond_wq_app.ui.layout import build_demo


def _compute_boot_context():
    """Compute all boot-time values needed by the layout and controllers."""
    # Discover operational models from Drive
    pond_registry, pond_encoders, pond_metadata, pond_logs = discover_pond_operational_models()
    for line in pond_logs:
        print(f"[Pond model scan] {line}")

    wq_registry, wq_logs = discover_wq_operational_models()
    for line in wq_logs:
        print(f"[WQ model scan] {line}")

    # Preload & preprocess datasets
    preloaded_datasets, preload_log = autoload_and_preprocess_all()
    dataset_names = list(preloaded_datasets.keys())

    active_folder = DATASET_FOLDER if os.path.exists(DATASET_FOLDER) else (
        "datasets" if os.path.exists("datasets") else "datasets"
    )
    folder_display = os.path.basename(active_folder)
    parent_name = os.path.basename(os.path.dirname(active_folder))
    if parent_name and parent_name not in ["MyDrive", "content", ""]:
        folder_display = f"{parent_name}/{folder_display}"

    initial_classes = ALL_KNOWN_CLASSES
    if dataset_names:
        first_df = preloaded_datasets[dataset_names[0]]["df"]
        if "label" in first_df.columns:
            initial_classes = sorted(list(first_df["label"].dropna().unique()))

    # Preload Water Quality datasets
    wq_preloaded_datasets, wq_preload_log = autoload_wq_datasets()
    wq_dataset_names = list(wq_preloaded_datasets.keys())

    wq_active_folder = WQ_DATASET_FOLDER if os.path.exists(WQ_DATASET_FOLDER) else (
        "datasets" if os.path.exists("datasets") else "datasets"
    )
    wq_folder_display = os.path.basename(wq_active_folder)

    # For backward compatibility, keep wq_preloaded as single dataset
    wq_preloaded = {
        "df_wq": None, "df_wsa": None,
        "ndci_cols": [], "ndti_cols": [], "wsa_cols": [],
        "status_html": "",
    }
    if wq_dataset_names:
        first_dataset = wq_preloaded_datasets[wq_dataset_names[0]]
        wq_preloaded["df_wq"] = first_dataset["df_wq"]
        wq_preloaded["df_wsa"] = first_dataset["df_wsa"]
        wq_preloaded["ndci_cols"] = first_dataset["ndci_cols"]
        wq_preloaded["ndti_cols"] = first_dataset["ndti_cols"]
        wq_preloaded["wsa_cols"] = first_dataset["wsa_cols"]
        wq_preloaded["status_html"] = success_summary_html(
            rows=len(first_dataset["df_wq"]),
            label_col="N/A (unsupervised)",
            ts_cols=len(first_dataset["ndci_cols"]) + len(first_dataset["ndti_cols"]),
            T=len(first_dataset["ndci_cols"]),
            classes={
                "NDCI cols": len(first_dataset["ndci_cols"]),
                "NDTI cols": len(first_dataset["ndti_cols"]),
                "WSA cols": len(first_dataset["wsa_cols"]),
            },
        )
    else:
        wq_preloaded["status_html"] = (
            "<p style='color:gray;'>No WQ datasets found in folder. "
            "Upload your data manually.</p>"
        )

    print(wq_preload_log)

    # Initialize Operational classes from the selected model
    initial_op_model = list(OPERATIONAL_MODELS.keys())[0] if OPERATIONAL_MODELS else None
    if initial_op_model:
        initial_op_classes, _ = get_operational_model_trained_classes(initial_op_model)
    else:
        initial_op_classes = []

    initial_wq_model = list(WQ_OPERATIONAL_MODELS.keys())[0] if WQ_OPERATIONAL_MODELS else None

    return {
        "preloaded_datasets": preloaded_datasets,
        "dataset_names": dataset_names,
        "folder_display": folder_display,
        "initial_classes": initial_classes,
        "wq_preloaded_datasets": wq_preloaded_datasets,
        "wq_dataset_names": wq_dataset_names,
        "wq_folder_display": wq_folder_display,
        "wq_preloaded": wq_preloaded,
        "initial_op_model": initial_op_model,
        "initial_op_classes": initial_op_classes,
        "initial_wq_model": initial_wq_model,
    }


def main():
    boot = _compute_boot_context()
    set_boot_ctx(boot)
    demo = build_demo(boot)
    demo.queue(max_size=20).launch(share=True, debug=True)


if __name__ == "__main__":
    main()
