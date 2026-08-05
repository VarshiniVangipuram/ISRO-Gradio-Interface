"""
data/validation.py
===================
Dataset validation functions for uploaded pond and water-quality data.
"""

import os
import html
import traceback

import gradio as gr

from ..config import ALL_KNOWN_CLASSES
from .loader import read_table_path, preprocess_dataset_to_unified
from .drive_utils import _safe_source_filename
from .wq_loader import wq_read_data, wq_detect_columns
from ..ui.render import error_html, success_summary_html
from ..state import new_state


def validate_uploaded_dataset(upload_file, drive_path, st, require_label=True):
    """Validate and register an uploaded dataset for training."""
    try:
        df = None
        source_name = "Uploaded Dataset"

        # --- STEP 1: LOAD DATA ---
        if upload_file is not None:
            source_name = os.path.basename(upload_file.name)
            df = read_table_path(upload_file)
        elif drive_path and str(drive_path).strip():
            path = drive_path.strip()
            source_name = os.path.basename(path) if not path.startswith("http") else "Google Drive File"
            df = read_table_path(path)

        # Basic Check
        if df is None:
            return False, error_html("No data found. Please upload a file or paste a link."), st, None, gr.update(), gr.update(), gr.update()

        # --- STEP 2: GET GLOBAL T FOR CONSISTENCY ---
        T = None
        if st["datasets"]:
            first_ds = list(st["datasets"].values())[0]
            T = first_ds["meta"]["T"]

        # --- STEP 3: PREPROCESS & VALIDATE (Labels, Numeric TS, etc.) ---
        try:
            processed_df, meta = preprocess_dataset_to_unified(
                df, source_name, T=T,
                require_label=require_label
            )
        except ValueError as e:
            error_msg = str(e)
            if "Missing label" in error_msg:
                error_msg = "Missing required label column.<br>Expected one of: <code>finalstatus</code>, <code>final status</code>, <code>final_status</code>, <code>final-status</code>"
            elif "No usable time-series columns" in error_msg:
                error_msg = f"{error_msg}<br>Please ensure your dataset has at least 5 numeric columns representing time-series data."
            return False, error_html(error_msg), st, None, gr.update(), gr.update(), gr.update()

        # --- STEP 4: FINAL CHECKS & REGISTRATION ---
        if len(processed_df) == 0:
            return False, error_html(
                "No valid data rows found in the dataset.<br>"
                "Please ensure the file is not empty."
            ), st, None, gr.update(), gr.update(), gr.update()

        # Register in State
        dataset_name = f"Uploaded: {source_name}"
        st["datasets"][dataset_name] = {"df": processed_df, "meta": meta}

        # Generate Success UI
        class_counts = processed_df["label"].value_counts()
        summary = success_summary_html(
            rows=len(processed_df),
            label_col=meta["orig_labelcol"],
            ts_cols=len(meta["orig_series_cols"]),
            T=meta["T"],
            classes=class_counts.to_dict()
        )

        all_datasets = list(st["datasets"].keys())

        # Determine unique classes dynamically for the CheckboxGroups
        unique_classes = sorted(list(processed_df["label"].dropna().unique())) if "label" in processed_df.columns else []
        if not unique_classes:
            unique_classes = ALL_KNOWN_CLASSES
        checkbox_update = gr.update(choices=unique_classes, value=unique_classes)

        return True, summary, st, dataset_name, gr.update(choices=all_datasets, value=[dataset_name]), checkbox_update, gr.update()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return False, error_html(f"Unexpected error during validation:<br><small>{str(e)}</small>"), st, None, gr.update(), gr.update(), gr.update()


def validate_operational_dataset(upload_file, drive_path, st):
    """Validate Operational input without changing either class selector.

    Operational class choices always remain model-driven. An unlabeled file is
    reported as prediction-only instead of displaying the internal dummy label.
    """
    try:
        if upload_file is not None:
            raw_df = read_table_path(upload_file)
            source_name = (
                raw_df.attrs.get("source_name")
                or _safe_source_filename(upload_file=upload_file)
            )
            source_id = str(upload_file)
        elif drive_path and str(drive_path).strip():
            path = str(drive_path).strip()
            raw_df = read_table_path(path)
            source_name = (
                raw_df.attrs.get("source_name")
                or _safe_source_filename(drive_path=path)
            )
            source_id = path
        else:
            return (
                False,
                error_html("No data found. Please upload a file or paste a link."),
                st,
                None,
                gr.update(),
                gr.update(),
                gr.update(),
            )

        processed_df, meta = preprocess_dataset_to_unified(
            raw_df, source_name, T=None, require_label=False
        )
        if processed_df.empty:
            return (
                False,
                error_html("No valid rows were found in the Operational dataset."),
                st,
                None,
                gr.update(),
                gr.update(),
                gr.update(),
            )

        # Cache only the raw source. Model-specific preprocessing is performed
        # after a model is selected and its exact training T is known.
        st["op_original_df"] = raw_df.copy()
        st["op_source_id"] = source_id
        st["op_source_name"] = source_name
        st["op_cache_key"] = None
        st["op_work_df"] = None
        st["op_meta"] = None
        st["op_predictions"] = {}

        if meta.get("is_real_label", False):
            class_counts = processed_df["label"].value_counts().to_dict()
            class_items = "".join(
                f"<li><b>{html.escape(str(cls))}</b>: {int(count)}</li>"
                for cls, count in class_counts.items()
            )
            label_text = html.escape(str(meta.get("orig_labelcol")))
            class_section = f"<ul style='margin:4px 0 0 20px;'>{class_items}</ul>"
        else:
            label_text = "Not present (prediction-only input)"
            class_section = (
                "<div class='muted-copy' style='margin-top:4px;'>"
                "No ground-truth classes were read. Available Operational classes "
                "continue to come only from the selected model.</div>"
            )

        if meta.get("is_real_label", False):
            class_pills = "".join(
                f"<span class='class-pill'>{html.escape(str(cls))} · {int(count)}</span>"
                for cls, count in class_counts.items()
            )
        else:
            class_pills = (
                "<span class='class-pill class-pill-muted'>Prediction-only input</span>"
            )

        summary = f"""
        <div class='dataset-ready-card'>
            <div class='dataset-ready-head'>
                <div>
                    <span class='eyebrow'>VALIDATED INPUT</span>
                    <div class='dataset-ready-title'>Dataset ready</div>
                </div>
                <span class='success-orb'>✓</span>
            </div>
            <div class='compact-stat-grid'>
                <div><span>File</span><b title='{html.escape(source_name)}'>{html.escape(source_name)}</b></div>
                <div><span>Rows</span><b>{len(processed_df):,}</b></div>
                <div><span>Temporal columns</span><b>{len(meta.get('orig_series_cols', []))}</b></div>
                <div><span>Series length</span><b>{meta.get('T')}</b></div>
            </div>
            <div class='dataset-ready-meta'>
                <span><b>Ground truth:</b> {label_text}</span>
            </div>
            <div class='class-pill-row'>{class_pills}</div>
        </div>
        """
        # No Advanced dataset/class update and no Operational class update.
        return True, summary, st, None, gr.update(), gr.update(), gr.update()

    except Exception as exc:
        return (
            False,
            error_html(
                f"Operational validation failed:<br><small>{html.escape(str(exc))}</small>"
            ),
            st,
            None,
            gr.update(),
            gr.update(),
            gr.update(),
        )


def wq_validate_data(upload_file, drive_link, sheet_wq, sheet_wsa, st):
    """Validate uploaded WQ data."""
    st = st or new_state()
    df_wq, df_wsa, err = wq_read_data(upload_file, drive_link, int(sheet_wq), int(sheet_wsa))
    if err:
        return error_html(f"Failed to load data: {err}"), st, gr.update()
    if df_wq is None:
        return error_html("No data loaded."), st, gr.update()
    ndci, ndti, wsa = wq_detect_columns(df_wq, df_wsa)
    if len(ndci) == 0:
        return error_html("No NDCI columns found in WQ sheet."), st, gr.update()

    # Determine a stable source filename for display and result downloads.
    source_name = _safe_source_filename(
        upload_file=upload_file,
        drive_path=drive_link,
        fallback="water_quality_dataset.xlsx",
    )

    dataset_name = f"Uploaded: {source_name}"

    if "wq_datasets" not in st:
        st["wq_datasets"] = {}

    st["wq_datasets"][dataset_name] = {
        "df_wq": df_wq,
        "df_wsa": df_wsa,
        "ndci_cols": ndci,
        "ndti_cols": ndti,
        "wsa_cols": wsa,
        "path": upload_file.name if hasattr(upload_file, 'name') else (upload_file if isinstance(upload_file, str) else "") if upload_file else (drive_link if drive_link else "")
    }

    st["wq_df_wq"]    = df_wq
    st["wq_df_wsa"]   = df_wsa
    st["wq_ndci_cols"] = ndci
    st["wq_ndti_cols"] = ndti
    st["wq_wsa_cols"]  = wsa
    st["wq_model1_done"] = False
    st["wq_result_df"] = None
    st["wq_master_df"] = None  # Reset master on new dataset upload
    st["wq_active_source_name"] = source_name

    summary = success_summary_html(
        rows=len(df_wq), label_col="N/A (unsupervised)", ts_cols=len(ndci)+len(ndti),
        T=len(ndci), classes={"NDCI cols": len(ndci), "NDTI cols": len(ndti), "WSA cols": len(wsa)}
    )

    all_wq_datasets = list(st["wq_datasets"].keys())
    return summary, st, gr.update(choices=all_wq_datasets, value=dataset_name)
