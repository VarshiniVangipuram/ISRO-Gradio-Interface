"""
Controller / orchestration functions that wire UI events to backend logic.
These are the "runner" functions invoked by Gradio button clicks and dropdown changes.
"""

import os
import re
import time
import html
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gradio as gr
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Boot context is set by app.py before the Gradio Blocks builder runs.
_boot_ctx = None

def set_boot_ctx(ctx):
    """Called by app.py to inject boot-time preloaded data."""
    global _boot_ctx
    _boot_ctx = ctx

def init_state():
    """Initialize application state with preloaded datasets."""
    st = new_state()
    if _boot_ctx is None:
        return st
    st["datasets"] = _boot_ctx["preloaded_datasets"]
    st["wq_datasets"] = _boot_ctx["wq_preloaded_datasets"].copy()
    # Inject preloaded WQ data
    wq_preloaded = _boot_ctx.get("wq_preloaded", {})
    if wq_preloaded.get("df_wq") is not None:
        st["wq_df_wq"]     = wq_preloaded["df_wq"]
        st["wq_df_wsa"]    = wq_preloaded["df_wsa"]
        st["wq_ndci_cols"] = wq_preloaded["ndci_cols"]
        st["wq_ndti_cols"] = wq_preloaded["ndti_cols"]
        st["wq_wsa_cols"]  = wq_preloaded["wsa_cols"]
        wq_dataset_names = _boot_ctx.get("wq_dataset_names", [])
        wq_preloaded_datasets = _boot_ctx.get("wq_preloaded_datasets", {})
        if wq_dataset_names:
            first_wq_entry = wq_preloaded_datasets[wq_dataset_names[0]]
            st["wq_active_source_name"] = os.path.basename(
                str(first_wq_entry.get("path") or wq_dataset_names[0])
            )
    return st

from .config import (
    DEFAULT_TARGET_CLASSES,
    ALL_KNOWN_CLASSES,
    DATASET_FOLDER,
    WQ_DATASET_FOLDER,
    WQ_SHEET_WQ,
    WQ_SHEET_WSA,
    OPERATIONAL_MODELS,
    OPERATIONAL_LABEL_ENCODERS,
    OPERATIONAL_MODEL_METADATA,
    WQ_OPERATIONAL_MODELS,
    _USER_TRAINED_MODELS,
    _USER_TRAINED_WQ_MODELS,
    _OPERATIONAL_MODEL_CACHE,
    _WQ_OPERATIONAL_MODEL_CACHE,
    GENERATED_RESULTS_FOLDER,
)
from .state import new_state, history_df, recompute_best, undo_to_experiment
from .data.loader import (
    read_table_path,
    preprocess_dataset_to_unified,
    autoload_and_preprocess_all,
    normalize_status,
)
from .data.drive_utils import (
    _safe_source_filename,
    save_result_csv,
    download_drive_link,
)
from .data.validation import validate_uploaded_dataset, validate_operational_dataset, wq_validate_data
from .data.wq_loader import (
    wq_load_excel,
    wq_detect_columns,
    wq_read_data,
    autoload_wq_datasets,
)
from .features.shape import extract_shape_only_features, extract_shape_features_8
from .features.fast import extractfeaturesfast
from .features.discriminative import extract_discriminative_features
from .features.universal import extract_universal_features
from .models.three_stage import train_three_stage
from .models.single_xgb import train_single_xgb
from .models.universal import train_generalized_universal_model
from .models.custom import train_custom_template
from .models.wq import wq_model1_run, wq_model2_run, wq_model3_run
from .registry import (
    discover_pond_operational_models,
    discover_wq_operational_models,
    rescan_pond_model_registry,
    rescan_wq_model_registry,
    _pond_model_choices,
    _wq_model_choices,
    deploy_experiment_to_operational,
    save_deployed_pond_model_to_drive,
    deploy_wq_model_to_operational,
    save_deployed_wq_model_to_drive,
    load_pretrained_model,
    build_model_display_name,
    _operational_model_path,
    _normalise_model_class_values,
    extract_trained_classes_from_bundle,
    get_operational_model_trained_classes,
    _safe_joblib_load,
)
from .ui.render import (
    error_html,
    success_summary_html,
    render_performance_dashboard,
    render_classification_report,
    render_best_model_status,
    render_dataset_cards,
    render_wq_dataset_cards,
    render_wq_summary_html,
    get_model_details,
    get_wq_model_details,
    update_operational_model_ui,
)
from .ui.plots import plot_confusion, plot_feature_importance_rf, plot_actual_vs_pred


# -----------------------------
# Download Best Model Function
# -----------------------------
def download_best_model(state):
    """
    Creates a downloadable file for the best model from training history.
    Returns the file path for Gradio to serve as a download.
    """
    try:
        if not state.get("history"):
            return None

        best_entry = max(state["history"], key=lambda e: e["acc"])
        bundle = best_entry.get("bundle")
        if bundle is None:
            return None

        import tempfile
        model_name = build_model_display_name(best_entry)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "best_model"
        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, f"{safe_name}.pkl")
        joblib.dump(bundle, file_path)
        return file_path
    except Exception:
        traceback.print_exc()
        return None


# -----------------------------
# Training runner
# -----------------------------
def run_experiment(dataset_name, st, model_kind,
                   # single xgb options
                   use_tempered_weights, alpha, cap, use_gpu,
                   x_n_estimators, x_lr, x_max_depth, x_min_child_weight, x_gamma,
                   x_subsample, x_colsample, x_reg_lambda, x_reg_alpha, x_early_stop,
                   # generalized universal (T-agnostic) options
                   u_use_tempered_weights, u_alpha, u_cap,
                   u_n_estimators, u_lr, u_max_depth, u_min_child_weight, u_gamma,
                   u_subsample, u_colsample, u_reg_lambda, u_reg_alpha, u_early_stop,
                   # feature groups
                   g_basic, g_water, g_quarters, g_trend, g_rejuv, g_vshape, g_zero, g_var, g_peaks, g_adv, g_disc,
                   # common
                   test_size, random_state,
                   # three-stage options
                   use_adasyn, adasyn_neighbors, auto_opt, fs_threshold,
                   s1_n, s1_depth, s1_lr, s1_spw,
                   s2_n, s2_depth, s2_lr, s2_spw,
                   s3_iter, s3_depth, s3_lr,
                   init_rth, init_nth,
                   custom_template_name, custom_use_scaler, custom_use_class_weight,
                   custom_lr_C, custom_lr_max_iter,
                   custom_svc_C, custom_svc_kernel, custom_svc_gamma,
                   custom_knn_k,
                   custom_rf_n, custom_rf_depth,
                   custom_cxgb_n, custom_cxgb_lr, custom_cxgb_depth,
                   # dynamic target classes
                   target_classes=None):

    # Resolve target classes
    if target_classes is None or len(target_classes) == 0:
        target_classes = DEFAULT_TARGET_CLASSES

    st = st or new_state()
    if isinstance(dataset_name, str):
        dataset_names = [dataset_name]
    else:
        dataset_names = dataset_name or []

    if not dataset_names:
        return st, "Please select at least one dataset.", None, None, None, None, history_df(st), "Please select at least one dataset.", gr.update(choices=[]), "Please select at least one dataset."

    # Validate that all selected datasets exist
    for name in dataset_names:
        if name not in st["datasets"]:
            return st, f"Dataset '{name}' not found.", None, None, None, None, history_df(st), f"Dataset '{name}' not found.", gr.update(choices=[]), f"Dataset '{name}' not found."

    # Concatenate all selected datasets
    dfs_to_concat = []
    ts_lengths = []
    for name in dataset_names:
        ds = st["datasets"][name]
        dfs_to_concat.append(ds["df"].copy())
        ts_lengths.append(ds["meta"]["T"])

    df = pd.concat(dfs_to_concat, ignore_index=True)
    labelcol = "label"
    tscols = [c for c in df.columns if c.startswith("TS_")]

    # --- Filter to selected target classes ---
    df = df[df[labelcol].isin(target_classes)].copy()
    if len(df) == 0:
        cls_str = ", ".join(target_classes)
        err = f"No rows remain after filtering to selected classes: {cls_str}.\nPlease select classes that actually appear in this dataset."
        return st, "", None, None, None, err, history_df(st), err, gr.update(choices=list(range(0, len(st["history"]) + 1))), err

    # Three-Stage model requires exactly rejuvenated + newly formed + (persistent and/or extinct)
    if model_kind.startswith("Three-Stage"):
        required_classes = {"rejuvenated", "newly formed"}
        available_classes = set(df[labelcol].unique())
        missing = required_classes - available_classes
        if missing:
            err = (f"Three-Stage model requires classes: {required_classes}.\n"
                   f"Missing from selected/dataset: {missing}.\n"
                   f"Please select all required classes or use a different model.")
            return st, "", None, None, None, err, history_df(st), err, gr.update(choices=list(range(0, len(st["history"]) + 1))), err
        if len(available_classes) < 2:
            err = "Three-Stage model needs at least 2 distinct classes in the data."
            return st, "", None, None, None, err, history_df(st), err, gr.update(choices=list(range(0, len(st["history"]) + 1))), err


    groups = {
        "basic": bool(g_basic),
        "water": bool(g_water),
        "quarters": bool(g_quarters),
        "trend": bool(g_trend),
        "rejuvenation": bool(g_rejuv),
        "vshape": bool(g_vshape),
        "zero_patterns": bool(g_zero),
        "variability": bool(g_var),
        "peaks": bool(g_peaks),
        "advanced": bool(g_adv),
        "discriminative": bool(g_disc),
    }

    try:
        if not model_kind:
            return st, "Please select a model type.", None, None, None, None, history_df(st), "", gr.update(choices=[]), ""

        if model_kind.startswith("Generalized Universal"):
            result = train_generalized_universal_model(
                df, labelcol, tscols,
                test_size=float(test_size),
                random_state=int(random_state),
                use_tempered_weights=bool(u_use_tempered_weights),
                alpha=float(u_alpha),
                cap=float(u_cap),
                n_estimators=int(u_n_estimators),
                learning_rate=float(u_lr),
                max_depth=int(u_max_depth),
                min_child_weight=float(u_min_child_weight),
                gamma=float(u_gamma),
                subsample=float(u_subsample),
                colsample_bytree=float(u_colsample),
                reg_lambda=float(u_reg_lambda),
                reg_alpha=float(u_reg_alpha),
                early_stopping_rounds=int(u_early_stop),
                use_gpu=bool(use_gpu),
            )
        elif model_kind.startswith("Three-Stage"):
            result = train_three_stage(
                df, tscols, labelcol, groups,
                test_size=float(test_size),
                random_state=int(random_state),
                use_adasyn=bool(use_adasyn),
                adasyn_neighbors=int(adasyn_neighbors),
                auto_optimize_thresholds=bool(auto_opt),
                s1_n=int(s1_n), s1_depth=int(s1_depth), s1_lr=float(s1_lr), s1_spw=float(s1_spw),
                s2_n=int(s2_n), s2_depth=int(s2_depth), s2_lr=float(s2_lr), s2_spw=float(s2_spw),
                s3_iter=int(s3_iter), s3_depth=int(s3_depth), s3_lr=float(s3_lr),
                init_rth=float(init_rth), init_nth=float(init_nth),
                fs_threshold=str(fs_threshold),
            )
        elif model_kind.startswith("Single XGBoost"):
            result = train_single_xgb(
                df, labelcol, tscols,
                test_size=float(test_size),
                random_state=int(random_state),
                use_tempered_weights=bool(use_tempered_weights),
                alpha=float(alpha), cap=float(cap),
                n_estimators=int(x_n_estimators),
                learning_rate=float(x_lr),
                max_depth=int(x_max_depth),
                min_child_weight=float(x_min_child_weight),
                gamma=float(x_gamma),
                subsample=float(x_subsample),
                colsample_bytree=float(x_colsample),
                reg_lambda=float(x_reg_lambda),
                reg_alpha=float(x_reg_alpha),
                early_stopping_rounds=int(x_early_stop),
                use_gpu=bool(use_gpu),
                dataset_name=dataset_name
            )

        elif model_kind.startswith("Custom Models"):
            result = train_custom_template(
                df, labelcol, tscols,
                test_size=float(test_size),
                random_state=int(random_state),
                template_name=str(custom_template_name),
                use_scaler=bool(custom_use_scaler),
                use_class_weight=bool(custom_use_class_weight),
                lr_C=float(custom_lr_C),
                lr_max_iter=int(custom_lr_max_iter),
                svc_C=float(custom_svc_C),
                svc_kernel=str(custom_svc_kernel),
                svc_gamma=str(custom_svc_gamma),
                knn_k=int(custom_knn_k),
                rf_n=int(custom_rf_n),
                rf_depth=int(custom_rf_depth),
                cxgb_n=int(custom_cxgb_n),
                cxgb_lr=float(custom_cxgb_lr),
                cxgb_depth=int(custom_cxgb_depth)
            )
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        exp_id = len(st["history"]) + 1
        thresholds = result["artifacts"].get("thresholds", {})
        features_selected = len(result["artifacts"].get("selectedfeatures", []))

        # Store ALL UI parameters for undo functionality
        params = {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "use_gpu": bool(use_gpu),
        }

        # Single XGBoost parameters
        if model_kind.startswith("Single XGBoost"):
            params.update({
                "use_tempered_weights": bool(use_tempered_weights),
                "alpha": float(alpha),
                "cap": float(cap),
                "x_n_estimators": int(x_n_estimators),
                "x_lr": float(x_lr),
                "x_max_depth": int(x_max_depth),
                "x_min_child_weight": int(x_min_child_weight),
                "x_gamma": float(x_gamma),
                "x_subsample": float(x_subsample),
                "x_colsample": float(x_colsample),
                "x_reg_lambda": float(x_reg_lambda),
                "x_reg_alpha": float(x_reg_alpha),
                "x_early_stop": int(x_early_stop),
            })

        # Generalized Universal (T-agnostic) parameters
        if model_kind.startswith("Generalized Universal"):
            params.update({
                "u_use_tempered_weights": bool(u_use_tempered_weights),
                "u_alpha": float(u_alpha),
                "u_cap": float(u_cap),
                "u_n_estimators": int(u_n_estimators),
                "u_lr": float(u_lr),
                "u_max_depth": int(u_max_depth),
                "u_min_child_weight": float(u_min_child_weight),
                "u_gamma": float(u_gamma),
                "u_subsample": float(u_subsample),
                "u_colsample": float(u_colsample),
                "u_reg_lambda": float(u_reg_lambda),
                "u_reg_alpha": float(u_reg_alpha),
                "u_early_stop": int(u_early_stop),
            })

        # Three-Stage parameters
        if model_kind.startswith("Three-Stage"):
            params.update({
                "use_adasyn": bool(use_adasyn),
                "adasyn_neighbors": int(adasyn_neighbors),
                "auto_optimize_thresholds": bool(auto_opt),
                "fs_threshold": str(fs_threshold),
                "s1": {"n": int(s1_n), "depth": int(s1_depth), "lr": float(s1_lr), "spw": float(s1_spw)},
                "s2": {"n": int(s2_n), "depth": int(s2_depth), "lr": float(s2_lr), "spw": float(s2_spw)},
                "s3": {"iter": int(s3_iter), "depth": int(s3_depth), "lr": float(s3_lr)},
                "init_thresholds": {"rejuv": float(init_rth), "newf": float(init_nth)},
                "feature_groups": groups
            })

        # Custom Models parameters
        if model_kind.startswith("Custom Models"):
            params.update({
                "custom_template_name": str(custom_template_name),
                "custom_use_scaler": bool(custom_use_scaler),
                "custom_use_class_weight": bool(custom_use_class_weight),
                "custom_lr_C": float(custom_lr_C),
                "custom_lr_max_iter": int(custom_lr_max_iter),
                "custom_svc_C": float(custom_svc_C),
                "custom_svc_kernel": str(custom_svc_kernel),
                "custom_svc_gamma": str(custom_svc_gamma),
                "custom_knn_k": int(custom_knn_k),
                "custom_rf_n": int(custom_rf_n),
                "custom_rf_depth": int(custom_rf_depth),
                "custom_cxgb_n": int(custom_cxgb_n),
                "custom_cxgb_lr": float(custom_cxgb_lr),
                "custom_cxgb_depth": int(custom_cxgb_depth),
            })

        entry = {
            "experiment_id": exp_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": ", ".join(dataset_names),
            "model_kind": model_kind,
            "acc": result["acc"],
            "macro_f1": result["macro_f1"],
            "features_selected": features_selected,
            "thresholds": thresholds,
            "params": params,
            "bundle": result["artifacts"],
            "result": result,  # Store full result for undo
        }
        st["history"].append(entry)

        best_msg = ""
        if result["acc"] > st["best"]["acc"]:
            st["best"]["acc"] = result["acc"]
            st["best"]["bundle"] = result["artifacts"]

          # Generate HTML for all models
        best_html = render_best_model_status(st["history"])

        cm_fig = plot_confusion(result["cm"], result["classes"], result["acc"])
        avp_fig = plot_actual_vs_pred(result["actual_counts"], result["pred_counts"], result["acc"])

        fi_fig = None
        if model_kind.startswith("Three-Stage"):
            fi_fig = plot_feature_importance_rf(result["artifacts"]["rfselector"], result["artifacts"].get("featurenames", []), topk=20)

        undo_choices = list(range(0, len(st["history"]) + 1))

        # Generate modern dashboard summary
        template_name = None
        if model_kind.startswith("Custom Models"):
            template_name = result['artifacts'].get('template_name', '')

        summary = render_performance_dashboard(
            exp_id=exp_id,
            model_kind=model_kind,
            dataset_name=", ".join(dataset_names),
            acc=result['acc'],
            macro_f1=result['macro_f1'],
            rows=len(df),
            ts_length=len(tscols),
            thresholds=thresholds if model_kind.startswith("Three-Stage") else None,
            template_name=template_name,
            features_selected=features_selected
        )

        best_html_content = render_best_model_status(st["history"])

        # Generate modern classification report
        report_out = render_classification_report(result["report"], result["classes"])

        return st, best_html_content, cm_fig, fi_fig, avp_fig, report_out, history_df(st), summary, gr.update(choices=undo_choices, value=len(st["history"])), ""

    except Exception as e:
        err = f"ERROR: {e}\n\n{traceback.format_exc()}"
        undo_choices = list(range(0, len(st["history"]) + 1))
        return st, "", None, None, None, err, history_df(st), err, gr.update(choices=undo_choices, value=len(st["history"])), err


# -----------------------------
# WQ Analysis runner
# -----------------------------
def wq_run_analysis(model_choice, n_clusters, k_range_max, pca_components,
                    bloom_threshold, m3_n_clusters, m3_pca_each,
                    upload_file, drive_link, sheet_wq, sheet_wsa, st):
    """Main WQ analysis runner."""
    st = st or new_state()

    # Auto-load if needed
    if st.get("wq_df_wq") is None:
        df_wq, df_wsa, err = wq_read_data(upload_file, drive_link, int(sheet_wq), int(sheet_wsa))
        if err or df_wq is None:
            return st, error_html("Load data first."), None, None, None, None, None, None, None, ""
        ndci, ndti, wsa = wq_detect_columns(df_wq, df_wsa)
        st["wq_df_wq"] = df_wq; st["wq_df_wsa"] = df_wsa
        st["wq_ndci_cols"] = ndci; st["wq_ndti_cols"] = ndti; st["wq_wsa_cols"] = wsa
        st["wq_active_source_name"] = _safe_source_filename(
            upload_file=upload_file,
            drive_path=drive_link,
            fallback="water_quality_dataset.xlsx",
        )

    df_wq = st["wq_df_wq"]
    ndci = st["wq_ndci_cols"]; ndti = st["wq_ndti_cols"]; wsa = st["wq_wsa_cols"]

    summary_html = ""; fig1 = None; fig2 = None; fig3 = None
    fig4 = None; fig5 = None; fig6 = None; fig7 = None; dl_path = ""

    try:
        # Initialize master DataFrame once per dataset (accumulates all model results side-by-side)
        if st.get("wq_master_df") is None:
            st["wq_master_df"] = df_wq.copy()
        master_df = st["wq_master_df"]

        if model_choice == "Model 1: PCA + KMeans + TS + Resilience":
            df_result, scores, figs, artifacts = wq_model1_run(
                df_wq, ndci, ndti, int(n_clusters), int(k_range_max),
                int(pca_components), float(bloom_threshold))
            st["wq_result_df"] = df_result; st["wq_scores"] = scores; st["wq_model1_done"] = True
            st["wq_last_artifacts"] = artifacts
            summary_html = render_wq_summary_html(df_result, scores, "PCA + KMeans + TS + Resilience")
            fig1 = figs.get('elbow_sil'); fig2 = figs.get('ch_score')
            fig3 = figs.get('behavioral'); fig4 = figs.get('temporal')
            fig5 = figs.get('ts_class'); fig6 = figs.get('resilience')
            # Merge M1 results into master with model prefix (never overwrites other models)
            for _col in ["Cluster", "Typology", "TS_Class", "Resilience_Class"]:
                if _col in df_result.columns:
                    master_df[f"M1_{_col}"] = df_result[_col].values
            st["wq_master_df"] = master_df

        elif model_choice == "Model 2: DTW Shape Classifier":
            if not st.get("wq_model1_done") or st.get("wq_result_df") is None:
                return st, error_html("Run Model 1 first to generate base classifications."), None, None, None, None, None, None, None, ""
            df_result, figs, artifacts = wq_model2_run(st["wq_result_df"], ndci)
            st["wq_result_df"] = df_result
            st["wq_last_artifacts"] = artifacts
            summary_html = render_wq_summary_html(df_result, model_name="DTW Shape Classifier")
            fig1 = figs.get('heatmap_counts'); fig2 = figs.get('heatmap_norm')
            fig3 = figs.get('stacked_bar'); fig4 = figs.get('shape_dist')
            # Merge M2 results into master with model prefix
            for _col in ["Shape_Regime"]:
                if _col in df_result.columns:
                    master_df[f"M2_{_col}"] = df_result[_col].values
            st["wq_master_df"] = master_df

        elif model_choice == "Model 3: WSA + NDCI + NDTI Triple-Stream":
            if st.get("wq_df_wsa") is None or len(wsa) == 0:
                return st, error_html("WSA sheet not loaded or no WSA columns found."), None, None, None, None, None, None, None, ""
            df_result, figs, artifacts = wq_model3_run(
                df_wq, st["wq_df_wsa"], ndci, ndti, wsa,
                int(m3_n_clusters), int(m3_pca_each))
            st["wq_result_df"] = df_result
            st["wq_last_artifacts"] = artifacts
            summary_html = render_wq_summary_html(df_result, model_name="Triple-Stream Analysis")
            fig1 = figs.get('scatter'); fig2 = figs.get('boxplot')
            fig3 = figs.get('typology_dist')
            # Merge M3 results into master with model prefix.
            # df_result is indexed by ID values (e.g. 1,6,8...), master_df by integer positions (0,1,2...).
            # Using ID-column mapping to correctly align every row — NOT reindex which mismatch them.
            for _col in ["Cluster", "Typology"]:
                if _col in df_result.columns:
                    if 'ID' in master_df.columns:
                        _id_to_val = df_result[_col].to_dict()  # {ID_value: result}
                        master_df[f"M3_{_col}"] = master_df['ID'].map(_id_to_val)
                    else:
                        # No ID column: align by position (both integer-indexed)
                        master_df[f"M3_{_col}"] = df_result[_col].values[:len(master_df)]
            st["wq_master_df"] = master_df

        # Save CSV from master DataFrame — all model results accumulated, columns never overwritten
        if st.get("wq_master_df") is not None:
            dl_path = save_result_csv(
                st["wq_master_df"],
                st.get("wq_active_source_name") or "water_quality_dataset.xlsx",
                suffix="wq_analysis",
            )

    except Exception as e:
        summary_html = error_html(f"Analysis error: {str(e)}<br><pre>{traceback.format_exc()}</pre>")

    return st, summary_html, fig1, fig2, fig3, fig4, fig5, fig6, fig7, dl_path


# -----------------------------
# WQ Operational Predict
# -----------------------------
def wq_operational_predict(model_key, upload_file, drive_link, sheet_wq, sheet_wsa, st):
    """Operational mode: load saved .pkl bundle and apply to new WQ data."""
    st = st or new_state()
    df_wq, df_wsa, err = wq_read_data(upload_file, drive_link, int(sheet_wq), int(sheet_wsa))
    if err or df_wq is None:
        return error_html(f"Failed to load: {err}"), None, None, None, None, st
    ndci, ndti, wsa = wq_detect_columns(df_wq, df_wsa)
    if len(ndci) == 0:
        return error_html("No NDCI columns detected."), None, None, None, None, st

    df_result = df_wq.copy()

    source_name = _safe_source_filename(
        upload_file=upload_file,
        drive_path=drive_link,
        fallback="water_quality_dataset.xlsx",
    )

    # Reset master_df if dataset changed (different upload file or link)
    current_source = str(upload_file) + str(drive_link)
    if st.get("wq_op_source") != current_source:
        st["wq_op_master_df"] = None
        st["wq_op_source"] = current_source
        st["wq_op_source_name"] = source_name
    else:
        source_name = st.get("wq_op_source_name") or source_name

    try:
        if model_key in _USER_TRAINED_WQ_MODELS:
            model_path = _USER_TRAINED_WQ_MODELS[model_key]
        else:
            model_path = WQ_OPERATIONAL_MODELS.get(model_key)
        if not model_path or not os.path.exists(model_path):
            return error_html(f"Model file not found: {model_path}"), None, None, None, None, st

        if model_path in _WQ_OPERATIONAL_MODEL_CACHE:
            bundle = _WQ_OPERATIONAL_MODEL_CACHE[model_path]
        else:
            bundle = joblib.load(model_path)
            _WQ_OPERATIONAL_MODEL_CACHE[model_path] = bundle
        m_name = bundle.get("model_name", "")
        figs = {}

        # Initialize op master DataFrame once per dataset
        if st.get("wq_op_master_df") is None:
            st["wq_op_master_df"] = df_wq.copy()
        op_master_df = st["wq_op_master_df"]

        # --- MODEL 1: PCA + KMeans ---
        if "Model1" in m_name:
            all_temporal = ndci + ndti
            X_raw = df_result[all_temporal].fillna(df_result[all_temporal].mean())

            # Feature Engine
            df_result['NDCI_Mean'] = df_result[ndci].mean(axis=1)
            df_result['NDTI_Mean'] = df_result[ndti].mean(axis=1)
            df_result['NDCI_CV']   = df_result[ndci].std(axis=1) / (df_result['NDCI_Mean'].abs() + 1e-6)
            df_result['NDCI_STD']  = df_result[ndci].std(axis=1)
            df_result['TS_Amplitude'] = df_result[ndci].apply(lambda r: np.max(r.values) - np.min(r.values), axis=1)
            thr = bundle["hyperparameters"].get("bloom_threshold", 0.2)
            df_result['Bloom_Freq']   = df_result[ndci].apply(lambda r: int(np.sum(r.values > thr)), axis=1)
            df_result['Resilience_Score'] = 1.0 / (df_result['NDCI_STD'] + df_result[ndti].std(axis=1) + 1e-6)

            # Apply PCA
            scaler_pca = bundle["scaler_pca"]
            pca = bundle["pca"]
            X_pca = pca.transform(scaler_pca.transform(X_raw))

            # Apply Behavioral
            beh_feats = bundle["behavioral_features"]
            scaler_beh = bundle["scaler_beh"]
            X_beh = scaler_beh.transform(df_result[beh_feats])

            X_final = np.hstack((X_pca, X_beh))

            # Apply KMeans
            kmeans = bundle["kmeans"]
            df_result['Cluster'] = kmeans.predict(X_final)

            # Typology (dynamic rank assignment like train time)
            rank = df_result.groupby('Cluster')['NDCI_Mean'].mean().sort_values().index.tolist()
            eco_map = {}
            eco_names = [
                "WQ-I: Pristine (Oligotrophic)", "WQ-II: Moderate (Mesotrophic)",
                "WQ-III: Stressed (Eutrophic)", "WQ-IV: Critical (Hyper-Eutrophic)",
                "WQ-IV: Critical (Severely Degraded)"
            ]
            for i, idx in enumerate(rank):
                eco_map[idx] = eco_names[min(i, len(eco_names)-1)]
            df_result['Typology'] = df_result['Cluster'].map(eco_map)

            ts_score = df_result['TS_Amplitude'] + df_result['Bloom_Freq']
            df_result['TS_Class'] = pd.qcut(ts_score, q=3, labels=["TS-Stable: Low Variability", "TS-Seasonal: Moderate Bloom", "TS-Acute: High Bloom Frequency"], duplicates='drop')
            df_result['Resilience_Class'] = pd.qcut(df_result['Resilience_Score'], q=3, labels=["Resilience-Low: Vulnerable", "Resilience-Med: Moderate", "Resilience-High: Stable"], duplicates='drop')

            metrics = bundle.get("metrics", {})
            render_scores = {
                "final_k": bundle["hyperparameters"].get("K", 0),
                "final_sil": metrics.get("silhouette", 0),
                "final_ch": metrics.get("calinski_harabasz", 0),
                "pca_variance": [metrics.get("pca_variance_pct", 0) / 100.0]
            }
            summary = render_wq_summary_html(df_result, render_scores, "M1 Predict")

            # Generate Operational Figs (reusing Model 1 plotting logic)
            fig_b, ax_b = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=df_result, x='NDTI_Mean', y='NDCI_Mean', hue='Typology', size='NDCI_CV', sizes=(20,200), alpha=0.7, ax=ax_b)
            ax_b.set_title("Behavioral Space (Operational)")
            plt.tight_layout(); figs['behavioral'] = fig_b

            fig_hm, ax_hm = plt.subplots(figsize=(10,6))
            mean_series = df_result.groupby('Typology')[ndci].mean()
            x_vals = range(len(ndci))
            for typ in mean_series.index:
                ax_hm.plot(x_vals, mean_series.loc[typ].values, marker='o', lw=2, label=typ)
            ax_hm.set_title("Temporal Fingerprints (NDCI) - Operational")
            ax_hm.legend()
            plt.tight_layout(); figs['temporal'] = fig_hm

            fig_rc, ax_rc = plt.subplots(figsize=(6,4))
            sns.countplot(data=df_result, x='Resilience_Class', palette='viridis', ax=ax_rc)
            plt.tight_layout(); figs['resilience'] = fig_rc

            # Mappings for output
            f1, f2, f3, f4 = figs.get('behavioral'), figs.get('temporal'), figs.get('resilience'), None

        # --- MODEL 2: DTW Shape Classifier ---
        elif "Model2" in m_name:
            from .models.wq import dtw_distance
            protos = bundle["prototypes"]
            shape_map = bundle["shape_map"]

            labels = []
            for _, row in df_result[ndci].iterrows():
                ts = row.values
                if np.std(ts) > 1e-5: ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
                else: ts = np.zeros_like(ts)
                dists = [dtw_distance(ts, p) for p in protos]
                labels.append(np.argmin(dists))

            df_result['Shape_Regime'] = [shape_map[i] for i in labels]

            # Need a pseudo-typology to show anything meaningful in the summary card if M1 didn't run
            if 'Typology' not in df_result.columns:
                df_result['NDCI_Mean'] = df_result[ndci].mean(axis=1)
                df_result['NDTI_Mean'] = df_result[ndti].mean(axis=1)
                df_result['Typology'] = df_result.apply(lambda r: "WQ-IV: Critical (Algal + Turbid)" if (r['NDCI_Mean'] > 0.3 and r['NDTI_Mean'] > 0.1) else ("WQ-III: Stressed" if r['NDCI_Mean'] > 0.3 else ("WQ-II: Moderate" if r['NDTI_Mean'] > 0.1 else "WQ-I: Pristine")), axis=1)

            summary = render_wq_summary_html(df_result, model_name="M2 Predict")

            fig_heat, ax_heat = plt.subplots(figsize=(8,6))
            ct = pd.crosstab(df_result['Typology'], df_result['Shape_Regime'])
            sns.heatmap(ct, annot=True, cmap='Blues', fmt='d', ax=ax_heat)
            plt.tight_layout(); figs['dtw_counts'] = fig_heat

            fig_bar, ax_bar = plt.subplots(figsize=(10,6))
            ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax_bar)
            plt.tight_layout(); figs['dtw_bar'] = fig_bar

            f1, f2, f3, f4 = figs.get('dtw_counts'), figs.get('dtw_bar'), None, None

        # --- MODEL 3: Triple Stream ---
        elif "Model3" in m_name:
            if not wsa:
                return error_html("Model 3 requires WSA/Area data, but none was found."), None, None, None, None, st

            # Subset and align using ID if possible (avoids KeyError)
            _df_wsa = df_wsa.copy()
            _df_wq = df_result.copy()
            if 'ID' in _df_wsa.columns and 'ID' in _df_wq.columns:
                _df_wsa = _df_wsa.set_index('ID')
                _df_wq = _df_wq.set_index('ID')
                common_ids = _df_wsa.index.intersection(_df_wq.index)
            else:
                common_ids = list(range(min(len(_df_wsa), len(_df_wq))))
                _df_wsa.index = list(range(len(_df_wsa)))
                _df_wq.index = list(range(len(_df_wq)))

            wsa_raw = _df_wsa.loc[common_ids, wsa].fillna(0).values if len(wsa) else np.zeros((len(common_ids), 1))
            ndci_raw = _df_wq.loc[common_ids, ndci].fillna(0).values
            ndti_raw = _df_wq.loc[common_ids, ndti].fillna(0).values

            # Create a separate df for results to merge back properly by ID handling
            df_m3 = pd.DataFrame(index=common_ids)

            # Use separate scalers per stream (matching retrained .pkl bundles)
            scaler_ndci = bundle.get("scaler_ndci")
            scaler_ndti = bundle.get("scaler_ndti")
            scaler_wsa  = bundle.get("scaler_wsa")
            try:
                X_ndci = scaler_ndci.transform(ndci_raw) if scaler_ndci else StandardScaler().fit_transform(ndci_raw)
                X_ndti = scaler_ndti.transform(ndti_raw) if scaler_ndti else StandardScaler().fit_transform(ndti_raw)
                X_wsa  = scaler_wsa.transform(wsa_raw)   if scaler_wsa  else StandardScaler().fit_transform(wsa_raw)
            except Exception:
                X_ndci = StandardScaler().fit_transform(ndci_raw)
                X_ndti = StandardScaler().fit_transform(ndti_raw)
                X_wsa  = StandardScaler().fit_transform(wsa_raw)

            X_pca_ndci = bundle["pca_ndci"].transform(X_ndci)
            X_pca_ndti = bundle["pca_ndti"].transform(X_ndti)
            X_pca_wsa  = bundle["pca_wsa"].transform(X_wsa)

            df_m3['WSA_Mean']  = np.mean(wsa_raw, axis=1)
            df_m3['WSA_CV']    = np.std(wsa_raw, axis=1) / (df_m3['WSA_Mean'].abs() + 1e-6)
            df_m3['NDCI_Mean'] = np.mean(ndci_raw, axis=1)
            df_m3['NDCI_CV']   = np.std(ndci_raw, axis=1) / (df_m3['NDCI_Mean'].abs() + 1e-6)
            df_m3['NDTI_Mean'] = np.mean(ndti_raw, axis=1)
            df_m3['NDTI_CV']   = np.std(ndti_raw, axis=1) / (df_m3['NDTI_Mean'].abs() + 1e-6)

            beh_feats = bundle["behavioral_features"]
            X_beh = bundle["scaler_beh"].transform(df_m3[beh_feats])

            X_final = np.hstack((X_pca_ndci, X_pca_ndti, X_pca_wsa, X_beh))
            df_m3['Cluster'] = bundle["kmeans"].predict(X_final)

            df_m3['Typology'] = df_m3.apply(lambda r: "WQ-IV: Critical (Algal + Turbid)" if (r['NDCI_Mean'] > 0.3 and r['NDTI_Mean'] > 0.1) else ("WQ-III: Stressed (Algal Bloom)" if r['NDCI_Mean'] > 0.3 else ("WQ-II: Moderate (Turbid/Sediment)" if r['NDTI_Mean'] > 0.1 else "WQ-I: Pristine (Clear/Stable)")), axis=1)

            # Transfer newly calculated fields back to df_result for proper merge later
            if 'ID' in df_result.columns and df_m3.index.name == 'ID':
                for _c in ['Cluster', 'Typology', 'WSA_Mean', 'WSA_CV', 'NDCI_Mean', 'NDCI_CV', 'NDTI_Mean', 'NDTI_CV']:
                    df_result[_c] = df_result['ID'].map(df_m3[_c])
            else:
                for _c in ['Cluster', 'Typology', 'WSA_Mean', 'WSA_CV', 'NDCI_Mean', 'NDCI_CV', 'NDTI_Mean', 'NDTI_CV']:
                    df_result.loc[df_result.index[:len(df_m3)], _c] = df_m3[_c].values


            metrics = bundle.get("metrics", {})
            render_scores = {
                "final_k": bundle["hyperparameters"].get("K", 0),
                "final_sil": metrics.get("silhouette", 0),
                "final_ch": metrics.get("calinski_harabasz", 0),
                "pca_variance": [0] # Triple stream PCA var is complex, so hide or 0
            }
            summary = render_wq_summary_html(df_result, render_scores, "M3 Predict")

            fig_sc, ax_sc = plt.subplots(figsize=(8,6))
            sp = ax_sc.scatter(df_result['NDTI_Mean'], df_result['NDCI_Mean'], c=df_result['WSA_Mean'], cmap='viridis', s=df_result['WSA_CV']*100, alpha=0.7)
            plt.colorbar(sp, label='Area')
            ax_sc.set_xlabel('Turbidity (NDTI)'); ax_sc.set_ylabel('Algae (NDCI)')
            plt.tight_layout(); figs['scatter'] = fig_sc

            f1, f2, f3, f4 = figs.get('scatter'), None, None, None

        # Merge results into op_master_df with model prefixes, similar to analysis logic
        if "Model1" in m_name:
            for _col in ["Cluster", "Typology", "TS_Class", "Resilience_Class"]:
                if _col in df_result.columns:
                    op_master_df[f"M1_{_col}"] = df_result[_col].values
            # Default other columns if not present
            for _col in ["M2_Shape_Regime", "M3_Cluster", "M3_Typology"]:
                if _col not in op_master_df.columns:
                    op_master_df[_col] = "N/A"

        elif "Model2" in m_name:
            for _col in ["Shape_Regime"]:
                if _col in df_result.columns:
                    op_master_df[f"M2_{_col}"] = df_result[_col].values
            for _col in ["M1_Cluster", "M1_Typology", "M1_TS_Class", "M1_Resilience_Class", "M3_Cluster", "M3_Typology"]:
                if _col not in op_master_df.columns:
                    op_master_df[_col] = "N/A"

        elif "Model3" in m_name:
            # Need to match by ID because M3 may have subset of common_ids
            for _col in ["Cluster", "Typology"]:
                if _col in df_result.columns:
                    if 'ID' in op_master_df.columns and 'ID' in df_result.columns:
                        _id_to_val = df_result[_col].to_dict() if df_result.index.name == 'ID' else df_result.set_index('ID')[_col].to_dict()
                        op_master_df[f"M3_{_col}"] = op_master_df['ID'].map(_id_to_val)
                    else:
                        # Fallback for positional merge
                        op_master_df[f"M3_{_col}"] = df_result[_col].values[:len(op_master_df)]
            for _col in ["M1_Cluster", "M1_Typology", "M1_TS_Class", "M1_Resilience_Class", "M2_Shape_Regime"]:
                if _col not in op_master_df.columns:
                    op_master_df[_col] = "N/A"

        st["wq_op_master_df"] = op_master_df
        dl_path = save_result_csv(op_master_df, source_name, suffix="wq_predictions")
        return summary, f1, f2, f3, f4, dl_path, st

    except Exception as e:
        return error_html(f"Error during prediction: {str(e)}<br><pre>{traceback.format_exc()}</pre>"), None, None, None, None, None, st


# -----------------------------
# Operational prediction helpers
# -----------------------------
def _operational_pipeline_name(bundle, model_name):
    """Identify the exact preprocessing pipeline required by a model bundle."""
    if isinstance(bundle, dict):
        explicit = bundle.get("pipeline") or bundle.get("preprocessing_pipeline")
        if explicit:
            return str(explicit)
        if bundle.get("is_universal", False):
            return "universal"
        if "rejuv_detector" in bundle and "newf_detector" in bundle:
            return "three_stage"
        if "xgb_model" in bundle:
            return "single_xgb_fast"
        if "model" in bundle:
            return "custom_raw_ts"

    configured = OPERATIONAL_MODEL_METADATA.get(model_name, {}).get("pipeline")
    if configured:
        return configured
    if "USO_5" in str(model_name):
        return "shape_only_8"
    return "shape_only_13"


def _positive_int_or_none(value):
    try:
        value = int(value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _operational_training_length(bundle, model_name, pipeline):
    """Read the model's training T instead of forcing every model to T=76."""
    if pipeline in {"universal", "shape_only_13", "shape_only_8"}:
        return None

    if isinstance(bundle, dict):
        for key in ("T", "T_original", "time_series_length", "ts_length", "sequence_length"):
            value = _positive_int_or_none(bundle.get(key))
            if value is not None:
                return value

        # Raw-TS custom models can infer T from their stored input columns.
        if pipeline == "custom_raw_ts":
            names = bundle.get("featurenames") or bundle.get("selectedfeatures")
            if names:
                return len(names)

    configured = _positive_int_or_none(
        OPERATIONAL_MODEL_METADATA.get(model_name, {}).get("T")
    )
    if configured is not None:
        return configured

    raise ValueError(
        f"The model '{model_name}' does not store its training time-series length T. "
        "Retrain/deploy it with this corrected code, or add its exact T to "
        "OPERATIONAL_MODEL_METADATA."
    )


def _three_stage_groups_for_prediction(bundle):
    """Restore saved Three-Stage feature groups; support legacy bundles safely."""
    keys = (
        "basic", "water", "quarters", "trend", "rejuvenation", "vshape",
        "zero_patterns", "variability", "peaks", "advanced", "discriminative"
    )
    saved = None
    if isinstance(bundle, dict):
        saved = bundle.get("feature_groups")
        if saved is None and isinstance(bundle.get("params"), dict):
            saved = bundle["params"].get("feature_groups")

    if isinstance(saved, dict):
        return {key: bool(saved.get(key, False)) for key in keys}

    # Legacy Three-Stage bundles did not save the checkboxes. Computing all
    # groups is safe because the final matrix is aligned to saved featurenames.
    return {key: True for key in keys}


def _ordered_ts_columns(df):
    def key(name):
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else 10**9
    return sorted([c for c in df.columns if str(c).startswith("TS_")], key=key)


def _matrix_from_feature_dicts(feature_dicts, feature_names, pipeline):
    if not feature_dicts:
        return np.empty((0, len(feature_names or [])), dtype=float)

    if feature_names:
        missing = sorted({
            name for name in feature_names
            if any(name not in features for features in feature_dicts)
        })
        if missing:
            preview = ", ".join(missing[:12])
            raise ValueError(
                f"{pipeline} preprocessing cannot reproduce {len(missing)} saved training "
                f"feature(s): {preview}. The model bundle and code are from different "
                "feature-engineering versions."
            )
        matrix = [[features[name] for name in feature_names] for features in feature_dicts]
    else:
        # Legacy fallback: Python dict insertion order matches the training extractor.
        matrix = [list(features.values()) for features in feature_dicts]

    return np.asarray(matrix, dtype=float)


def _expected_feature_count(estimator):
    if estimator is None:
        return None
    value = getattr(estimator, "n_features_in_", None)
    if value is not None:
        return _positive_int_or_none(value)
    try:
        booster = estimator.get_booster()
        return _positive_int_or_none(booster.num_features())
    except Exception:
        return None


def _check_feature_count(X, expected, model_name, stage):
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch for {model_name} ({stage}): model expects {expected} "
            f"features, but Operational preprocessing produced {X.shape[1]}. "
            "Prediction was stopped to prevent misleading low accuracy."
        )


def _normalise_prediction_labels(values):
    return np.asarray([normalize_status(value) for value in np.asarray(values).reshape(-1)], dtype=object)


def _label_encoder_index(label_encoder, class_name):
    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        raise ValueError("Three-Stage bundle is missing its LabelEncoder.")
    normalized = [normalize_status(value) for value in label_encoder.classes_]
    wanted = normalize_status(class_name)
    if wanted not in normalized:
        raise ValueError(
            f"Three-Stage LabelEncoder does not contain required class '{wanted}'. "
            f"Available classes: {normalized}"
        )
    return int(normalized.index(wanted))


def _build_non_shape_operational_matrix(bundle, df, feature_cols, pipeline, model_name):
    feature_names = list(bundle.get("featurenames") or []) if isinstance(bundle, dict) else []

    if pipeline == "three_stage":
        groups = _three_stage_groups_for_prediction(bundle)
        dictionaries = [
            extract_discriminative_features(row, feature_cols, groups)
            for _, row in df.iterrows()
        ]
        X = _matrix_from_feature_dicts(dictionaries, feature_names, pipeline)
        expected = _expected_feature_count(bundle.get("scaler"))
        _check_feature_count(X, expected, model_name, "before Three-Stage scaler")
        return X

    if pipeline == "single_xgb_fast":
        dictionaries = [extractfeaturesfast(row, feature_cols) for _, row in df.iterrows()]
        X = _matrix_from_feature_dicts(dictionaries, feature_names, pipeline)
        expected = _expected_feature_count(bundle.get("xgb_model"))
        _check_feature_count(X, expected, model_name, "Single-XGBoost input")
        return X

    if pipeline == "universal":
        dictionaries = [extract_universal_features(row, feature_cols) for _, row in df.iterrows()]
        X = _matrix_from_feature_dicts(dictionaries, feature_names, pipeline)
        expected = _expected_feature_count(bundle.get("xgb_model"))
        _check_feature_count(X, expected, model_name, "Universal-XGBoost input")
        return X

    if pipeline == "custom_raw_ts":
        X = df[feature_cols].to_numpy(dtype=float)
        expected = _expected_feature_count(bundle.get("model"))
        if expected is None and feature_names:
            expected = len(feature_names)
        _check_feature_count(X, expected, model_name, "custom raw time-series input")
        return X

    raise ValueError(f"Unsupported Operational pipeline: {pipeline}")


# -----------------------------
# Operational Predict (Pond)
# -----------------------------
def operational_predict(model_selector, uploaded_file, drive_path, st, target_classes=None):
    """Run Operational prediction with the selected model's exact training pipeline.

    Class choices come only from the selected model. Labeled rows from classes
    the model was not trained on (or classes the user deselected) are excluded
    from evaluation. Unlabeled inputs retain every original row.
    """
    requested_classes = list(target_classes or [])
    try:
        # 1. Load model and optional separate LabelEncoder.
        model_path = _operational_model_path(model_selector)
        if not model_path:
            return "<div style='color:red;'>❌ Model path not found</div>", gr.update(visible=False), None, None, None, None, st

        bundle, error = load_pretrained_model(model_path)
        if error:
            return f"<div style='color:red;'>❌ {error}</div>", gr.update(visible=False), None, None, None, None, st

        separate_le = None
        le_path = OPERATIONAL_LABEL_ENCODERS.get(model_selector)
        if le_path:
            separate_le, le_error = load_pretrained_model(le_path)
            if le_error:
                return f"<div style='color:red;'>❌ Label encoder load error: {le_error}</div>", gr.update(visible=False), None, None, None, None, st

        model_classes = extract_trained_classes_from_bundle(bundle, separate_le)
        if not model_classes:
            return (
                "<div style='color:red;'>❌ Could not determine the classes learned by this model. "
                "Store its LabelEncoder in the model bundle or configure "
                "OPERATIONAL_LABEL_ENCODERS.</div>",
                gr.update(visible=False), None, None, None, None, st
            )

        normalized_requested = _normalise_model_class_values(requested_classes)
        selected_model_classes = [c for c in model_classes if c in normalized_requested]
        target_classes = selected_model_classes or model_classes

        pipeline = _operational_pipeline_name(bundle, model_selector)
        T_target = _operational_training_length(bundle, model_selector, pipeline)

        # 2. Load source data. Raw data are cached by source; processed data are
        # cached by source + model + pipeline + T, never globally at T=76.
        source_id = str(uploaded_file) if uploaded_file is not None else str(drive_path or "")
        if not source_id.strip():
            return "<div style='color:red;'>❌ No data source provided</div>", gr.update(visible=False), None, None, None, None, st

        if st.get("op_source_id") != source_id or st.get("op_original_df") is None:
            if uploaded_file is not None:
                orig_df = read_table_path(uploaded_file)
                source_name = (
                    orig_df.attrs.get("source_name")
                    or _safe_source_filename(upload_file=uploaded_file)
                )
            elif drive_path and str(drive_path).strip():
                active_drive_path = str(drive_path).strip()
                orig_df = read_table_path(active_drive_path)
                source_name = (
                    orig_df.attrs.get("source_name")
                    or _safe_source_filename(drive_path=active_drive_path)
                )
            else:
                return "<div style='color:red;'>❌ No data source provided</div>", gr.update(visible=False), None, None, None, None, st

            st["op_original_df"] = orig_df.copy()
            st["op_source_id"] = source_id
            st["op_source_name"] = source_name
            st["op_predictions"] = {}
            st["op_cache_key"] = None
        else:
            source_name = (
                st.get("op_source_name")
                or _safe_source_filename(
                    upload_file=uploaded_file,
                    drive_path=drive_path,
                    fallback="dataset.csv",
                )
            )

        cache_key = (source_id, model_selector, pipeline, T_target)
        if st.get("op_cache_key") != cache_key or st.get("op_work_df") is None:
            work_df, meta = preprocess_dataset_to_unified(
                st["op_original_df"],
                f"operational:{model_selector}",
                T=T_target,
                require_label=False,
            )
            st["op_work_df"] = work_df
            st["op_meta"] = meta
            st["op_cache_key"] = cache_key

        full_work_df = st["op_work_df"]
        meta = st["op_meta"]
        df = full_work_df.copy()

        # 3. Keep only labels within the selected model's trained class scope.
        has_real_label = bool(meta.get("is_real_label", False))
        ignored_untrained_rows = 0
        if has_real_label:
            before = len(df)
            df = df[df["label"].isin(target_classes)].copy()
            ignored_untrained_rows = before - len(df)
            if df.empty:
                return (
                    "<div style='color:red;'>❌ No labeled rows match the selected "
                    f"model-trained classes: {', '.join(target_classes)}.</div>",
                    gr.update(visible=False), None, None, None, None, st
                )

        feature_cols = _ordered_ts_columns(df)
        if len(feature_cols) < 5:
            raise ValueError("Operational preprocessing produced fewer than 5 temporal columns.")

        # 4. Reproduce the exact model-specific feature pipeline.
        predictions_raw = None
        probabilities = None
        le = separate_le

        if pipeline in {"shape_only_13", "shape_only_8"}:
            original = st["op_original_df"].copy()
            label_col_name = meta.get("orig_labelcol")

            if pipeline == "shape_only_8":
                exclude_set = {
                    "status", "finalstatus", "final_status", "label", "_dummy_label",
                    "class", "type", "category", "id", "s.no", "village id",
                    "village_id", "latitude", "longitude", "area", "perimeter",
                    "shape_leng", "shape_area"
                }
                if label_col_name:
                    exclude_set.add(str(label_col_name).lower().strip())
                for col in original.columns:
                    original[col] = pd.to_numeric(original[col], errors="ignore")
                raw_feature_cols = [
                    col for col in original.columns
                    if str(col).lower().strip() not in exclude_set
                    and pd.api.types.is_numeric_dtype(original[col])
                ]
            else:
                raw_feature_cols = list(meta.get("orig_series_cols", []))

            if len(raw_feature_cols) < 5:
                raise ValueError("Shape-only preprocessing found fewer than 5 usable temporal columns.")

            source_positions = np.asarray(meta.get("source_row_positions", range(len(full_work_df))), dtype=int)
            active_positions = source_positions[df.index.to_numpy(dtype=int)]
            feature_rows = []
            for position in active_positions:
                values = original.iloc[int(position)][raw_feature_cols].to_numpy(dtype=float)
                features = extract_shape_features_8(values) if pipeline == "shape_only_8" else extract_shape_only_features(values)
                feature_rows.append(list(features.values()))
            X_test = np.asarray(feature_rows, dtype=float)
            _check_feature_count(X_test, _expected_feature_count(bundle), model_selector, pipeline)

            estimator = bundle
            if hasattr(estimator, "predict_proba"):
                probabilities = estimator.predict_proba(X_test)
            encoded = estimator.predict(X_test)
            predictions_raw = le.inverse_transform(encoded) if le is not None else encoded

        else:
            if not isinstance(bundle, dict):
                raise ValueError(f"Pipeline '{pipeline}' requires a dictionary model bundle.")

            X_test = _build_non_shape_operational_matrix(
                bundle, df, feature_cols, pipeline, model_selector
            )
            le = bundle.get("labelencoder") or separate_le

            if pipeline in {"single_xgb_fast", "universal"}:
                estimator = bundle["xgb_model"]
                if hasattr(estimator, "predict_proba"):
                    probabilities = estimator.predict_proba(X_test)
                encoded = estimator.predict(X_test)
                predictions_raw = le.inverse_transform(encoded) if le is not None else encoded

            elif pipeline == "three_stage":
                scaler = bundle.get("scaler")
                selector = bundle.get("selector")
                if scaler is None or selector is None:
                    raise ValueError("Three-Stage bundle is missing its scaler or feature selector.")

                X_scaled = scaler.transform(X_test)
                X_selected = selector.transform(X_scaled)
                _check_feature_count(
                    X_selected,
                    _expected_feature_count(bundle.get("rejuv_detector")),
                    model_selector,
                    "after Three-Stage selector",
                )

                rprob = bundle["rejuv_detector"].predict_proba(X_selected)[:, 1]
                nprob = bundle["newf_detector"].predict_proba(X_selected)[:, 1]
                thresholds = bundle.get("thresholds") or {}
                r_th = float(thresholds.get("rejuv", thresholds.get("rejuvenated", 0.30)))
                n_th = float(thresholds.get("newf", thresholds.get("newly formed", 0.35)))
                idx_rejuv = _label_encoder_index(le, "rejuvenated")
                idx_newf = _label_encoder_index(le, "newly formed")

                encoded = []
                pe_model = bundle.get("persistextinct_clf")
                if pe_model is None:
                    raise ValueError("Three-Stage bundle is missing persistextinct_clf.")
                for i in range(len(X_selected)):
                    if rprob[i] >= r_th:
                        encoded.append(idx_rejuv)
                    elif nprob[i] >= n_th:
                        encoded.append(idx_newf)
                    else:
                        encoded.append(int(np.asarray(pe_model.predict(X_selected[i:i+1])).reshape(-1)[0]))
                predictions_raw = le.inverse_transform(np.asarray(encoded, dtype=int))

            elif pipeline == "custom_raw_ts":
                estimator = bundle["model"]
                scaler = bundle.get("scaler")
                X_model = scaler.transform(X_test) if scaler is not None else X_test
                if hasattr(estimator, "predict_proba"):
                    probabilities = estimator.predict_proba(X_model)
                encoded = estimator.predict(X_model)
                predictions_raw = le.inverse_transform(encoded) if le is not None else encoded

        if predictions_raw is None:
            raise ValueError("The selected model did not produce predictions.")

        predictions = _normalise_prediction_labels(predictions_raw)
        if len(predictions) != len(df):
            raise ValueError(
                f"Prediction row mismatch: produced {len(predictions)} predictions for {len(df)} rows."
            )

        # 5. Save predictions against exact original source-row positions.
        source_positions = np.asarray(meta.get("source_row_positions", range(len(full_work_df))), dtype=int)
        active_positions = source_positions[df.index.to_numpy(dtype=int)]
        col_clean = re.sub(r"[^a-zA-Z0-9]", "_", model_selector)
        prediction_column = col_clean + "_status"
        st["op_predictions"][prediction_column] = pd.Series(
            predictions,
            index=active_positions,
            dtype=object,
        )

        download_df = st["op_original_df"].copy()
        for column, series in st["op_predictions"].items():
            download_df[column] = pd.Series(dtype=object)
            valid_positions = [int(i) for i in series.index if 0 <= int(i) < len(download_df)]
            if valid_positions:
                download_df.iloc[valid_positions, download_df.columns.get_loc(column)] = series.loc[valid_positions].to_numpy()

        if has_real_label:
            download_df = download_df.iloc[active_positions].copy()

        csv_path = save_result_csv(download_df, source_name, suffix="predictions")

        # 6. Results and normalized evaluation metrics.
        results_html, _ = generate_operational_results(
            df, predictions, probabilities, bundle,
            target_classes=target_classes,
        )
        ignored_note = (
            f"<br><b>Ignored labeled rows from untrained/unselected classes:</b> {ignored_untrained_rows}"
            if has_real_label else
            "<br><b>Unlabeled input:</b> all rows were retained."
        )
        T_note = "original length" if T_target is None else str(T_target)
        class_scope_html = f"""
        <div style='margin-bottom:12px; padding:12px 14px; border-radius:10px;
                    background:#ecfdf5; border-left:4px solid #10b981;
                    color:#065f46; font-size:13px;'>
            <b>Pipeline:</b> {html.escape(pipeline)} &nbsp;|&nbsp;
            <b>Model T:</b> {html.escape(T_note)}<br>
            <b>Active model-trained classes:</b>
            {", ".join(html.escape(str(c)) for c in target_classes)}
            {ignored_note}<br>
            <b>Download file:</b> {html.escape(os.path.basename(csv_path))}
        </div>
        """
        results_html = class_scope_html + results_html

        metrics_df = None
        cm_plot_fig = None
        report_html = None
        if has_real_label:
            y_true = _normalise_prediction_labels(df["label"].to_numpy())
            accuracy = accuracy_score(y_true, predictions)
            class_report = classification_report(
                y_true, predictions, output_dict=True, zero_division=0
            )
            available_classes = sorted(set(y_true) | set(predictions))
            cm = confusion_matrix(y_true, predictions, labels=available_classes)

            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
                "Value": [
                    f"{accuracy:.4f}",
                    f"{class_report['weighted avg']['precision']:.4f}",
                    f"{class_report['weighted avg']['recall']:.4f}",
                    f"{class_report['weighted avg']['f1-score']:.4f}",
                ],
            })

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=available_classes, yticklabels=available_classes, ax=ax
            )
            ax.set_title("Confusion Matrix - Operational Mode")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            cm_plot_fig = fig
            plt.close(fig)

            acc_val = accuracy
            macro_f1 = class_report.get("macro avg", {}).get("f1-score", 0)
            weighted_f1 = class_report.get("weighted avg", {}).get("f1-score", 0)
            skip_keys = {"macro avg", "weighted avg", "accuracy"}
            class_rows = {
                key: value for key, value in class_report.items()
                if key not in skip_keys and isinstance(value, dict)
            }
            sorted_classes = sorted(class_rows.items(), key=lambda item: item[1].get("f1-score", 0))
            f1_scores = [value.get("f1-score", 0) for _, value in sorted_classes]
            min_f1 = min(f1_scores) if f1_scores else 0
            max_f1 = max(f1_scores) if f1_scores else 1

            rows_html = ""
            for cls, metrics in sorted_classes:
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)
                f1 = metrics.get("f1-score", 0)
                support = int(metrics.get("support", 0))
                bar_width = int(f1 * 100)
                row_bg = "#fff1f1" if f1 == min_f1 else ("#f0fff4" if f1 == max_f1 else "#ffffff")
                rows_html += f"""
                <tr style='background:{row_bg}; border-bottom:1px solid #e2e8f0;'>
                    <td style='padding:10px 12px; font-weight:600;'>{html.escape(str(cls))}</td>
                    <td style='padding:10px 12px; text-align:center;'>{precision:.2f}</td>
                    <td style='padding:10px 12px; text-align:center;'>{recall:.2f}</td>
                    <td style='padding:10px 12px; text-align:center;'>{f1:.2f}</td>
                    <td style='padding:10px 12px; text-align:center;'>{support}</td>
                    <td style='padding:10px 16px;'>
                        <div style='background:#e9d5ff; border-radius:20px; overflow:hidden; min-width:80px;'>
                            <div style='background:linear-gradient(90deg,#7c3aed,#a855f7); width:{bar_width}%; padding:4px 8px; border-radius:20px; color:#fff; font-size:0.8em; font-weight:700; text-align:right; white-space:nowrap;'>
                                {f1*100:.1f}%
                            </div>
                        </div>
                    </td>
                </tr>"""

            report_html = f"""
            <div style='background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; font-family:Inter,sans-serif;'>
                <h3 style='margin:0 0 18px 0; font-size:1.1em;'>📊 Classification Report</h3>
                <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px;'>
                    <div style='background:linear-gradient(135deg,#667eea,#764ba2); border-radius:12px; padding:18px; color:#fff; text-align:center;'>
                        <div style='font-size:0.7em; font-weight:700; letter-spacing:1px; opacity:0.85;'>ACCURACY</div>
                        <div style='font-size:2em; font-weight:800;'>{acc_val*100:.2f}%</div>
                    </div>
                    <div style='background:linear-gradient(135deg,#f093fb,#f5576c); border-radius:12px; padding:18px; color:#fff; text-align:center;'>
                        <div style='font-size:0.7em; font-weight:700; letter-spacing:1px; opacity:0.85;'>MACRO AVG F1</div>
                        <div style='font-size:2em; font-weight:800;'>{macro_f1*100:.2f}%</div>
                    </div>
                    <div style='background:linear-gradient(135deg,#4facfe,#00f2fe); border-radius:12px; padding:18px; color:#fff; text-align:center;'>
                        <div style='font-size:0.7em; font-weight:700; letter-spacing:1px; opacity:0.85;'>WEIGHTED AVG F1</div>
                        <div style='font-size:2em; font-weight:800;'>{weighted_f1*100:.2f}%</div>
                    </div>
                </div>
                <table style='width:100%; border-collapse:collapse; font-size:0.88em;'>
                    <thead><tr style='background:#f8fafc; border-bottom:2px solid #e2e8f0;'>
                        <th style='padding:10px 12px; text-align:left;'>CLASS</th>
                        <th style='padding:10px 12px;'>PRECISION</th>
                        <th style='padding:10px 12px;'>RECALL</th>
                        <th style='padding:10px 12px;'>F1-SCORE</th>
                        <th style='padding:10px 12px;'>SUPPORT</th>
                        <th style='padding:10px 16px;'>F1 VISUALIZATION</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>"""
        else:
            report_html = (
                "<div style='padding:20px; text-align:center; color:#64748b;'>"
                "Validation label column not found. Metrics and plots are hidden. "
                "Only predictions are generated.</div>"
            )

        return (
            results_html, gr.update(visible=True), metrics_df, cm_plot_fig,
            report_html, csv_path, st
        )

    except Exception as exc:
        return (
            f"<div style='color:red;'>❌ Prediction error: {html.escape(str(exc))}"
            f"<br><pre>{html.escape(traceback.format_exc())}</pre></div>",
            gr.update(visible=False), None, None, None, None, st
        )


def generate_operational_results(df, predictions, probabilities, model=None, target_classes=None):
    """Render a compact, colorful Operational prediction overview."""
    res_df = df.copy()
    res_df["Predicted_Status"] = predictions

    if target_classes is None or len(target_classes) == 0:
        target_classes = DEFAULT_TARGET_CLASSES

    all_predicted = list(pd.Series(predictions).unique())
    display_classes = list(dict.fromkeys(
        list(target_classes) + [c for c in all_predicted if c not in target_classes]
    ))
    counts = pd.Series(predictions).value_counts().reindex(display_classes, fill_value=0)
    total = int(counts.sum())

    palette = [
        ("#6366f1", "#8b5cf6"),
        ("#06b6d4", "#3b82f6"),
        ("#10b981", "#14b8a6"),
        ("#f59e0b", "#f97316"),
        ("#ec4899", "#f43f5e"),
        ("#8b5cf6", "#d946ef"),
    ]

    cards = []
    for index, cls in enumerate(display_classes):
        count = int(counts[cls])
        pct = (count / total * 100.0) if total else 0.0
        c1, c2 = palette[index % len(palette)]
        cards.append(
            f"""
            <div class='prediction-class-card'
                 style='--card-a:{c1};--card-b:{c2};'>
                <div class='prediction-class-top'>
                    <span class='prediction-class-name'>{html.escape(str(cls).title())}</span>
                    <span class='prediction-class-percent'>{pct:.1f}%</span>
                </div>
                <div class='prediction-class-count'>{count:,}</div>
                <div class='prediction-class-caption'>predicted rows</div>
                <div class='prediction-progress'>
                    <span style='width:{max(2.0, pct):.2f}%'></span>
                </div>
            </div>
            """
        )

    summary_html = f"""
    <div class='prediction-overview'>
        <div class='prediction-overview-head'>
            <div>
                <span class='eyebrow'>PREDICTION OVERVIEW</span>
                <div class='results-title'>Status distribution</div>
            </div>
            <div class='total-badge'><b>{total:,}</b><span>rows</span></div>
        </div>
        <div class='prediction-card-grid'>{''.join(cards)}</div>
    </div>
    """
    return summary_html, res_df


# -----------------------------
# UI toggle helpers
# -----------------------------
def toggle_model_panels(mk):
    is_three = (mk is not None) and str(mk).startswith("Three-Stage")
    is_single = (mk is not None) and str(mk).startswith("Single XGBoost")
    is_universal = (mk is not None) and str(mk).startswith("Generalized Universal")
    is_custom = (mk is not None) and str(mk).startswith("Custom Models")
    return (
        gr.update(visible=is_single),     # single_xgb_block
        gr.update(visible=is_single),     # xgb_hypers_block
        gr.update(visible=is_universal),  # universal_hypers_block
        gr.update(visible=is_three),      # three_feat_block
        gr.update(visible=is_custom),     # custom_models_block
        gr.update(visible=(mk is not None)),  # training_common_block
        gr.update(visible=is_three),      # three_params_block
    )


def toggle_custom_template_panels(tname):
    tname = str(tname)
    return (
        gr.update(visible=(tname == "Logistic Regression")), # custom_lr_block
        gr.update(visible=(tname == "SVC")),                 # custom_svc_block
        gr.update(visible=(tname == "KNN")),                 # custom_knn_block
        gr.update(visible=(tname == "Random Forest")),       # custom_rf_block
        gr.update(visible=(tname == "XGBoost (simple)")),    # custom_cxgb_block
    )


# -----------------------------
# Refresh helpers
# -----------------------------
def _notify_refresh(message):
    """Show a concise Gradio toast without adding permanent UI clutter."""
    try:
        gr.Info(message)
    except Exception:
        print(message)


def refresh_operational_models_ui(current_model=None):
    """Rescan Pond Drive folder and refresh dropdown, details, and classes."""
    logs = rescan_pond_model_registry()
    for line in logs:
        print(f"[Pond model scan] {line}")

    choices = _pond_model_choices()
    selected = current_model if current_model in choices else (choices[0] if choices else None)

    if selected:
        description, class_update = update_operational_model_ui(selected)
    else:
        description = "<div class='empty-state'>No compatible Pond models found.</div>"
        class_update = gr.update(choices=[], value=[], interactive=False)

    _notify_refresh(f"Pond models refreshed — {len(choices)} available")
    return gr.update(choices=choices, value=selected), description, class_update


def refresh_wq_operational_models_ui(current_model=None):
    """Rescan WQ Drive folder and refresh dropdown and description."""
    logs = rescan_wq_model_registry()
    for line in logs:
        print(f"[WQ model scan] {line}")

    choices = _wq_model_choices()
    selected = current_model if current_model in choices else (choices[0] if choices else None)
    description = get_wq_model_details(selected) if selected else (
        "<div class='empty-state'>No compatible Water Quality models found.</div>"
    )

    _notify_refresh(f"Water Quality models refreshed — {len(choices)} available")
    return gr.update(choices=choices, value=selected), description


def _dataset_refresh_status(label, count, logs):
    """Compact status chip for dataset refresh actions."""
    errors = [
        line for line in str(logs or "").splitlines()
        if "error" in line.lower() or "not found" in line.lower()
    ]
    if errors:
        return (
            "<div class='refresh-chip refresh-chip-warning'>"
            f"⚠️ {html.escape(label)} refreshed with {len(errors)} warning(s) · "
            f"{count} available"
            "</div>"
        )

    return (
        "<div class='refresh-chip'>"
        f"✓ {html.escape(label)} refreshed · {count} available · "
        f"{time.strftime('%H:%M:%S')}"
        "</div>"
    )


def _pond_dataset_selection_updates(state, ds_names):
    """Return class and summary updates for the selected Pond datasets."""
    if not ds_names:
        return (
            gr.update(choices=[], value=[]),
            "<div class='empty-state'>No datasets selected.</div>",
        )

    names_list = [ds_names] if isinstance(ds_names, str) else list(ds_names)
    all_classes = set()
    selected_datasets = {}

    for name in names_list:
        if name in state.get("datasets", {}):
            selected_datasets[name] = state["datasets"][name]
            df = state["datasets"][name]["df"]
            if "label" in df.columns:
                all_classes.update(df["label"].dropna().unique())

    classes = sorted(all_classes) or list(ALL_KNOWN_CLASSES)
    summary_html = (
        render_dataset_cards(selected_datasets)
        if selected_datasets
        else "<div class='empty-state'>No datasets selected.</div>"
    )
    return gr.update(choices=classes, value=classes), summary_html


def refresh_pond_datasets_ui(state, current_selection=None):
    """Reload Pond datasets from Drive into the current runtime."""
    state = state or new_state()
    fresh, logs = autoload_and_preprocess_all()

    # Keep datasets uploaded during this runtime while replacing Drive-loaded data.
    session_uploads = {
        name: value
        for name, value in state.get("datasets", {}).items()
        if str(name).startswith("Uploaded:")
    }
    state["datasets"] = {**fresh, **session_uploads}

    choices = list(state["datasets"].keys())
    requested = (
        [current_selection]
        if isinstance(current_selection, str)
        else list(current_selection or [])
    )
    selected = [name for name in requested if name in choices]
    if not selected and choices:
        selected = [choices[0]]

    class_update, summary_html = _pond_dataset_selection_updates(state, selected)
    status = _dataset_refresh_status("Pond datasets", len(choices), logs)
    _notify_refresh(f"Pond datasets refreshed — {len(choices)} available")

    return (
        gr.update(choices=choices, value=selected),
        class_update,
        summary_html,
        state,
        status,
    )


def _activate_wq_dataset(state, dataset_name):
    """Activate one WQ dataset in state and return its summary."""
    if not dataset_name or dataset_name not in state.get("wq_datasets", {}):
        state["wq_df_wq"] = None
        state["wq_df_wsa"] = None
        state["wq_ndci_cols"] = []
        state["wq_ndti_cols"] = []
        state["wq_wsa_cols"] = []
        state["wq_model1_done"] = False
        state["wq_result_df"] = None
        state["wq_master_df"] = None
        state["wq_active_source_name"] = None
        return "<div class='empty-state'>No Water Quality dataset selected.</div>"

    ds = state["wq_datasets"][dataset_name]
    state["wq_df_wq"] = ds["df_wq"]
    state["wq_df_wsa"] = ds["df_wsa"]
    state["wq_ndci_cols"] = ds["ndci_cols"]
    state["wq_ndti_cols"] = ds["ndti_cols"]
    state["wq_wsa_cols"] = ds["wsa_cols"]
    state["wq_model1_done"] = False
    state["wq_result_df"] = None
    state["wq_master_df"] = None
    state["wq_active_source_name"] = os.path.basename(str(ds.get("path") or dataset_name))
    return render_wq_dataset_cards({dataset_name: ds})


def refresh_wq_datasets_ui(state, current_selection=None):
    """Reload Water Quality datasets from Drive into the current runtime."""
    state = state or new_state()
    fresh, logs = autoload_wq_datasets()

    session_uploads = {
        name: value
        for name, value in state.get("wq_datasets", {}).items()
        if str(name).startswith("Uploaded:")
    }
    state["wq_datasets"] = {**fresh, **session_uploads}

    choices = list(state["wq_datasets"].keys())
    selected = current_selection if current_selection in choices else (
        choices[0] if choices else None
    )
    summary_html = _activate_wq_dataset(state, selected)
    status = _dataset_refresh_status("Water Quality datasets", len(choices), logs)
    _notify_refresh(f"Water Quality datasets refreshed — {len(choices)} available")

    return (
        gr.update(choices=choices, value=selected),
        summary_html,
        state,
        status,
    )
