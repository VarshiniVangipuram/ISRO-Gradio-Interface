"""
state.py
========
Application state initialization, experiment history management,
and undo/restore functionality.
"""

import os
import json
import traceback

import pandas as pd
import gradio as gr


def new_state():
    """Create a fresh application state dictionary."""
    return {
        "datasets": {},
        "history": [],
        "best": {"acc": -1, "bundle": None},
        "op_original_df": None,
        "op_source_id": None,
        "op_source_name": None,
        "op_cache_key": None,
        "op_work_df": None,
        "op_meta": None,
        "op_predictions": {},
        # Water Quality state
        "wq_datasets": {},
        "wq_df_wq": None, "wq_df_wsa": None, "wq_result_df": None,
        "wq_ndci_cols": [], "wq_ndti_cols": [], "wq_wsa_cols": [],
        "wq_scores": None, "wq_model1_done": False,
        "wq_active_source_name": None,
        "wq_op_source_name": None,
    }


def wq_new_state():
    """Create a fresh Water Quality state dictionary."""
    return {
        "wq_df_wq": None,
        "wq_df_wsa": None,
        "wq_result_df": None,
        "wq_ndci_cols": [],
        "wq_ndti_cols": [],
        "wq_wsa_cols": [],
        "wq_scores": None,
        "wq_model1_done": False,
        "wq_master_df": None,
        "wq_op_master_df": None,
        "wq_op_source": None,
        "wq_active_source_name": None,
        "wq_op_source_name": None,
    }


def history_df(st):
    """Convert experiment history into a DataFrame for display."""
    rows = []
    for h in st["history"]:
        rows.append({
            "experiment_id": h["experiment_id"],
            "timestamp": h["timestamp"],
            "dataset": h["dataset"],
            "model_kind": h["model_kind"],
            "accuracy": round(h["acc"]*100, 2),
            "f1_score": round(h["macro_f1"]*100, 2),
            "features_selected": h.get("features_selected", 0),
            "thresholds": json.dumps(h.get("thresholds", {})),
            "params": json.dumps(h.get("params", {})),
        })
    return pd.DataFrame(rows)


def recompute_best(history):
    """Find the best model in the experiment history."""
    best_acc = -1
    best_bundle = None
    for h in history:
        if h.get("acc", -1) > best_acc and h.get("bundle") is not None:
            best_acc = h["acc"]
            best_bundle = h["bundle"]
    return {"acc": best_acc, "bundle": best_bundle}


def _default_hyperparams():
    """Return the default hyperparameter values for undo/clear/error cases."""
    return (
        # Single XGBoost defaults
        False, 0.6, 3.0, False,
        2500, 0.03, 8, 1, 0.0, 0.9, 0.9, 2.0, 0.5, 50,
        # Generalized Universal defaults
        True, 0.6, 3.0,
        1000, 0.05, 6, 1.0, 0.0, 0.9, 0.9, 1.0, 0.0, 50,
        # Feature groups
        True, True, True, True, True, True, True, True, True, True, True,
        # Common params
        0.2, 42,
        # Three-Stage params
        True, 3, True, "mean",
        400, 10, 0.05, 15.0,
        400, 10, 0.05, 12.0,
        400, 8, 0.05,
        0.30, 0.35,
        # Custom Models params
        "Logistic Regression", True, True,
        1.0, 200,
        1.0, "rbf", "scale",
        5,
        300, 0,
        800, 0.05, 6,
    )


def undo_to_experiment(st, exp_id):
    """
    Undo/revert to a specific experiment ID.
    Restores all hyperparameters, visualizations, and results.
    """
    # Deferred imports to avoid circular dependencies
    from .ui.render import (
        render_best_model_status,
        render_performance_dashboard,
        render_classification_report,
    )
    from .ui.plots import (
        plot_confusion,
        plot_feature_importance_rf,
        plot_actual_vs_pred,
    )

    try:
        st = st or new_state()
        exp_id = int(exp_id)

        if exp_id < 0 or exp_id > len(st["history"]):
            err_msg = f"Experiment id must be 0..{len(st['history'])}"
            return (st, "", None, None, None, err_msg, history_df(st), err_msg,
                    gr.update(choices=[]), err_msg, *_default_hyperparams())

        # Truncate history
        st["history"] = st["history"][:exp_id]
        st["best"] = recompute_best(st["history"])

        # If clearing all (exp_id == 0)
        if exp_id == 0:
            msg = "Cleared all experiments."
            choices = [0]
            best_html = render_best_model_status(st["history"])
            return (st, best_html, None, None, None, "", history_df(st), msg,
                    gr.update(choices=choices, value=0), "", *_default_hyperparams())

        # Get the last experiment in truncated history
        last_exp = st["history"][-1]
        params = last_exp.get("params", {})
        result = last_exp.get("result", {})
        model_kind = last_exp.get("model_kind", "")

        # Regenerate visualizations (with safety checks)
        best_html = render_best_model_status(st["history"])

        cm_fig = None
        if result.get("cm") is not None and result.get("classes"):
            try:
                cm_fig = plot_confusion(result["cm"], result["classes"], last_exp["acc"])
            except:
                pass

        avp_fig = None
        if result.get("actual_counts") is not None and result.get("pred_counts") is not None:
            try:
                avp_fig = plot_actual_vs_pred(result["actual_counts"], result["pred_counts"], last_exp["acc"])
            except:
                pass

        fi_fig = None
        if model_kind.startswith("Three-Stage") and result.get("artifacts"):
            try:
                rfselector = result["artifacts"].get("rfselector")
                featurenames = result["artifacts"].get("featurenames", [])
                if rfselector and featurenames:
                    fi_fig = plot_feature_importance_rf(rfselector, featurenames, topk=20)
            except:
                pass

        # Generate summary
        template_name = None
        if model_kind.startswith("Custom Models"):
            template_name = params.get("custom_template_name", "")

        summary = ""
        try:
            summary = render_performance_dashboard(
                exp_id=last_exp.get("experiment_id", exp_id),
                model_kind=model_kind,
                dataset_name=last_exp.get("dataset", ""),
                acc=last_exp.get("acc", 0),
                macro_f1=last_exp.get("macro_f1", 0),
                rows=None,
                ts_length=None,
                thresholds=last_exp.get("thresholds"),
                template_name=template_name,
                features_selected=last_exp.get("features_selected")
            )
        except:
            summary = f"Reverted to Experiment #{exp_id}"

        # Generate classification report
        report_out = ""
        if result.get("report") and result.get("classes"):
            try:
                report_out = render_classification_report(result["report"], result["classes"])
            except:
                pass

        msg = f"Reverted to Experiment #{exp_id}."
        choices = list(range(0, len(st["history"]) + 1))

        # Extract hyperparameters from params (with defaults)
        # Single XGBoost params
        use_tempered_weights = params.get("use_tempered_weights", False)
        alpha = params.get("alpha", 0.6)
        cap = params.get("cap", 3.0)
        use_gpu = params.get("use_gpu", False)
        x_n_estimators = params.get("x_n_estimators", 2500)
        x_lr = params.get("x_lr", 0.03)
        x_max_depth = params.get("x_max_depth", 8)
        x_min_child_weight = params.get("x_min_child_weight", 1)
        x_gamma = params.get("x_gamma", 0.0)
        x_subsample = params.get("x_subsample", 0.9)
        x_colsample = params.get("x_colsample", 0.9)
        x_reg_lambda = params.get("x_reg_lambda", 2.0)
        x_reg_alpha = params.get("x_reg_alpha", 0.5)
        x_early_stop = params.get("x_early_stop", 50)

        # Generalized Universal (T-agnostic) params
        u_use_tempered_weights = params.get("u_use_tempered_weights", True)
        u_alpha = params.get("u_alpha", 0.6)
        u_cap = params.get("u_cap", 3.0)
        u_n_estimators = params.get("u_n_estimators", 1000)
        u_lr = params.get("u_lr", 0.05)
        u_max_depth = params.get("u_max_depth", 6)
        u_min_child_weight = params.get("u_min_child_weight", 1.0)
        u_gamma = params.get("u_gamma", 0.0)
        u_subsample = params.get("u_subsample", 0.9)
        u_colsample = params.get("u_colsample", 0.9)
        u_reg_lambda = params.get("u_reg_lambda", 1.0)
        u_reg_alpha = params.get("u_reg_alpha", 0.0)
        u_early_stop = params.get("u_early_stop", 50)

        # Feature groups
        fg = params.get("feature_groups", {})
        if not isinstance(fg, dict):
            fg = {}
        g_basic = fg.get("basic", True)
        g_water = fg.get("water", True)
        g_quarters = fg.get("quarters", True)
        g_trend = fg.get("trend", True)
        g_rejuv = fg.get("rejuvenation", fg.get("rejuv", True))
        g_vshape = fg.get("vshape", True)
        g_zero = fg.get("zero_patterns", fg.get("zero", True))
        g_var = fg.get("variability", fg.get("var", True))
        g_peaks = fg.get("peaks", True)
        g_adv = fg.get("advanced", fg.get("adv", True))
        g_disc = fg.get("discriminative", fg.get("disc", True))

        # Common params
        test_size = params.get("test_size", 0.2)
        random_state = params.get("random_state", 42)

        # Three-Stage params
        use_adasyn = params.get("use_adasyn", True)
        adasyn_neighbors = params.get("adasyn_neighbors", 3)
        auto_opt = params.get("auto_optimize_thresholds", True)
        fs_threshold = params.get("fs_threshold", "mean")

        s1 = params.get("s1", {})
        if not isinstance(s1, dict):
            s1 = {}
        s1_n = s1.get("n", 400)
        s1_depth = s1.get("depth", 10)
        s1_lr = s1.get("lr", 0.05)
        s1_spw = s1.get("spw", 15.0)

        s2 = params.get("s2", {})
        if not isinstance(s2, dict):
            s2 = {}
        s2_n = s2.get("n", 400)
        s2_depth = s2.get("depth", 10)
        s2_lr = s2.get("lr", 0.05)
        s2_spw = s2.get("spw", 12.0)

        s3 = params.get("s3", {})
        if not isinstance(s3, dict):
            s3 = {}
        s3_iter = s3.get("iter", 400)
        s3_depth = s3.get("depth", 8)
        s3_lr = s3.get("lr", 0.05)

        init_th = params.get("init_thresholds", {})
        if not isinstance(init_th, dict):
            init_th = {}
        init_rth = init_th.get("rejuv", 0.30)
        init_nth = init_th.get("newf", 0.35)

        # Custom Models params
        custom_template_name = params.get("custom_template_name", "Logistic Regression")
        custom_use_scaler = params.get("custom_use_scaler", True)
        custom_use_class_weight = params.get("custom_use_class_weight", True)
        custom_lr_C = params.get("custom_lr_C", 1.0)
        custom_lr_max_iter = params.get("custom_lr_max_iter", 200)
        custom_svc_C = params.get("custom_svc_C", 1.0)
        custom_svc_kernel = params.get("custom_svc_kernel", "rbf")
        custom_svc_gamma = params.get("custom_svc_gamma", "scale")
        custom_knn_k = params.get("custom_knn_k", 5)
        custom_rf_n = params.get("custom_rf_n", 300)
        custom_rf_depth = params.get("custom_rf_depth", 0)
        custom_cxgb_n = params.get("custom_cxgb_n", 800)
        custom_cxgb_lr = params.get("custom_cxgb_lr", 0.05)
        custom_cxgb_depth = params.get("custom_cxgb_depth", 6)

        return (st, best_html, cm_fig, fi_fig, avp_fig, report_out, history_df(st), summary,
                gr.update(choices=choices, value=len(st["history"])), "",
                # All hyperparameters
                use_tempered_weights, alpha, cap, use_gpu,
                x_n_estimators, x_lr, x_max_depth, x_min_child_weight, x_gamma, x_subsample, x_colsample, x_reg_lambda, x_reg_alpha, x_early_stop,
                u_use_tempered_weights, u_alpha, u_cap,
                u_n_estimators, u_lr, u_max_depth, u_min_child_weight, u_gamma,
                u_subsample, u_colsample, u_reg_lambda, u_reg_alpha, u_early_stop,
                g_basic, g_water, g_quarters, g_trend, g_rejuv, g_vshape, g_zero, g_var, g_peaks, g_adv, g_disc,
                test_size, random_state,
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
                custom_cxgb_n, custom_cxgb_lr, custom_cxgb_depth)

    except Exception as e:
        import traceback
        err_msg = f"Error during undo: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        return (st, "", None, None, None, err_msg, history_df(st), err_msg,
                gr.update(choices=[]), err_msg, *_default_hyperparams())
