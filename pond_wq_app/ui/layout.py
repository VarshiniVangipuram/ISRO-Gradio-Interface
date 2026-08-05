"""
Gradio Blocks layout builder.
Constructs the full UI and wires all event handlers.
Called by app.py after boot-time data loading.
"""

import gradio as gr

from ..config import (
    DATASET_REQUIREMENTS_MD,
    WQ_SHEET_WQ,
    WQ_SHEET_WSA,
    OPERATIONAL_MODELS,
    WQ_OPERATIONAL_MODELS,
    ALL_KNOWN_CLASSES,
)
from ..state import new_state, history_df, undo_to_experiment
from ..data.validation import validate_uploaded_dataset, validate_operational_dataset, wq_validate_data
from ..registry import (
    deploy_experiment_to_operational,
    save_deployed_pond_model_to_drive,
    deploy_wq_model_to_operational,
    save_deployed_wq_model_to_drive,
    get_operational_model_trained_classes,
)
from ..controllers import (
    run_experiment,
    operational_predict,
    wq_run_analysis,
    wq_operational_predict,
    toggle_model_panels,
    toggle_custom_template_panels,
    download_best_model,
    init_state,
    refresh_operational_models_ui,
    refresh_wq_operational_models_ui,
    refresh_pond_datasets_ui,
    refresh_wq_datasets_ui,
    _pond_dataset_selection_updates,
    _activate_wq_dataset,
)
from ..ui.render import (
    render_best_model_status,
    render_dataset_cards,
    render_wq_dataset_cards,
    get_wq_model_details,
    update_operational_model_ui,
)
from ..ui.css import CSS
from ..ui.js import DROPDOWN_JS


def build_demo(boot):
    """Build and return the Gradio Blocks demo.

    Parameters
    ----------
    boot : dict
        Boot-time computed values:
        - preloaded_datasets, dataset_names, folder_display, initial_classes
        - wq_preloaded_datasets, wq_dataset_names, wq_folder_display, wq_preloaded
        - initial_op_model, initial_op_classes, initial_op_class_error
        - initial_wq_model
    """
    _preloaded_datasets = boot["preloaded_datasets"]
    _dataset_names = boot["dataset_names"]
    _folder_display = boot["folder_display"]
    _initial_classes = boot["initial_classes"]
    _wq_preloaded_datasets = boot["wq_preloaded_datasets"]
    _wq_dataset_names = boot["wq_dataset_names"]
    _wq_folder_display = boot["wq_folder_display"]
    _initial_op_model = boot["initial_op_model"]
    _initial_op_classes = boot["initial_op_classes"]
    _initial_wq_model = boot["initial_wq_model"]

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

    with gr.Blocks(theme=theme, css=CSS, js=DROPDOWN_JS, title="Multi-Temporal Surface Water Quality & Classification Framework") as demo:
        st = gr.State(init_state())
        pond_deployment_state = gr.State(None)
        wq_deployment_state = gr.State(None)

        gr.Markdown(
            "<div id='title'>"
            "<h1>🌍 Multi-Temporal Surface Water Quality & Classification Framework</h1>"
            "<p>Earth observation analytics · pond classification · water-quality intelligence</p>"
            "</div>"
        )

        with gr.Tabs():
          # ============= OUTER TAB 1: Pond Classification =============
          with gr.TabItem("🌊 Pond Classification Analysis"):
            with gr.Tabs():
             # Tab 1: Advanced — modern training workspace
             with gr.TabItem("Advanced"):
                with gr.Column(elem_id="advanced-workspace"):
                    gr.HTML("""
                    <div class='workspace-hero advanced-hero'>
                        <div class='workspace-hero-icon advanced-hero-icon'>◆</div>
                        <div>
                            <div class='workspace-kicker'>MODEL DEVELOPMENT</div>
                            <div class='workspace-title'>Advanced Training Workspace</div>
                            <div class='workspace-copy'>
                                Select data, define classes, configure a model, and compare experiments in one guided workspace.
                            </div>
                        </div>
                        <div class='workspace-badge'>TRAIN · COMPARE · DEPLOY</div>
                    </div>
                    """)

                    # ---------------------------------------------------------
                    # Step cards: data, classes, and model architecture
                    # ---------------------------------------------------------
                    with gr.Row(equal_height=False, elem_classes=["advanced-top-grid"]):
                        with gr.Column(scale=5, elem_classes=["ui-card", "accent-cyan", "advanced-step-card"]):
                            gr.HTML("""
                            <div class='card-heading'>
                                <span class='advanced-card-index'>1</span>
                                <div><span class='card-step'>STEP 01</span><b>Select training data</b></div>
                            </div>
                            <p class='card-helper'>Choose one or more preloaded datasets, or add a custom dataset for this session.</p>
                            """)
                            dataset_dd = gr.Dropdown(
                                label=f"Training dataset(s) · {_folder_display}",
                                choices=_dataset_names,
                                value=[_dataset_names[0]] if _dataset_names else [],
                                multiselect=True,
                                interactive=True,
                            )
                            pond_refresh_datasets_btn = gr.Button(
                                "↻ Refresh datasets",
                                variant="secondary",
                                size="sm",
                                elem_classes=["soft-button", "compact-action"],
                            )
                            pond_dataset_refresh_status = gr.HTML("")

                            with gr.Accordion("Dataset summary", open=False, elem_classes=["soft-accordion"]):
                                initial_summary_data = {_dataset_names[0]: _preloaded_datasets[_dataset_names[0]]} if (_preloaded_datasets and _dataset_names) else {}
                                dataset_summary_html = gr.HTML(
                                    render_dataset_cards(initial_summary_data) if initial_summary_data
                                    else "<div class='advanced-empty-state'><span class='advanced-empty-mark'>DATA</span><div><b>No preloaded datasets</b><small>Add a custom dataset below to begin.</small></div></div>"
                                )

                            with gr.Accordion("Add a custom dataset", open=False, elem_classes=["soft-accordion"]):
                                with gr.Accordion("Dataset requirements", open=False, elem_classes=["soft-accordion"]):
                                    gr.Markdown(DATASET_REQUIREMENTS_MD)
                                with gr.Tabs(elem_classes=["compact-tabs"]):
                                    with gr.Tab("Local file"):
                                        upload_file = gr.File(
                                            label="CSV or Excel",
                                            file_types=[".csv", ".xlsx", ".xls"],
                                            type="filepath",
                                        )
                                    with gr.Tab("Google Drive"):
                                        drive_path = gr.Textbox(
                                            label="Drive link or mounted path",
                                            placeholder="https://drive.google.com/... or /content/drive/...",
                                            info="Use a public sharing link or a mounted Google Drive path.",
                                            lines=1,
                                        )
                                validate_btn = gr.Button(
                                    "Validate dataset",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["soft-button"],
                                )
                                validation_status = gr.HTML(
                                    "<div class='mini-empty'><span class='mini-empty-icon'>✓</span><div><b>Custom dataset not loaded</b><small>Select a file or Drive path, then validate it.</small></div></div>"
                                )

                        with gr.Column(scale=3, elem_classes=["ui-card", "accent-rose", "advanced-step-card"]):
                            gr.HTML("""
                            <div class='card-heading'>
                                <span class='advanced-card-index'>2</span>
                                <div><span class='card-step'>STEP 02</span><b>Choose target classes</b></div>
                            </div>
                            <p class='card-helper'>Training rows are filtered to the selected classes. Three-Stage requires rejuvenated and newly formed.</p>
                            """)
                            target_classes_cb = gr.CheckboxGroup(
                                choices=_initial_classes,
                                value=_initial_classes,
                                label="Classes included in training",
                                interactive=True,
                                elem_classes=["class-selector", "advanced-class-selector"],
                            )

                        with gr.Column(scale=4, elem_classes=["ui-card", "accent-violet", "advanced-step-card"]):
                            gr.HTML("""
                            <div class='card-heading'>
                                <span class='advanced-card-index'>3</span>
                                <div><span class='card-step'>STEP 03</span><b>Select model architecture</b></div>
                            </div>
                            <p class='card-helper'>The configuration area below updates automatically for the selected architecture.</p>
                            """)
                            model_kind = gr.Radio(
                                label="Model architecture",
                                choices=[
                                    "Three-Stage Ensemble (XGB+XGB+CatBoost)",
                                    "Single XGBoost (FAST 76-feature)",
                                    "Generalized Universal Ensemble (T-Agnostic)",
                                    "Custom Models (Template-based)",
                                ],
                                value="Three-Stage Ensemble (XGB+XGB+CatBoost)",
                                elem_classes=["architecture-selector"],
                            )
                            use_gpu = gr.Checkbox(
                                value=False,
                                label="Use GPU acceleration when supported",
                            )

                    gr.HTML("""
                    <div class='advanced-section-heading'>
                        <div><span class='workspace-kicker'>CONFIGURATION</span><div class='results-title'>Model and training settings</div></div>
                        <span>Only settings relevant to the selected architecture are displayed.</span>
                    </div>
                    """)

                    # ---------------------------------------------------------
                    # Model-specific configuration and common training setup
                    # ---------------------------------------------------------
                    with gr.Row(equal_height=False, elem_classes=["advanced-config-grid"]):
                        with gr.Column(scale=8, elem_classes=["advanced-config-main"]):
                            # Single XGBoost options
                            with gr.Column(visible=False, elem_classes=["ui-card", "accent-blue", "config-card"]) as single_xgb_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>SINGLE XGBOOST</span><b>Class weighting</b></div></div>
                                <p class='card-helper'>Control how class imbalance influences sample weights.</p>
                                """)
                                use_tempered_weights = gr.Checkbox(value=True, label="Use tempered sample weights")
                                alpha = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Weight alpha")
                                cap = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Weight cap")

                            with gr.Column(visible=False, elem_classes=["ui-card", "accent-violet", "config-card"]) as xgb_hypers_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>SINGLE XGBOOST</span><b>Hyperparameters</b></div></div>
                                """)
                                x_n_estimators = gr.Slider(200, 6000, value=2500, step=100, label="n_estimators")
                                x_lr = gr.Slider(0.005, 0.2, value=0.03, step=0.005, label="learning_rate")
                                x_max_depth = gr.Slider(3, 12, value=8, step=1, label="max_depth")
                                x_min_child_weight = gr.Slider(1, 10, value=1, step=1, label="min_child_weight")
                                x_gamma = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="gamma")
                                x_subsample = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="subsample")
                                x_colsample = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="colsample_bytree")
                                x_reg_lambda = gr.Slider(0.0, 10.0, value=2.0, step=0.5, label="reg_lambda")
                                x_reg_alpha = gr.Slider(0.0, 5.0, value=0.5, step=0.1, label="reg_alpha")
                                x_early_stop = gr.Slider(10, 200, value=50, step=10, label="early_stopping_rounds")

                            # Generalized Universal options — intentionally one control per line.
                            with gr.Column(visible=False, elem_classes=["ui-card", "accent-violet", "config-card"]) as universal_hypers_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>T-AGNOSTIC ENSEMBLE</span><b>Universal-model hyperparameters</b></div></div>
                                <p class='card-helper'>These values tune the length-independent XGBoost classifier and are stored with the experiment.</p>
                                """)
                                u_use_tempered_weights = gr.Checkbox(value=True, label="Use tempered class weights")
                                u_alpha = gr.Slider(0.1, 2.0, value=0.6, step=0.05, label="Weight alpha")
                                u_cap = gr.Slider(1.0, 20.0, value=3.0, step=0.5, label="Weight cap")
                                u_n_estimators = gr.Slider(100, 6000, value=1000, step=100, label="n_estimators")
                                u_lr = gr.Slider(0.005, 0.30, value=0.05, step=0.005, label="learning_rate")
                                u_max_depth = gr.Slider(2, 14, value=6, step=1, label="max_depth")
                                u_min_child_weight = gr.Slider(0.0, 20.0, value=1.0, step=0.5, label="min_child_weight")
                                u_gamma = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="gamma")
                                u_subsample = gr.Slider(0.4, 1.0, value=0.9, step=0.05, label="subsample")
                                u_colsample = gr.Slider(0.4, 1.0, value=0.9, step=0.05, label="colsample_bytree")
                                u_reg_lambda = gr.Slider(0.0, 20.0, value=1.0, step=0.5, label="reg_lambda")
                                u_reg_alpha = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="reg_alpha")
                                u_early_stop = gr.Slider(0, 300, value=50, step=10, label="early_stopping_rounds · 0 disables")

                            # Three-Stage features
                            with gr.Column(visible=True, elem_classes=["ui-card", "accent-cyan", "config-card"]) as three_feat_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>THREE-STAGE</span><b>Feature engineering groups</b></div></div>
                                <p class='card-helper'>Enable the temporal feature families included before model selection and scaling.</p>
                                """)
                                gr.HTML("<div class='feature-family-intro'>Select the feature families used by all three stages.</div>")
                                with gr.Row(equal_height=True, elem_classes=["feature-toggle-row"]):
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_basic = gr.Checkbox(value=True, label="Basic statistics")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_water = gr.Checkbox(value=True, label="Water presence")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_quarters = gr.Checkbox(value=True, label="Temporal quarters")
                                with gr.Row(equal_height=True, elem_classes=["feature-toggle-row"]):
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_trend = gr.Checkbox(value=True, label="Trend analysis")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_rejuv = gr.Checkbox(value=True, label="Rejuvenation")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_vshape = gr.Checkbox(value=True, label="V-shape")
                                with gr.Row(equal_height=True, elem_classes=["feature-toggle-row"]):
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_zero = gr.Checkbox(value=True, label="Zero patterns")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_var = gr.Checkbox(value=True, label="Variability")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_peaks = gr.Checkbox(value=True, label="Peaks")
                                with gr.Row(equal_height=True, elem_classes=["feature-toggle-row", "feature-toggle-row-last"]):
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_adv = gr.Checkbox(value=True, label="Advanced statistics")
                                    with gr.Column(min_width=180, elem_classes=["feature-toggle-cell"]):
                                        g_disc = gr.Checkbox(value=True, label="Discriminative features")

                            # Three-Stage hyperparameters
                            with gr.Column(visible=True, elem_classes=["ui-card", "accent-violet", "config-card"]) as three_params_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>THREE-STAGE</span><b>Stage parameters and thresholds</b></div></div>
                                """)
                                with gr.Accordion("Stage 1 · Rejuvenated detector", open=True, elem_classes=["soft-accordion"]):
                                    s1_n = gr.Slider(100, 1000, value=400, step=50, label="n_estimators")
                                    s1_depth = gr.Slider(3, 20, value=10, step=1, label="max_depth")
                                    s1_lr = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="learning_rate")
                                    s1_spw = gr.Slider(1, 30, value=15, step=1, label="scale_pos_weight")
                                with gr.Accordion("Stage 2 · Newly formed detector", open=False, elem_classes=["soft-accordion"]):
                                    s2_n = gr.Slider(100, 1000, value=400, step=50, label="n_estimators")
                                    s2_depth = gr.Slider(3, 20, value=10, step=1, label="max_depth")
                                    s2_lr = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="learning_rate")
                                    s2_spw = gr.Slider(1, 30, value=12, step=1, label="scale_pos_weight")
                                with gr.Accordion("Stage 3 · Persistent versus extinct", open=False, elem_classes=["soft-accordion"]):
                                    s3_iter = gr.Slider(100, 1000, value=400, step=50, label="iterations")
                                    s3_depth = gr.Slider(3, 15, value=8, step=1, label="depth")
                                    s3_lr = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="learning_rate")
                                with gr.Accordion("Decision thresholds", open=False, elem_classes=["soft-accordion"]):
                                    init_rth = gr.Slider(0.10, 0.70, value=0.30, step=0.01, label="Initial rejuvenated threshold")
                                    init_nth = gr.Slider(0.10, 0.70, value=0.35, step=0.01, label="Initial newly formed threshold")

                            # Custom model templates
                            with gr.Column(visible=False, elem_classes=["ui-card", "accent-rose", "config-card"]) as custom_models_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>CUSTOM TEMPLATE</span><b>Template and hyperparameters</b></div></div>
                                """)
                                custom_template_name = gr.Dropdown(
                                    choices=[
                                        "Logistic Regression",
                                        "SVC",
                                        "KNN",
                                        "Random Forest",
                                        "XGBoost (simple)",
                                    ],
                                    value="Logistic Regression",
                                    label="Template",
                                )
                                custom_use_scaler = gr.Checkbox(value=True, label="Use StandardScaler")
                                custom_use_class_weight = gr.Checkbox(value=True, label="Use balanced class weights when supported")

                                with gr.Column(visible=True, elem_classes=["sub-config-card"]) as custom_lr_block:
                                    gr.Markdown("#### Logistic Regression")
                                    custom_lr_C = gr.Slider(0.01, 20.0, value=1.0, step=0.05, label="C")
                                    custom_lr_max_iter = gr.Slider(50, 2000, value=200, step=50, label="max_iter")

                                with gr.Column(visible=False, elem_classes=["sub-config-card"]) as custom_svc_block:
                                    gr.Markdown("#### SVC")
                                    custom_svc_C = gr.Slider(0.01, 20.0, value=1.0, step=0.05, label="C")
                                    custom_svc_kernel = gr.Dropdown(choices=["rbf", "linear", "poly", "sigmoid"], value="rbf", label="kernel")
                                    custom_svc_gamma = gr.Dropdown(choices=["scale", "auto"], value="scale", label="gamma")

                                with gr.Column(visible=False, elem_classes=["sub-config-card"]) as custom_knn_block:
                                    gr.Markdown("#### KNN")
                                    custom_knn_k = gr.Slider(1, 50, value=5, step=1, label="n_neighbors")

                                with gr.Column(visible=False, elem_classes=["sub-config-card"]) as custom_rf_block:
                                    gr.Markdown("#### Random Forest")
                                    custom_rf_n = gr.Slider(50, 2000, value=300, step=50, label="n_estimators")
                                    custom_rf_depth = gr.Slider(0, 50, value=0, step=1, label="max_depth · 0 means None")

                                with gr.Column(visible=False, elem_classes=["sub-config-card"]) as custom_cxgb_block:
                                    gr.Markdown("#### XGBoost")
                                    custom_cxgb_n = gr.Slider(100, 3000, value=800, step=50, label="n_estimators")
                                    custom_cxgb_lr = gr.Slider(0.005, 0.3, value=0.05, step=0.005, label="learning_rate")
                                    custom_cxgb_depth = gr.Slider(2, 12, value=6, step=1, label="max_depth")

                        with gr.Column(scale=4, elem_classes=["advanced-side-stack"]):
                            with gr.Column(visible=True, elem_classes=["ui-card", "accent-blue", "config-card", "training-setup-card"]) as training_common_block:
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>TRAINING SETUP</span><b>Validation and optimization</b></div></div>
                                """)
                                test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Test size")
                                random_state = gr.Number(value=42, label="Random state", precision=0)
                                use_adasyn = gr.Checkbox(value=True, label="Use ADASYN for Three-Stage")
                                auto_opt = gr.Checkbox(value=True, label="Auto-optimize Three-Stage thresholds")
                                adasyn_neighbors = gr.Slider(2, 10, value=3, step=1, label="ADASYN n_neighbors")
                                fs_threshold = gr.Dropdown(choices=["mean", "median"], value="mean", label="Feature-selection threshold")

                            with gr.Column(elem_classes=["ui-card", "accent-rose", "training-action-card"]):
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>RUN CONTROL</span><b>Train or restore an experiment</b></div></div>
                                <p class='card-helper'>Training adds a new experiment. Undo restores the saved settings and outputs from an earlier run.</p>
                                """)
                                undo_to_id = gr.Dropdown(
                                    choices=[0],
                                    value=0,
                                    label="Restore experiment · 0 clears history",
                                )
                                with gr.Row(equal_height=True, elem_classes=["advanced-run-buttons"]):
                                    undo_btn = gr.Button("Undo", variant="secondary")
                                    train_btn = gr.Button(
                                        "Train model",
                                        variant="primary",
                                        size="lg",
                                        elem_classes=["hero-action", "advanced-train-button"],
                                    )

                    # ---------------------------------------------------------
                    # Results and deployment
                    # ---------------------------------------------------------
                    gr.HTML("""
                    <div class='results-heading advanced-results-heading'>
                        <div><span class='workspace-kicker'>RESULTS</span><div class='results-title'>Experiment analysis</div></div>
                        <span class='results-hint'>Compare runs, inspect diagnostics, and deploy the selected experiment.</span>
                    </div>
                    """)

                    with gr.Row(equal_height=False, elem_classes=["advanced-results-grid"]):
                        with gr.Column(scale=5, elem_classes=["advanced-result-panel"]):
                            best_html = gr.HTML(render_best_model_status([]))
                        with gr.Column(scale=7, elem_classes=["advanced-result-panel"]):
                            summary_out = gr.HTML(
                                "<div class='advanced-empty-state advanced-summary-empty'><span class='advanced-empty-mark'>KPI</span><div><b>No performance summary yet</b><small>Run training to view accuracy, Macro F1, and technical details.</small></div></div>"
                            )

                    with gr.Accordion("Diagnostics and errors", open=False, elem_classes=["advanced-diagnostics"]):
                        error_box = gr.Textbox(label="Runtime messages", lines=6)

                    with gr.Column(elem_classes=["deployment-panel"]):
                        gr.HTML("""
                        <div class='deployment-panel-copy'>
                            <div><span class='workspace-kicker'>DEPLOYMENT</span><div class='deployment-title'>Move a trained experiment into Operational mode</div></div>
                            <p>Deployment is session-only. Save to Drive separately only when the model should remain available after the runtime ends.</p>
                        </div>
                        """)
                        with gr.Row(equal_height=True, elem_classes=["deployment-controls"]):
                            deploy_exp_id = gr.Number(
                                label="Experiment ID · blank selects best accuracy",
                                value=None,
                                precision=0,
                                scale=2,
                            )
                            deploy_btn = gr.Button(
                                "Deploy to Operational",
                                variant="primary",
                                scale=2,
                            )
                            save_pond_model_to_drive_btn = gr.Button(
                                "Save deployed model to Drive",
                                variant="secondary",
                                interactive=False,
                                scale=2,
                            )
                        deploy_status_html = gr.HTML("")

                    with gr.Tabs(elem_classes=["results-tabs", "advanced-analysis-tabs"]):
                        with gr.TabItem("Confusion matrix"):
                            cm_plot = gr.Plot()
                        with gr.TabItem("Feature importance"):
                            fi_plot = gr.Plot()
                        with gr.TabItem("Actual vs predicted"):
                            avp_plot = gr.Plot()
                        with gr.TabItem("Classification report"):
                            report_out = gr.HTML(
                                "<div class='advanced-empty-state'><span class='advanced-empty-mark'>REPORT</span><div><b>No classification report yet</b><small>Train a model to populate per-class metrics.</small></div></div>"
                            )
                        with gr.TabItem("Experiment table"):
                            hist_table = gr.Dataframe(interactive=False)

            # Tab 2: Operational — compact prediction workspace
             with gr.TabItem("Operational"):
                 gr.HTML("""
                 <div class='workspace-hero pond-hero'>
                     <div class='workspace-hero-icon'>✦</div>
                     <div>
                         <div class='workspace-kicker'>POND CLASSIFICATION</div>
                         <div class='workspace-title'>Operational Prediction Workspace</div>
                         <div class='workspace-copy'>
                             Load data, choose a trained model, select its available classes, and run prediction.
                         </div>
                     </div>
                     <div class='workspace-badge'>LIVE WORKSPACE</div>
                 </div>
                 """)

                 with gr.Row(equal_height=False, elem_classes=["operational-grid"]):
                     with gr.Column(scale=5, elem_classes=["ui-card", "accent-cyan"]):
                         gr.HTML("""
                         <div class='card-heading'>
                             <span class='card-icon'>📂</span>
                             <div><span class='card-step'>STEP 01</span><b>Choose input data</b></div>
                         </div>
                         """)
                         with gr.Accordion("Dataset requirements", open=False, elem_classes=["soft-accordion"]):
                             gr.Markdown(DATASET_REQUIREMENTS_MD)

                         with gr.Tabs(elem_classes=["compact-tabs"]):
                             with gr.Tab("Local file"):
                                 op_upload_file = gr.File(
                                     label="CSV or Excel",
                                     file_types=[".csv", ".xlsx"],
                                     elem_classes=["compact-file"],
                                 )
                             with gr.Tab("Google Drive"):
                                 op_drive_path = gr.Textbox(
                                     label="Drive link or mounted path",
                                     placeholder="https://drive.google.com/... or /content/drive/...",
                                     lines=1,
                                 )

                         op_validate_btn = gr.Button(
                             "✓ Validate input",
                             variant="secondary",
                             size="sm",
                             elem_classes=["soft-button"],
                         )
                         op_validation_status = gr.HTML(
                             value=(
                                 "<div class='mini-empty'>"
                                 "<span class='mini-empty-icon'>✓</span>"
                                 "<div><b>Validation is optional</b>"
                                 "<small>Run it to inspect rows, temporal columns, and labels before prediction.</small></div>"
                                 "</div>"
                             )
                         )

                     with gr.Column(scale=4, elem_classes=["ui-card", "accent-violet"]):
                         gr.HTML("""
                         <div class='card-heading'>
                             <span class='card-icon'>🧠</span>
                             <div><span class='card-step'>STEP 02</span><b>Select a model</b></div>
                         </div>
                         """)
                         op_model_selector = gr.Dropdown(
                             label="Operational model",
                             choices=list(OPERATIONAL_MODELS.keys()),
                             value=_initial_op_model,
                             interactive=True,
                         )
                         op_refresh_models_btn = gr.Button(
                             "↻ Refresh from Drive",
                             variant="secondary",
                             size="sm",
                             elem_classes=["soft-button", "compact-action"],
                         )
                         _initial_op_desc, _initial_op_class_update = (
                             update_operational_model_ui(_initial_op_model)
                             if _initial_op_model
                             else ("", gr.update())
                         )
                         op_model_desc = gr.HTML(_initial_op_desc)

                     with gr.Column(scale=3, elem_classes=["ui-card", "accent-rose"]):
                         gr.HTML("""
                         <div class='card-heading'>
                             <span class='card-icon'>🎯</span>
                             <div><span class='card-step'>STEP 03</span><b>Choose classes</b></div>
                         </div>
                         <p class='card-helper'>
                             Only classes learned by the selected model are shown.
                         </p>
                         """)
                         op_target_classes_cb = gr.CheckboxGroup(
                             choices=_initial_op_classes,
                             value=_initial_op_classes,
                             label="Active prediction classes",
                             interactive=True,
                             elem_classes=["class-selector"],
                         )

                 with gr.Row(elem_classes=["action-dock"]):
                     gr.HTML("""
                     <div class='action-dock-copy'>
                         <span class='action-orb'>⚡</span>
                         <div><b>Ready to classify?</b><small>Prediction uses the selected model's saved preprocessing pipeline.</small></div>
                     </div>
                     """)
                     op_predict_btn = gr.Button(
                         "✨ Run Prediction",
                         variant="primary",
                         size="lg",
                         elem_classes=["hero-action"],
                     )

                 gr.HTML("""
                 <div class='results-heading'>
                     <div>
                         <span class='workspace-kicker'>OUTPUT</span>
                         <div class='results-title'>Prediction results</div>
                     </div>
                     <span class='results-hint'>Metrics appear when ground-truth labels are available</span>
                 </div>
                 """)

                 with gr.Column(elem_classes=["results-stage"]):
                     op_results_summary = gr.HTML(
                         "<div class='results-empty'>"
                         "<span>✦</span><b>Your prediction summary will appear here</b>"
                         "<small>Select data and a model, then click Run Prediction.</small>"
                         "</div>"
                     )

                     with gr.Tabs(visible=False, elem_classes=["results-tabs"]) as op_results_tabs:
                         with gr.TabItem("Metrics"):
                             op_metrics_out = gr.Dataframe(
                                 label="Performance metrics",
                                 interactive=False,
                             )
                         with gr.TabItem("Confusion matrix"):
                             op_cm_plot = gr.Plot()
                         with gr.TabItem("Classification report"):
                             op_report_out = gr.HTML()
                         with gr.TabItem("Download"):
                             gr.HTML(
                                 "<div class='download-panel'>"
                                 "<div class='download-panel-title'>⬇ Download predictions</div>"
                                 "<p class='download-panel-copy'>"
                                 "Prediction columns are appended to the source data and the output "
                                 "filename is derived from the original dataset name."
                                 "</p>"
                                 "</div>"
                             )
                             op_results_download = gr.File(
                                 label="Prediction file",
                                 interactive=False,
                             )

          # ============= OUTER TAB 2: Water Quality Assessment =============
          with gr.TabItem("Water Quality Assessment"):
            with gr.Tabs(elem_id="wq-subtabs", elem_classes=["stable-subtabs"]):
              # Sub-tab 1: Analysis Workspace
              with gr.TabItem("Analysis Workspace"):
                with gr.Column(elem_id="wq-analysis-workspace"):
                    gr.HTML("""
                    <div class='workspace-hero wq-analysis-hero'>
                        <div class='workspace-hero-icon wq-analysis-hero-icon'>WQ</div>
                        <div>
                            <div class='workspace-kicker'>WATER QUALITY ANALYSIS</div>
                            <div class='workspace-title'>Analysis Workspace</div>
                            <div class='workspace-copy'>
                                Select a workbook, choose an analytical model, tune its settings, and review the generated diagnostics.
                            </div>
                        </div>
                        <div class='workspace-badge'>ANALYZE · COMPARE · DEPLOY</div>
                    </div>
                    """)

                    with gr.Row(equal_height=False, elem_classes=["wq-analysis-top-grid"]):
                        with gr.Column(scale=6, elem_classes=["ui-card", "accent-cyan", "wq-analysis-card"]):
                            gr.HTML("""
                            <div class='card-heading'>
                                <span class='advanced-card-index'>1</span>
                                <div><span class='card-step'>STEP 01</span><b>Select a Water Quality dataset</b></div>
                            </div>
                            <p class='card-helper'>Choose a Drive dataset or add a workbook for the current runtime.</p>
                            """)
                            wq_dataset_dd = gr.Dropdown(
                                label=f"Water Quality dataset · {_wq_folder_display}",
                                choices=_wq_dataset_names,
                                value=_wq_dataset_names[0] if _wq_dataset_names else None,
                                multiselect=False,
                                interactive=True,
                            )
                            wq_refresh_datasets_btn = gr.Button(
                                "↻ Refresh datasets",
                                variant="secondary",
                                size="sm",
                                elem_classes=["soft-button", "compact-action"],
                            )
                            wq_dataset_refresh_status = gr.HTML("")

                            with gr.Accordion("Dataset summary", open=False, elem_classes=["soft-accordion"]):
                                initial_wq_summary_data = {_wq_dataset_names[0]: _wq_preloaded_datasets[_wq_dataset_names[0]]} if (_wq_preloaded_datasets and _wq_dataset_names) else {}
                                wq_dataset_summary_html = gr.HTML(
                                    render_wq_dataset_cards(initial_wq_summary_data) if initial_wq_summary_data
                                    else "<div class='advanced-empty-state'><span class='advanced-empty-mark'>WQ</span><div><b>No preloaded datasets</b><small>Add a workbook below to begin.</small></div></div>"
                                )

                            with gr.Accordion("Add a workbook", open=False, elem_classes=["soft-accordion"]):
                                with gr.Tabs(elem_classes=["compact-tabs"]):
                                    with gr.Tab("Local file"):
                                        wq_upload_file = gr.File(
                                            label="Excel workbook",
                                            file_types=[".xlsx", ".xls"],
                                        )
                                    with gr.Tab("Google Drive"):
                                        wq_drive_link = gr.Textbox(
                                            label="Drive link or mounted path",
                                            placeholder="https://drive.google.com/... or /content/drive/...",
                                            lines=1,
                                        )
                                with gr.Row(equal_height=True, elem_classes=["wq-sheet-row"]):
                                    wq_sheet_wq = gr.Number(
                                        value=WQ_SHEET_WQ,
                                        label="WQ sheet index",
                                        precision=0,
                                    )
                                    wq_sheet_wsa = gr.Number(
                                        value=WQ_SHEET_WSA,
                                        label="WSA sheet index",
                                        precision=0,
                                    )
                                wq_validate_btn = gr.Button(
                                    "Validate workbook",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["soft-button"],
                                )
                                wq_validation_status = gr.HTML(
                                    "<div class='mini-empty'><span class='mini-empty-icon'>✓</span><div><b>Workbook not validated</b><small>Select a local file or Drive path, then validate it.</small></div></div>"
                                )

                        with gr.Column(scale=4, elem_classes=["ui-card", "accent-violet", "wq-analysis-card"]):
                            gr.HTML("""
                            <div class='card-heading'>
                                <span class='advanced-card-index'>2</span>
                                <div><span class='card-step'>STEP 02</span><b>Select an analytical model</b></div>
                            </div>
                            <p class='card-helper'>Model 2 uses the Model 1 result as its reference, so run Model 1 first when required.</p>
                            """)
                            wq_model_choice = gr.Radio(
                                label="Analysis model",
                                choices=[
                                    "Model 1: PCA + KMeans + TS + Resilience",
                                    "Model 2: DTW Shape Classifier",
                                    "Model 3: WSA + NDCI + NDTI Triple-Stream",
                                ],
                                value="Model 1: PCA + KMeans + TS + Resilience",
                                elem_classes=["wq-model-choice"],
                            )
                            gr.HTML("""
                            <div class='wq-model-note'>
                                <b>Dependency note</b>
                                <span>Run Model 1 before Model 2 in a new session.</span>
                            </div>
                            """)

                    gr.HTML("""
                    <div class='advanced-section-heading wq-analysis-section-heading'>
                        <div><span class='workspace-kicker'>CONFIGURATION</span><div class='results-title'>Model parameters</div></div>
                        <span>Adjust only the settings needed for the selected analysis.</span>
                    </div>
                    """)

                    with gr.Row(equal_height=False, elem_classes=["wq-analysis-params-grid"]):
                        with gr.Column(scale=6, elem_classes=["ui-card", "accent-blue", "config-card", "wq-parameter-card"]):
                            gr.HTML("""
                            <div class='config-card-heading'><div><span class='card-step'>MODEL 1</span><b>PCA, clustering, and resilience</b></div></div>
                            <p class='card-helper'>Controls the cluster search, dimensionality reduction, and bloom-frequency classification.</p>
                            """)
                            wq_n_clusters = gr.Slider(2, 10, value=3, step=1, label="Number of clusters (K)")
                            wq_k_range_max = gr.Slider(5, 15, value=10, step=1, label="K-search range maximum")
                            wq_pca_components = gr.Slider(2, 6, value=4, step=1, label="PCA components")
                            wq_bloom_threshold = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Bloom-frequency threshold")

                        with gr.Column(scale=4, elem_classes=["ui-card", "accent-violet", "config-card", "wq-parameter-card"]):
                            gr.HTML("""
                            <div class='config-card-heading'><div><span class='card-step'>MODEL 3</span><b>Triple-stream clustering</b></div></div>
                            <p class='card-helper'>Configures the combined WSA, NDCI, and NDTI representation.</p>
                            """)
                            wq_m3_n_clusters = gr.Slider(2, 6, value=4, step=1, label="Number of clusters (K)")
                            wq_m3_pca_each = gr.Slider(1, 4, value=2, step=1, label="PCA components per stream")

                    with gr.Row(elem_classes=["action-dock", "wq-analysis-run-dock"]):
                        gr.HTML("""
                        <div class='action-dock-copy'>
                            <span class='action-orb wq-run-orb'>RUN</span>
                            <div><b>Ready to analyze?</b><small>The selected model and current parameter values will be applied to the active dataset.</small></div>
                        </div>
                        """)
                        wq_run_btn = gr.Button(
                            "Run analysis",
                            variant="primary",
                            size="lg",
                            elem_classes=["hero-action", "wq-analysis-run-button"],
                        )

                    gr.HTML("""
                    <div class='results-heading wq-analysis-results-heading'>
                        <div><span class='workspace-kicker'>RESULTS</span><div class='results-title'>Water Quality analysis</div></div>
                        <span class='results-hint'>Review the summary, diagnostic views, and generated output file.</span>
                    </div>
                    """)

                    with gr.Column(elem_classes=["results-stage", "wq-analysis-results-stage"]):
                        wq_summary_html = gr.HTML(
                            "<div class='results-empty'><span>WQ</span><b>Your analysis summary will appear here</b><small>Select a dataset and model, then run the analysis.</small></div>"
                        )

                        with gr.Tabs(elem_classes=["results-tabs", "wq-analysis-tabs"]):
                            with gr.TabItem("Elbow and silhouette"):
                                wq_plot1 = gr.Plot()
                            with gr.TabItem("CH score and normalized heatmap"):
                                wq_plot2 = gr.Plot()
                            with gr.TabItem("Behavioral distribution"):
                                wq_plot3 = gr.Plot()
                            with gr.TabItem("Temporal shape distribution"):
                                wq_plot4 = gr.Plot()
                            with gr.TabItem("Time-series pattern"):
                                wq_plot5 = gr.Plot()
                            with gr.TabItem("Resilience"):
                                wq_plot6 = gr.Plot()
                            with gr.TabItem("Additional view"):
                                wq_plot7 = gr.Plot()

                        with gr.Row(equal_height=False, elem_classes=["wq-analysis-footer-grid"]):
                            with gr.Column(scale=4, elem_classes=["ui-card", "accent-cyan", "wq-analysis-footer-card"]):
                                gr.HTML("""
                                <div class='config-card-heading'><div><span class='card-step'>EXPORT</span><b>Download analysis results</b></div></div>
                                <p class='card-helper'>The generated CSV contains the analytical outputs for the active dataset.</p>
                                """)
                                wq_download_file = gr.File(label="Analysis result file", interactive=False)

                            with gr.Column(scale=6, elem_classes=["deployment-panel", "wq-analysis-deployment"]):
                                gr.HTML("""
                                <div class='deployment-panel-copy wq-analysis-deployment-copy'>
                                    <div><span class='workspace-kicker'>DEPLOYMENT</span><div class='deployment-title'>Make this model available in Operational Predict</div></div>
                                    <p>Deploy for the current session first. Save to Drive separately only when the model should remain available after the runtime ends.</p>
                                </div>
                                """)
                                with gr.Row(equal_height=True, elem_classes=["deployment-controls", "wq-deployment-buttons"]):
                                    wq_deploy_btn = gr.Button(
                                        "Deploy analyzed model",
                                        variant="primary",
                                    )
                                    wq_save_model_to_drive_btn = gr.Button(
                                        "Save deployed model to Drive",
                                        variant="secondary",
                                        interactive=False,
                                    )
                                wq_deploy_status_html = gr.HTML("")

              # Sub-tab 2: Operational
              with gr.TabItem("Operational Predict"):
                gr.HTML("""
                <div class='workspace-hero water-hero'>
                    <div class='workspace-hero-icon'>◌</div>
                    <div>
                        <div class='workspace-kicker'>WATER QUALITY</div>
                        <div class='workspace-title'>Operational Assessment Workspace</div>
                        <div class='workspace-copy'>
                            Validate an Excel workbook, select a saved analytical model, and generate typology predictions.
                        </div>
                    </div>
                    <div class='workspace-badge'>WQ OPERATIONS</div>
                </div>
                """)

                with gr.Row(equal_height=False, elem_classes=["operational-grid"]):
                    with gr.Column(scale=6, elem_classes=["ui-card", "accent-cyan"]):
                        gr.HTML("""
                        <div class='card-heading'>
                            <span class='card-icon'>📊</span>
                            <div><span class='card-step'>STEP 01</span><b>Choose workbook</b></div>
                        </div>
                        <p class='card-helper'>The workbook must contain the configured NDCI/NDTI and WSA sheets.</p>
                        """)
                        with gr.Tabs(elem_classes=["compact-tabs"]):
                            with gr.Tab("Local file"):
                                wq_op_upload = gr.File(
                                    label="Excel workbook",
                                    file_types=[".xlsx", ".xls"],
                                    elem_classes=["compact-file"],
                                )
                            with gr.Tab("Google Drive"):
                                wq_op_drive = gr.Textbox(
                                    label="Drive link or mounted path",
                                    placeholder="https://drive.google.com/... or /content/drive/...",
                                    lines=1,
                                )
                        with gr.Row():
                            wq_op_sheet_wq = gr.Number(
                                value=WQ_SHEET_WQ,
                                label="WQ sheet index",
                                precision=0,
                            )
                            wq_op_sheet_wsa = gr.Number(
                                value=WQ_SHEET_WSA,
                                label="WSA sheet index",
                                precision=0,
                            )
                        wq_op_validate_btn = gr.Button(
                            "✓ Validate workbook",
                            variant="secondary",
                            size="sm",
                            elem_classes=["soft-button"],
                        )
                        wq_op_validation_status = gr.HTML(
                            "<div class='mini-empty'>"
                            "<span class='mini-empty-icon'>✓</span>"
                            "<div><b>Validation is optional</b>"
                            "<small>Inspect workbook structure before running prediction.</small></div>"
                            "</div>"
                        )

                    with gr.Column(scale=5, elem_classes=["ui-card", "accent-blue"]):
                        gr.HTML("""
                        <div class='card-heading'>
                            <span class='card-icon'>🧬</span>
                            <div><span class='card-step'>STEP 02</span><b>Select a WQ model</b></div>
                        </div>
                        """)
                        wq_op_model = gr.Dropdown(
                            label="Operational WQ model",
                            choices=list(WQ_OPERATIONAL_MODELS.keys()),
                            value=_initial_wq_model,
                            interactive=True,
                        )
                        wq_refresh_models_btn = gr.Button(
                            "↻ Refresh from Drive",
                            variant="secondary",
                            size="sm",
                            elem_classes=["soft-button", "compact-action"],
                        )
                        wq_op_model_desc = gr.HTML(
                            get_wq_model_details(_initial_wq_model)
                            if _initial_wq_model else ""
                        )

                with gr.Row(elem_classes=["action-dock", "action-dock-water"]):
                    gr.HTML("""
                    <div class='action-dock-copy'>
                        <span class='action-orb'>💧</span>
                        <div><b>Ready to assess?</b><small>The saved WQ pipeline will be applied to the selected workbook.</small></div>
                    </div>
                    """)
                    wq_op_run_btn = gr.Button(
                        "✨ Predict Water Quality",
                        variant="primary",
                        size="lg",
                        elem_classes=["hero-action"],
                    )

                gr.HTML("""
                <div class='results-heading'>
                    <div>
                        <span class='workspace-kicker'>OUTPUT</span>
                        <div class='results-title'>Water-quality results</div>
                    </div>
                    <span class='results-hint'>Explore plots, fingerprints, and the generated result file</span>
                </div>
                """)

                with gr.Column(elem_classes=["results-stage"]):
                    wq_op_summary = gr.HTML(
                        "<div class='results-empty'>"
                        "<span>💧</span><b>Your water-quality summary will appear here</b>"
                        "<small>Select a workbook and model, then run prediction.</small>"
                        "</div>"
                    )

                    with gr.Tabs(visible=False, elem_classes=["results-tabs"]) as wq_op_tabs:
                        with gr.TabItem("Behavioral space"):
                            wq_op_plot1 = gr.Plot()
                        with gr.TabItem("Temporal fingerprints"):
                            wq_op_plot2 = gr.Plot()
                        with gr.TabItem("Distribution"):
                            wq_op_plot3 = gr.Plot()
                        with gr.TabItem("Additional view"):
                            wq_op_plot4 = gr.Plot()
                        with gr.TabItem("Download"):
                            gr.HTML(
                                "<div class='download-panel'>"
                                "<div class='download-panel-title'>⬇ Download predictions</div>"
                                "<p class='download-panel-copy'>"
                                "Water-quality predictions are appended to the source data and the "
                                "filename is derived from the original workbook name."
                                "</p>"
                                "</div>"
                            )
                            wq_op_download = gr.File(
                                label="Prediction file",
                                interactive=False,
                            )

        # ===================== EVENT WIRING =====================

        # Show/hide blocks after selecting model
        model_kind.change(
            fn=toggle_model_panels,
            inputs=[model_kind],
            outputs=[single_xgb_block, xgb_hypers_block, universal_hypers_block, three_feat_block, custom_models_block, training_common_block, three_params_block]
        )

        # Custom template: show correct hyperparam subgroup
        custom_template_name.change(
            fn=toggle_custom_template_panels,
            inputs=[custom_template_name],
            outputs=[custom_lr_block, custom_svc_block, custom_knn_block, custom_rf_block, custom_cxgb_block]
        )

        # Train click
        train_btn.click(
            fn=run_experiment,
            inputs=[
                dataset_dd, st, model_kind,
                use_tempered_weights, alpha, cap, use_gpu,
                x_n_estimators, x_lr, x_max_depth, x_min_child_weight, x_gamma,
                x_subsample, x_colsample, x_reg_lambda, x_reg_alpha, x_early_stop,
                # Generalized Universal (T-agnostic) params
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
                # custom model inputs
                custom_template_name, custom_use_scaler, custom_use_class_weight,
                custom_lr_C, custom_lr_max_iter,
                custom_svc_C, custom_svc_kernel, custom_svc_gamma,
                custom_knn_k,
                custom_rf_n, custom_rf_depth,
                custom_cxgb_n, custom_cxgb_lr, custom_cxgb_depth,
                # dynamic target classes
                target_classes_cb
            ],
            outputs=[st, best_html, cm_plot, fi_plot, avp_plot, report_out, hist_table, summary_out, undo_to_id, error_box]
        )

        # Undo click - restores ALL state including hyperparameters
        undo_btn.click(
            fn=undo_to_experiment,
            inputs=[st, undo_to_id],
            outputs=[
                st, best_html, cm_plot, fi_plot, avp_plot, report_out, hist_table, summary_out, undo_to_id, error_box,
                # Single XGBoost params
                use_tempered_weights, alpha, cap, use_gpu,
                x_n_estimators, x_lr, x_max_depth, x_min_child_weight, x_gamma,
                x_subsample, x_colsample, x_reg_lambda, x_reg_alpha, x_early_stop,
                # Generalized Universal (T-agnostic) params
                u_use_tempered_weights, u_alpha, u_cap,
                u_n_estimators, u_lr, u_max_depth, u_min_child_weight, u_gamma,
                u_subsample, u_colsample, u_reg_lambda, u_reg_alpha, u_early_stop,
                # Feature groups
                g_basic, g_water, g_quarters, g_trend, g_rejuv, g_vshape, g_zero, g_var, g_peaks, g_adv, g_disc,
                # Common params
                test_size, random_state,
                # Three-Stage params
                use_adasyn, adasyn_neighbors, auto_opt, fs_threshold,
                s1_n, s1_depth, s1_lr, s1_spw,
                s2_n, s2_depth, s2_lr, s2_spw,
                s3_iter, s3_depth, s3_lr,
                init_rth, init_nth,
                # Custom Models params
                custom_template_name, custom_use_scaler, custom_use_class_weight,
                custom_lr_C, custom_lr_max_iter,
                custom_svc_C, custom_svc_kernel, custom_svc_gamma,
                custom_knn_k,
                custom_rf_n, custom_rf_depth,
                custom_cxgb_n, custom_cxgb_lr, custom_cxgb_depth
            ]
        )

        # Validate uploaded dataset (Advanced)
        validate_btn.click(
            fn=lambda f, d, s: validate_uploaded_dataset(f, d, s, require_label=True),
            inputs=[upload_file, drive_path, st],
            outputs=[gr.State(), validation_status, st, gr.State(), dataset_dd, target_classes_cb, op_target_classes_cb]
        )
        upload_file.change(
            fn=lambda f, d, s: validate_uploaded_dataset(f, d, s, require_label=True),
            inputs=[upload_file, drive_path, st],
            outputs=[gr.State(), validation_status, st, gr.State(), dataset_dd, target_classes_cb, op_target_classes_cb]
        )

        # Validate uploaded dataset (Operational)
        op_validate_btn.click(
            fn=validate_operational_dataset,
            inputs=[op_upload_file, op_drive_path, st],
            outputs=[gr.State(), op_validation_status, st, gr.State(), dataset_dd, gr.State(), op_target_classes_cb]
        )
        op_upload_file.change(
            fn=validate_operational_dataset,
            inputs=[op_upload_file, op_drive_path, st],
            outputs=[gr.State(), op_validation_status, st, gr.State(), dataset_dd, gr.State(), op_target_classes_cb]
        )

        def update_classes_and_summary_for_dataset(state, ds_names):
            return _pond_dataset_selection_updates(state or new_state(), ds_names)

        dataset_dd.change(
            fn=update_classes_and_summary_for_dataset,
            inputs=[st, dataset_dd],
            outputs=[target_classes_cb, dataset_summary_html]
        )

        pond_refresh_datasets_btn.click(
            fn=refresh_pond_datasets_ui,
            inputs=[st, dataset_dd],
            outputs=[
                dataset_dd,
                target_classes_cb,
                dataset_summary_html,
                st,
                pond_dataset_refresh_status,
            ],
        )

        def update_wq_classes_and_summary_for_dataset(state, ds_name):
            state = state or new_state()
            return _activate_wq_dataset(state, ds_name), state

        wq_dataset_dd.change(
            fn=update_wq_classes_and_summary_for_dataset,
            inputs=[st, wq_dataset_dd],
            outputs=[wq_dataset_summary_html, st]
        )

        wq_refresh_datasets_btn.click(
            fn=refresh_wq_datasets_ui,
            inputs=[st, wq_dataset_dd],
            outputs=[
                wq_dataset_dd,
                wq_dataset_summary_html,
                st,
                wq_dataset_refresh_status,
            ],
        )

        # Deploy trained model → Operational Mode
        deploy_btn.click(
            fn=deploy_experiment_to_operational,
            inputs=[st, deploy_exp_id],
            outputs=[
                op_model_selector,
                op_model_desc,
                op_target_classes_cb,
                deploy_status_html,
                pond_deployment_state,
                save_pond_model_to_drive_btn,
            ]
        )

        save_pond_model_to_drive_btn.click(
            fn=save_deployed_pond_model_to_drive,
            inputs=[pond_deployment_state],
            outputs=[
                op_model_selector,
                op_model_desc,
                op_target_classes_cb,
                deploy_status_html,
                pond_deployment_state,
                save_pond_model_to_drive_btn,
            ]
        )

        # --- OPERATIONAL TAB EVENTS ---

        op_refresh_models_btn.click(
            fn=refresh_operational_models_ui,
            inputs=[op_model_selector],
            outputs=[op_model_selector, op_model_desc, op_target_classes_cb]
        )

        op_model_selector.change(
            fn=update_operational_model_ui,
            inputs=[op_model_selector],
            outputs=[op_model_desc, op_target_classes_cb]
        )

        op_predict_btn.click(
            fn=operational_predict,
            inputs=[op_model_selector, op_upload_file, op_drive_path, st, op_target_classes_cb],
            outputs=[op_results_summary, op_results_tabs, op_metrics_out, op_cm_plot, op_report_out, op_results_download, st]
        )


        # --- WATER QUALITY TAB EVENTS ---

        # WQ Validate
        wq_validate_btn.click(
            fn=wq_validate_data,
            inputs=[wq_upload_file, wq_drive_link, wq_sheet_wq, wq_sheet_wsa, st],
            outputs=[wq_validation_status, st, wq_dataset_dd]
        )

        wq_upload_file.change(
            fn=wq_validate_data,
            inputs=[wq_upload_file, wq_drive_link, wq_sheet_wq, wq_sheet_wsa, st],
            outputs=[wq_validation_status, st, wq_dataset_dd]
        )

        # WQ Run Analysis
        wq_run_btn.click(
            fn=wq_run_analysis,
            inputs=[wq_model_choice, wq_n_clusters, wq_k_range_max, wq_pca_components,
                    wq_bloom_threshold, wq_m3_n_clusters, wq_m3_pca_each,
                    wq_upload_file, wq_drive_link, wq_sheet_wq, wq_sheet_wsa, st],
            outputs=[st, wq_summary_html, wq_plot1, wq_plot2, wq_plot3, wq_plot4, wq_plot5, wq_plot6, wq_plot7, wq_download_file]
        )

        # WQ Operational Validate
        wq_op_validate_btn.click(
            fn=wq_validate_data,
            inputs=[wq_op_upload, wq_op_drive, wq_op_sheet_wq, wq_op_sheet_wsa, st],
            outputs=[wq_op_validation_status, st, wq_dataset_dd]
        )

        wq_op_upload.change(
            fn=wq_validate_data,
            inputs=[wq_op_upload, wq_op_drive, wq_op_sheet_wq, wq_op_sheet_wsa, st],
            outputs=[wq_op_validation_status, st, wq_dataset_dd]
        )

        # WQ Operational Predict
        wq_op_run_btn.click(
            fn=lambda: gr.update(visible=True),  # Show results tabs immediately when clicked
            inputs=None,
            outputs=[wq_op_tabs]
        ).then(
            fn=wq_operational_predict,
            inputs=[wq_op_model, wq_op_upload, wq_op_drive, wq_op_sheet_wq, wq_op_sheet_wsa, st],
            outputs=[wq_op_summary, wq_op_plot1, wq_op_plot2, wq_op_plot3, wq_op_plot4, wq_op_download, st]
        )

        # WQ Deploy Model
        wq_deploy_btn.click(
            fn=deploy_wq_model_to_operational,
            inputs=[st],
            outputs=[
                wq_op_model,
                wq_op_model_desc,
                wq_deploy_status_html,
                wq_deployment_state,
                wq_save_model_to_drive_btn,
            ]
        )

        wq_save_model_to_drive_btn.click(
            fn=save_deployed_wq_model_to_drive,
            inputs=[wq_deployment_state],
            outputs=[
                wq_op_model,
                wq_op_model_desc,
                wq_deploy_status_html,
                wq_deployment_state,
                wq_save_model_to_drive_btn,
            ]
        )

        wq_refresh_models_btn.click(
            fn=refresh_wq_operational_models_ui,
            inputs=[wq_op_model],
            outputs=[wq_op_model, wq_op_model_desc]
        )

        # WQ Operational Model Change Description
        wq_op_model.change(
            fn=get_wq_model_details,
            inputs=[wq_op_model],
            outputs=[wq_op_model_desc]
        )

        return demo
