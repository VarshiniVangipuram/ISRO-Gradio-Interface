# Codebase Map — interface.py (10,566 lines)

Complete inventory of all functions, globals, constants, and their dependencies.

---

## 1. Imports (lines 1–61, 567–568)

**Standard library:** os, re, json, time, copy, traceback, mimetypes, html, pathlib.Path, urllib.parse
**Third-party:** numpy, pandas, matplotlib, seaborn, joblib, requests, gradio
**sklearn:** train_test_split, StandardScaler, LabelEncoder, confusion_matrix, classification_report, accuracy_score, f1_score, RandomForestClassifier, SelectFromModel, compute_sample_weight, KMeans, silhouette_score, calinski_harabasz_score, PCA, LogisticRegression, SVC, KNeighborsClassifier
**imblearn:** ADASYN
**scipy:** stats
**xgboost:** xgb
**catboost:** CatBoostClassifier
**Colab:** google.colab.drive (try/except)

---

## 2. Constants & Configuration (lines 63–118)

| Name | Value | Line |
|------|-------|------|
| `DATASET_FOLDER` | `"/content/drive/MyDrive/Pond Classification/Data"` | 68 |
| `BEST_MODELS_ROOT` | `Path("/content/drive/MyDrive/Best_Models")` | 70 |
| `POND_OPERATIONAL_MODELS_FOLDER` | `BEST_MODELS_ROOT / "Pond_Classification_Models" / "Operational_Active"` | 71 |
| `WQ_OPERATIONAL_MODELS_FOLDER` | `BEST_MODELS_ROOT / "Water_Quality_Models" / "Operational_Active"` | 74 |
| `SESSION_DEPLOYMENTS_ROOT` | `Path("/content/session_deployed_models")` | 81 |
| `POND_SESSION_DEPLOYMENTS_FOLDER` | `SESSION_DEPLOYMENTS_ROOT / "Pond_Classification"` | 82 |
| `WQ_SESSION_DEPLOYMENTS_FOLDER` | `SESSION_DEPLOYMENTS_ROOT / "Water_Quality"` | 83 |
| `TRAINED_MODELS_FOLDER` | `str(POND_SESSION_DEPLOYMENTS_FOLDER)` | 86 |
| `WQ_TRAINED_MODELS_FOLDER` | `str(WQ_SESSION_DEPLOYMENTS_FOLDER)` | 87 |
| `GENERATED_RESULTS_FOLDER` | `Path("/content/generated_results")` | 108 |
| `WQ_DATASET_FOLDER` | `"/content/drive/MyDrive/Water_Quality_Exports"` | 114 |
| `WQ_DATASET_PATH` | `"/content/drive/MyDrive/Water_Quality_Exports/HYD_WSA_WQ_Feb2026.xlsx"` | 115 |
| `WQ_SHEET_WQ` | `1` | 116 |
| `WQ_SHEET_WSA` | `0` | 117 |
| `ALL_KNOWN_CLASSES` | `["rejuvenated", "persistent", "newly formed", "extinct"]` | 414 |
| `DEFAULT_TARGET_CLASSES` | `["rejuvenated", "persistent", "newly formed", "extinct"]` | 416 |

---

## 3. Global Variables / Registries (lines 89–105)

| Name | Initial | Line | Purpose |
|------|---------|------|---------|
| `OPERATIONAL_MODELS` | `{}` | 90 | Pond model name → file path |
| `OPERATIONAL_LABEL_ENCODERS` | `{}` | 91 | Pond model name → encoder path |
| `OPERATIONAL_MODEL_METADATA` | `{}` | 92 | Pond model name → metadata dict |
| `WQ_OPERATIONAL_MODELS` | `{}` | 93 | WQ model name → file path |
| `_USER_TRAINED_MODELS` | `{}` | 97 | Session-deployed pond models |
| `_USER_TRAINED_WQ_MODELS` | `{}` | 98 | Session-deployed WQ models |
| `_OPERATIONAL_MODEL_CACHE` | `{}` | 101 | Loaded pond model cache |
| `_WQ_OPERATIONAL_MODEL_CACHE` | `{}` | 102 | Loaded WQ model cache |
| `_DRIVE_URL_FILENAME_CACHE` | `{}` | 105 | Drive download filename cache |

---

## 4. UI Constants

| Name | Lines | Purpose |
|------|-------|---------|
| `CSS` | 5198–7865 | All Gradio CSS styling (~2,667 lines) |
| `DATASET_REQUIREMENTS_MD` | 7867–7891 | Markdown for dataset requirements accordion |
| `theme` | 7893 | `gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")` |
| `DROPDOWN_JS` | 8917–9198 | JavaScript for dropdown portal fix + slider sync |

---

## 5. Boot-time Variables (lines 410–411, 7895–7935, 9385–9391)

| Name | Line | Purpose |
|------|------|---------|
| `_POND_MODEL_SCAN_LOG` | 410 | Output of initial pond model scan |
| `_WQ_MODEL_SCAN_LOG` | 411 | Output of initial WQ model scan |
| `_preloaded_datasets` | 7896 | Auto-loaded pond datasets |
| `_preload_log` | 7896 | Loading log |
| `_dataset_names` | 7897 | List of dataset names |
| `_active_folder` | 7899 | Resolved dataset folder |
| `_folder_display` | 7900–7903 | Display name for folder |
| `_initial_classes` | 7905–7909 | Classes from first dataset |
| `_wq_preloaded_datasets` | 7912 | Auto-loaded WQ datasets |
| `_wq_preload_log` | 7912 | WQ loading log |
| `_wq_dataset_names` | 7913 | List of WQ dataset names |
| `_wq_active_folder` | 7915 | WQ dataset folder |
| `_wq_folder_display` | 7916 | WQ folder display name |
| `_wq_preloaded` | 7919–7933 | First WQ dataset for backward compat |
| `_initial_op_model` | 9385 | First operational pond model |
| `_initial_op_classes` | 9387–9389 | Classes from first op model |
| `_initial_wq_model` | 9391 | First operational WQ model |

---

## 6. Functions by Module

### 6A. Model Discovery & Registry (→ `registry.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `_safe_joblib_load(path)` | 120 | joblib |
| `_is_label_encoder_filename(filename)` | 128 | — |
| `_legacy_pond_display_name(filename)` | 133 | Path |
| `_legacy_pond_pipeline(filename)` | 152 | — |
| `_find_matching_encoder(model_path, encoder_paths)` | 161 | Path |
| `_looks_like_pond_model(obj)` | 194 | — |
| `discover_pond_operational_models()` | 204 | _safe_joblib_load, _is_label_encoder_filename, _legacy_pond_display_name, _legacy_pond_pipeline, _find_matching_encoder, _looks_like_pond_model, POND_OPERATIONAL_MODELS_FOLDER |
| `_wq_display_name(bundle, model_path)` | 285 | — |
| `_looks_like_wq_model(obj)` | 294 | — |
| `discover_wq_operational_models()` | 306 | _safe_joblib_load, _looks_like_wq_model, WQ_OPERATIONAL_MODELS_FOLDER |
| `rescan_pond_model_registry()` | 338 | discover_pond_operational_models, OPERATIONAL_MODELS, OPERATIONAL_LABEL_ENCODERS, OPERATIONAL_MODEL_METADATA |
| `rescan_wq_model_registry()` | 354 | discover_wq_operational_models, WQ_OPERATIONAL_MODELS, _WQ_OPERATIONAL_MODEL_CACHE |
| `_unique_drive_destination(base_name, folder)` | 363 | re |
| `_unique_session_destination(base_name, folder, occupied_names)` | 377 | re |
| `_pond_model_choices()` | 393 | OPERATIONAL_MODELS, _USER_TRAINED_MODELS |
| `_wq_model_choices()` | 401 | WQ_OPERATIONAL_MODELS, _USER_TRAINED_WQ_MODELS |

### 6B. Dataset Card Rendering (→ `ui/cards.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `render_dataset_cards(preloaded_datasets)` | 422 | — |
| `render_wq_dataset_cards(wq_preloaded_datasets)` | 494 | — |

### 6C. File & Drive Utilities (→ `data/drive_utils.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `_filename_from_content_disposition(header_value)` | 572 | re |
| `_google_drive_file_id(url)` | 604 | re, urllib.parse |
| `_safe_source_filename(upload_file, drive_path, fallback)` | 619 | _filename_from_content_disposition, _google_drive_file_id, _DRIVE_URL_FILENAME_CACHE |
| `_result_csv_path(source_filename, suffix)` | 645 | os, Path, GENERATED_RESULTS_FOLDER |
| `download_drive_link(url)` | 659 | _google_drive_file_id, requests, _filename_from_content_disposition, _DRIVE_URL_FILENAME_CACHE |

### 6D. Best Model Display & Download (→ `ui/render.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `render_best_model_status(history)` | 716 | html |
| `download_best_model(state)` | 776 | joblib, GENERATED_RESULTS_FOLDER |
| `build_model_display_name(entry)` | 823 | — |

### 6E. Deployment Functions (→ `registry.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `deploy_experiment_to_operational(state, exp_id)` | 914 | _unique_session_destination, POND_SESSION_DEPLOYMENTS_FOLDER, _USER_TRAINED_MODELS |
| `save_deployed_pond_model_to_drive(deployment_record)` | 1064 | _unique_drive_destination, POND_OPERATIONAL_MODELS_FOLDER, _USER_TRAINED_MODELS, OPERATIONAL_MODELS |
| `_build_wq_drive_name(artifacts)` | 1165 | re |
| `deploy_wq_model_to_operational(state)` | 1194 | _unique_session_destination, WQ_SESSION_DEPLOYMENTS_FOLDER, _USER_TRAINED_WQ_MODELS |
| `save_deployed_wq_model_to_drive(deployment_record)` | 1281 | _build_wq_drive_name, _unique_drive_destination, WQ_OPERATIONAL_MODELS_FOLDER, _USER_TRAINED_WQ_MODELS, WQ_OPERATIONAL_MODELS |

### 6F. Performance Dashboard & Classification Report (→ `ui/render.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `render_performance_dashboard(...)` | 1377 | html |
| `render_classification_report(report_str, classes)` | 1454 | html, parse_classification_report |
| `parse_classification_report(report_str)` | ~1654 | pandas |
| `plot_classification_report_safe(report_str)` | ~1654+ | matplotlib, parse_classification_report |

### 6G. Data Loading & Preprocessing (→ `data/loader.py`, `data/preprocess.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `smart_read_csv(path_or_buf)` | ~1674 | pandas, io, csv |
| `read_table_path(path_or_obj)` | ~1674+ | smart_read_csv, download_drive_link, pandas |
| `normalize_status(s)` | ~1854 | re |
| `find_label_col_finalstatus_only(df)` | ~1854+ | normalize_status |
| `pick_areacols_like_three_stage(df)` | ~1854+ | — |
| `pick_temporalcols_like_xgb(df)` | ~1854+ | — |
| `resample_series_to_length(a, T)` | ~1854+ | numpy |
| `preprocess_dataset_to_unified(df, path, T, require_label)` | ~1854+ | normalize_status, find_label_col_finalstatus_only, pick_areacols_like_three_stage, pick_temporalcols_like_xgb, resample_series_to_length |
| `autoload_and_preprocess_all()` | ~2053 | DATASET_FOLDER, preprocess_dataset_to_unified, smart_read_csv, read_table_path |
| `autoload_wq_datasets()` | ~2053+ | WQ_DATASET_FOLDER, wq_load_excel, wq_detect_columns |

### 6H. Feature Extractors (→ `features/`)

| Function | Line | Module | Dependencies |
|----------|------|--------|--------------|
| `extract_discriminative_features(row, areacols, groups)` | ~1973 | `features/discriminative.py` | numpy, scipy.stats, safex |
| `build_feature_frame_three_stage(df, areacols, labelcol, groups)` | ~2253 | `features/discriminative.py` | extract_discriminative_features |
| `extract_shape_only_features(a)` | ~2453 | `features/shape.py` | numpy, safex |
| `extract_shape_features_8(a)` | ~2453+ | `features/shape.py` | numpy, safex |
| `safex(x)` | ~2453+ | `features/utils.py` | numpy |
| `extract_universal_features(row, areacols)` | ~2652 | `features/universal.py` | numpy, safex |
| `extractfeaturesfast(row, areacols)` | ~2852 | `features/fast.py` | numpy, scipy.stats, safex |

### 6I. Model Training Functions (→ `models/`)

| Function | Line | Module | Dependencies |
|----------|------|--------|--------------|
| `train_three_stage(df, areacols, labelcol, groups, ...)` | ~2253 | `models/three_stage.py` | xgb, CatBoostClassifier, ADASYN, StandardScaler, SelectFromModel, RandomForestClassifier, build_feature_frame_three_stage, extract_discriminative_features |
| `train_generalized_universal_model(df, labelcol, areacols, ...)` | ~2453+ | `models/universal.py` | xgb, extract_universal_features, compute_sample_weight |
| `train_single_xgb(df, labelcol, tscols, ...)` | ~2852+ | `models/single_xgb.py` | xgb, extractfeaturesfast, compute_sample_weight |
| `train_custom_template(df, labelcol, tscols, ...)` | ~3052 | `models/custom.py` | LogisticRegression, SVC, KNN, RandomForestClassifier, xgb, StandardScaler |

### 6J. Plotting Functions (→ `ui/plots.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `plot_confusion(cm, classes, acc)` | ~3052+ | matplotlib, seaborn |
| `plot_feature_importance_rf(rfselector, all_feature_names, topk)` | ~3052+ | matplotlib, numpy |
| `plot_actual_vs_pred(actual_counts, pred_counts, acc)` | ~3252 | matplotlib |

### 6K. HTML Utilities (→ `ui/render.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `error_html(message)` | ~3252+ | html |
| `success_summary_html(rows, label_col, ts_cols, T, classes)` | ~3252+ | html |

### 6L. Dataset Validation (→ `data/validation.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `validate_uploaded_dataset(upload_file, drive_path, st, require_label)` | ~3252+ | read_table_path, preprocess_dataset_to_unified, success_summary_html, error_html, _safe_source_filename |
| `validate_operational_dataset(upload_file, drive_path, st)` | ~3252+ | read_table_path, preprocess_dataset_to_unified, success_summary_html, error_html, _safe_source_filename |

### 6M. State Management & Experiment History (→ `state.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `new_state()` | ~3452+ | — |
| `history_df(st)` | ~3452+ | pandas |
| `recompute_best(history)` | ~3452+ | — |
| `undo_to_experiment(st, exp_id)` | ~3452+ | history_df, plot_confusion, plot_feature_importance_rf, plot_actual_vs_pred, render_performance_dashboard, render_classification_report |

### 6N. Experiment Runner (→ `models/runner.py` or `app.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `run_experiment(dataset_name, st, model_kind, ...)` | ~3852 | train_three_stage, train_generalized_universal_model, train_single_xgb, train_custom_template, build_feature_frame_three_stage, extract_discriminative_features, extract_universal_features, extractfeaturesfast, plot_confusion, plot_actual_vs_pred, plot_feature_importance_rf, render_performance_dashboard, render_classification_report, recompute_best |

### 6O. Water Quality Backend (→ `models/wq_models.py`, `data/wq_loader.py`)

| Function | Line | Module | Dependencies |
|----------|------|--------|--------------|
| `wq_load_excel(file_path_or_obj, sheet_wq, sheet_wsa)` | ~4052+ | `data/wq_loader.py` | pandas, openpyxl |
| `wq_detect_columns(df_wq, df_wsa)` | ~4052+ | `data/wq_loader.py` | — |
| `wq_read_data(upload_file, drive_link, sheet_wq, sheet_wsa)` | ~4052+ | `data/wq_loader.py` | wq_load_excel, download_drive_link |
| `wq_model1_run(df_wq, ndci_cols, ndti_cols, ...)` | ~4052+ | `models/wq_model1.py` | StandardScaler, PCA, KMeans, silhouette_score, calinski_harabasz_score, matplotlib |
| `dtw_distance(ts1, ts2)` | 4388 | `models/wq_model2.py` | numpy |
| `wq_model2_run(df_result, ndci_cols)` | 4399 | `models/wq_model2.py` | dtw_distance, pandas, matplotlib, seaborn |
| `wq_model3_run(df_wq, df_wsa, ndci_cols, ndti_cols, wsa_cols, ...)` | 4483 | `models/wq_model3.py` | StandardScaler, PCA, KMeans, silhouette_score, calinski_harabasz_score, matplotlib |
| `render_wq_summary_html(df, scores, model_name)` | 4602 | `ui/render.py` | html |
| `wq_new_state()` | 4663 | `state.py` | — |
| `get_wq_model_details(model_name)` | 4681 | `ui/render.py` | _safe_joblib_load, WQ_OPERATIONAL_MODELS, _USER_TRAINED_WQ_MODELS |
| `wq_validate_data(upload_file, drive_link, sheet_wq, sheet_wsa, st)` | 4757 | `data/wq_loader.py` | wq_read_data, wq_detect_columns, error_html, success_summary_html, _safe_source_filename |
| `wq_run_analysis(model_choice, ...)` | 4808 | `models/wq_runner.py` | wq_model1_run, wq_model2_run, wq_model3_run, render_wq_summary_html, error_html, _result_csv_path |
| `wq_operational_predict(model_key, ...)` | 4908 | `models/wq_operational.py` | wq_read_data, wq_detect_columns, _safe_joblib_load, render_wq_summary_html, _result_csv_path |

### 6P. Operational Prediction Helpers (→ `models/operational.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `load_pretrained_model(model_path)` | 7990 | joblib, _OPERATIONAL_MODEL_CACHE |
| `get_model_details(model_name)` | 8004 | _operational_model_path, load_pretrained_model, OPERATIONAL_MODEL_METADATA, extract_trained_classes_from_bundle |
| `_operational_model_path(model_name)` | 8109 | _USER_TRAINED_MODELS, OPERATIONAL_MODELS |
| `_normalise_model_class_values(values)` | 8116 | normalize_status |
| `extract_trained_classes_from_bundle(bundle, separate_encoder)` | 8135 | _normalise_model_class_values, normalize_status |
| `get_operational_model_trained_classes(model_name)` | 8202 | _operational_model_path, load_pretrained_model, extract_trained_classes_from_bundle, OPERATIONAL_LABEL_ENCODERS |
| `update_operational_model_ui(model_name)` | 8230 | get_model_details, get_operational_model_trained_classes |
| `_operational_pipeline_name(bundle, model_name)` | 8270 | OPERATIONAL_MODEL_METADATA |
| `_positive_int_or_none(value)` | 8293 | — |
| `_operational_training_length(bundle, model_name, pipeline)` | 8301 | _positive_int_or_none, OPERATIONAL_MODEL_METADATA |
| `_three_stage_groups_for_prediction(bundle)` | 8331 | — |
| `_ordered_ts_columns(df)` | 8351 | re |
| `_matrix_from_feature_dicts(feature_dicts, feature_names, pipeline)` | 8358 | numpy |
| `_expected_feature_count(estimator)` | 8382 | _positive_int_or_none |
| `_check_feature_count(X, expected, model_name, stage)` | 8395 | — |
| `_normalise_prediction_labels(values)` | 8404 | normalize_status, numpy |
| `_label_encoder_index(label_encoder, class_name)` | 8408 | normalize_status |
| `_build_non_shape_operational_matrix(bundle, df, feature_cols, pipeline, model_name)` | 8421 | _three_stage_groups_for_prediction, extract_discriminative_features, _matrix_from_feature_dicts, _expected_feature_count, _check_feature_count, extractfeaturesfast, extract_universal_features |
| `operational_predict(model_selector, uploaded_file, drive_path, st, target_classes)` | 8460 | _operational_model_path, load_pretrained_model, extract_trained_classes_from_bundle, _normalise_model_class_values, _operational_pipeline_name, _operational_training_length, read_table_path, _safe_source_filename, preprocess_dataset_to_unified, _ordered_ts_columns, extract_shape_features_8, extract_shape_only_features, _check_feature_count, _expected_feature_count, _build_non_shape_operational_matrix, _label_encoder_index, _normalise_prediction_labels, _result_csv_path, generate_operational_results, accuracy_score, classification_report, confusion_matrix |
| `generate_operational_results(df, predictions, probabilities, model, target_classes)` | 8857 | html, DEFAULT_TARGET_CLASSES |

### 6Q. UI Refresh Helpers (→ `ui/refresh.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `_notify_refresh(message)` | 9200 | gr |
| `refresh_operational_models_ui(current_model)` | 9208 | rescan_pond_model_registry, _pond_model_choices, update_operational_model_ui |
| `refresh_wq_operational_models_ui(current_model)` | 9227 | rescan_wq_model_registry, _wq_model_choices, get_wq_model_details |
| `_dataset_refresh_status(label, count, logs)` | 9243 | html, time |
| `_pond_dataset_selection_updates(state, ds_names)` | 9265 | render_dataset_cards, ALL_KNOWN_CLASSES |
| `refresh_pond_datasets_ui(state, current_selection)` | 9293 | autoload_and_preprocess_all, _pond_dataset_selection_updates, _dataset_refresh_status |
| `_activate_wq_dataset(state, dataset_name)` | 9329 | render_wq_dataset_cards |
| `refresh_wq_datasets_ui(state, current_selection)` | 9356 | autoload_wq_datasets, _activate_wq_dataset, _dataset_refresh_status |

### 6R. UI Panel Toggles (→ `ui/toggles.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `toggle_model_panels(mk)` | 7958 | gr |
| `toggle_custom_template_panels(tname)` | 7976 | gr |

### 6S. State Initialization (→ `state.py`)

| Function | Line | Dependencies |
|----------|------|--------------|
| `init_state()` | 7937 | new_state, _preloaded_datasets, _wq_preloaded_datasets, _wq_preloaded, _wq_dataset_names |

---

## 7. Gradio UI Assembly (lines 9393–10565)

- `gr.Blocks(theme=theme, css=CSS, js=DROPDOWN_JS, ...)` as `demo` [9393]
- **State:** `st`, `pond_deployment_state`, `wq_deployment_state`
- **Outer Tab 1: Pond Classification Analysis** [9407]
  - Sub-tab: Advanced (training workspace) [9410]
    - Step 1: Data selection [9430]
    - Step 2: Target classes [9487]
    - Step 3: Model architecture [9503]
    - Configuration panels (Single XGBoost, Universal, Three-Stage, Custom) [9537–9685]
    - Training setup + run controls [9686–9716]
    - Results & deployment [9717–9778]
  - Sub-tab: Operational (prediction workspace) [9780]
    - Data upload [9796]
    - Model selection [9836]
    - Class selection [9862]
    - Run prediction [9880]
    - Results [9904–9935]
- **Outer Tab 2: Water Quality Assessment** [9938]
  - Sub-tab: Analysis Workspace [9941]
    - Dataset selection [9957]
    - Model selection [10022]
    - Parameters [10054]
    - Run + results [10073–10141]
  - Sub-tab: Operational Predict [10143]
    - Workbook upload [10159]
    - Model selection [10205]
    - Run + results [10229–10283]
- **Event Wiring** [10285–10563]
  - model_kind.change → toggle_model_panels
  - custom_template_name.change → toggle_custom_template_panels
  - train_btn.click → run_experiment
  - undo_btn.click → undo_to_experiment
  - validate_btn.click / upload_file.change → validate_uploaded_dataset
  - op_validate_btn.click / op_upload_file.change → validate_operational_dataset
  - dataset_dd.change → _pond_dataset_selection_updates
  - pond_refresh_datasets_btn.click → refresh_pond_datasets_ui
  - wq_dataset_dd.change → _activate_wq_dataset
  - wq_refresh_datasets_btn.click → refresh_wq_datasets_ui
  - deploy_btn.click → deploy_experiment_to_operational
  - save_pond_model_to_drive_btn.click → save_deployed_pond_model_to_drive
  - op_refresh_models_btn.click → refresh_operational_models_ui
  - op_model_selector.change → update_operational_model_ui
  - op_predict_btn.click → operational_predict
  - wq_validate_btn.click / wq_upload_file.change → wq_validate_data
  - wq_run_btn.click → wq_run_analysis
  - wq_op_validate_btn.click / wq_op_upload.change → wq_validate_data
  - wq_op_run_btn.click → wq_operational_predict
  - wq_deploy_btn.click → deploy_wq_model_to_operational
  - wq_save_model_to_drive_btn.click → save_deployed_wq_model_to_drive
  - wq_refresh_models_btn.click → refresh_wq_operational_models_ui
  - wq_op_model.change → get_wq_model_details
- **Launch:** `demo.queue(max_size=20).launch(share=True, debug=True)` [10565]

---

## 8. Proposed Module Mapping

```
pond_wq_app/
├── config.py              ← Constants, paths, ALL_KNOWN_CLASSES, DEFAULT_TARGET_CLASSES
├── state.py               ← new_state, init_state, history_df, recompute_best, undo_to_experiment, wq_new_state
├── registry.py            ← Model discovery, deployment, persistence, caches, registries
├── data/
│   ├── __init__.py
│   ├── loader.py          ← smart_read_csv, read_table_path, normalize_status, find_label_col, pick_areacols, pick_temporalcols, resample_series, preprocess_dataset_to_unified, autoload_and_preprocess_all
│   ├── wq_loader.py       ← wq_load_excel, wq_detect_columns, wq_read_data, autoload_wq_datasets
│   ├── drive_utils.py     ← _filename_from_content_disposition, _google_drive_file_id, _safe_source_filename, _result_csv_path, download_drive_link
│   └── validation.py      ← validate_uploaded_dataset, validate_operational_dataset, wq_validate_data
├── features/
│   ├── __init__.py
│   ├── utils.py           ← safex
│   ├── discriminative.py  ← extract_discriminative_features, build_feature_frame_three_stage
│   ├── shape.py           ← extract_shape_only_features, extract_shape_features_8
│   ├── universal.py       ← extract_universal_features
│   └── fast.py            ← extractfeaturesfast
├── models/
│   ├── __init__.py
│   ├── three_stage.py     ← train_three_stage
│   ├── universal.py       ← train_generalized_universal_model
│   ├── single_xgb.py      ← train_single_xgb
│   ├── custom.py          ← train_custom_template
│   ├── runner.py          ← run_experiment
│   ├── operational.py     ← All operational prediction helpers + operational_predict + generate_operational_results
│   ├── wq_model1.py       ← wq_model1_run
│   ├── wq_model2.py       ← dtw_distance, wq_model2_run
│   ├── wq_model3.py       ← wq_model3_run
│   └── wq_runner.py       ← wq_run_analysis, wq_operational_predict
├── ui/
│   ├── __init__.py
│   ├── css.py             ← CSS string
│   ├── js.py              ← DROPDOWN_JS string
│   ├── theme.py           ← theme, DATASET_REQUIREMENTS_MD
│   ├── cards.py           ← render_dataset_cards, render_wq_dataset_cards
│   ├── render.py          ← render_best_model_status, download_best_model, build_model_display_name, render_performance_dashboard, render_classification_report, parse_classification_report, plot_classification_report_safe, error_html, success_summary_html, render_wq_summary_html, get_wq_model_details, get_model_details
│   ├── plots.py           ← plot_confusion, plot_feature_importance_rf, plot_actual_vs_pred
│   ├── toggles.py         ← toggle_model_panels, toggle_custom_template_panels
│   ├── refresh.py         ← _notify_refresh, refresh_operational_models_ui, refresh_wq_operational_models_ui, _dataset_refresh_status, _pond_dataset_selection_updates, refresh_pond_datasets_ui, _activate_wq_dataset, refresh_wq_datasets_ui
│   └── blocks.py          ← Gradio UI assembly (gr.Blocks definition + event wiring)
├── app.py                 ← Entry point: imports everything, calls demo.launch()
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_registry.py
├── run_in_colab.ipynb     ← Thin Colab notebook
├── requirements.txt
└── README.md
```
