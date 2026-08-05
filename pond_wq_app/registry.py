"""
registry.py
===========
Model discovery, deployment, persistence, and caching for both
Pond Classification and Water Quality models.

* Scans Google Drive folders for operational model bundles.
* Manages session-only deployments and optional Drive persistence.
* Caches loaded models to avoid repeated PKL reads.
* Builds display names following the established naming convention.
"""

import os
import re
import time
import html
import joblib
from pathlib import Path

import numpy as np
import gradio as gr

from .config import (
    POND_OPERATIONAL_MODELS_FOLDER,
    WQ_OPERATIONAL_MODELS_FOLDER,
    POND_SESSION_DEPLOYMENTS_FOLDER,
    WQ_SESSION_DEPLOYMENTS_FOLDER,
    OPERATIONAL_MODELS,
    OPERATIONAL_LABEL_ENCODERS,
    OPERATIONAL_MODEL_METADATA,
    WQ_OPERATIONAL_MODELS,
    _USER_TRAINED_MODELS,
    _USER_TRAINED_WQ_MODELS,
    _OPERATIONAL_MODEL_CACHE,
    _WQ_OPERATIONAL_MODEL_CACHE,
)
from .data.loader import normalize_status


# -----------------------------
# Safe loading helpers
# -----------------------------
def _safe_joblib_load(path):
    """Load a trusted Drive artifact and return (object, error)."""
    try:
        return joblib.load(str(path)), None
    except Exception as exc:
        return None, str(exc)


def _is_label_encoder_filename(filename):
    name = str(filename).lower().replace("-", "_")
    return "labelencoder" in name or "label_encoder" in name


# -----------------------------
# Legacy helpers
# -----------------------------
def _legacy_pond_display_name(filename):
    """Preserve the familiar display names for the two legacy USO models."""
    stem = Path(filename).stem
    lower = stem.lower()

    if "pond_shapeonly_xgb" in lower:
        depth = re.search(r"_d(\d+)", stem, flags=re.I)
        lr = re.search(r"_lr(\d+)", stem, flags=re.I)
        nests = re.search(r"_(?:est|nest)(\d+)", stem, flags=re.I)
        acc = re.search(r"_acc(\d+)", stem, flags=re.I)
        d = depth.group(1) if depth else "6"
        l = lr.group(1) if lr else "003"
        n = nests.group(1) if nests else "2000"
        a = acc.group(1) if acc else "89"
        return f"USO_XGB_d{d}_lr{l}_nest{n}_Acc{a}"

    return stem


def _legacy_pond_pipeline(filename):
    lower = str(filename).lower()
    if "pond_shapeonly_xgb" in lower:
        return "shape_only_13"
    if "uso_5_xgb" in lower:
        return "shape_only_8"
    return None


def _find_matching_encoder(model_path, encoder_paths):
    """Pair the existing separate encoder files by their filename pattern."""
    model_name = model_path.name.lower()

    if "pond_shapeonly_xgb" in model_name:
        matches = [
            p for p in encoder_paths
            if "pond_shapeonly" in p.name.lower()
            and _is_label_encoder_filename(p.name)
        ]
        return matches[0] if matches else None

    if "uso_5_xgb" in model_name:
        acc_match = re.search(r"acc(\d+)", model_name, flags=re.I)
        acc = acc_match.group(1) if acc_match else None
        exact = [
            p for p in encoder_paths
            if "uso_5" in p.name.lower()
            and _is_label_encoder_filename(p.name)
            and (acc is None or f"acc{acc}" in p.name.lower())
        ]
        if exact:
            return exact[0]
        fallback = [
            p for p in encoder_paths
            if "uso_5" in p.name.lower()
            and _is_label_encoder_filename(p.name)
        ]
        return fallback[0] if len(fallback) == 1 else None

    return None


def _looks_like_pond_model(obj):
    if isinstance(obj, dict):
        model_keys = {
            "xgb_model", "model", "rejuv_detector", "newf_detector",
            "persistextinct_clf", "classifier", "estimator"
        }
        return bool(model_keys.intersection(obj.keys()))
    return hasattr(obj, "predict")


# -----------------------------
# Pond model discovery
# -----------------------------
def discover_pond_operational_models():
    """
    Scan Pond Operational_Active.

    Supports:
      * the current two legacy USO model files with separate encoders;
      * self-contained bundles deployed from the Advanced UI;
      * compatible self-contained custom bundles placed in the folder manually.
    """
    POND_OPERATIONAL_MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    registry = {}
    encoder_registry = {}
    metadata_registry = {}
    logs = []

    all_pkls = sorted(POND_OPERATIONAL_MODELS_FOLDER.glob("*.pkl"))
    encoder_paths = [p for p in all_pkls if _is_label_encoder_filename(p.name)]
    model_paths = [p for p in all_pkls if not _is_label_encoder_filename(p.name)]

    for model_path in model_paths:
        obj, error = _safe_joblib_load(model_path)
        if error:
            logs.append(f"Skipped {model_path.name}: {error}")
            continue
        if not _looks_like_pond_model(obj):
            logs.append(f"Skipped {model_path.name}: not a supported Pond model artifact")
            continue

        legacy_pipeline = _legacy_pond_pipeline(model_path.name)
        if not isinstance(obj, dict) and legacy_pipeline is None:
            logs.append(
                f"Skipped {model_path.name}: raw estimator has no saved pipeline metadata"
            )
            continue

        if isinstance(obj, dict):
            display_name = str(obj.get("display_name") or _legacy_pond_display_name(model_path.name))
            pipeline = str(
                obj.get("pipeline")
                or obj.get("preprocessing_pipeline")
                or legacy_pipeline
                or ""
            ).strip() or None
            T_value = obj.get("T", obj.get("T_original"))
        else:
            display_name = _legacy_pond_display_name(model_path.name)
            pipeline = legacy_pipeline
            T_value = None

        base_name = display_name
        counter = 2
        while display_name in registry:
            display_name = f"{base_name}_{counter}"
            counter += 1

        registry[display_name] = str(model_path)
        metadata_registry[display_name] = {
            "pipeline": pipeline,
            "T": T_value,
            "filename": model_path.name,
        }

        encoder_path = _find_matching_encoder(model_path, encoder_paths)
        if encoder_path is not None:
            encoder_registry[display_name] = str(encoder_path)

        logs.append(
            f"Loaded {display_name} | file={model_path.name} | "
            f"pipeline={pipeline or 'bundle inference'}"
        )

    if not model_paths:
        logs.append(f"No Pond model PKL files found in {POND_OPERATIONAL_MODELS_FOLDER}")

    return registry, encoder_registry, metadata_registry, logs


# -----------------------------
# WQ model discovery
# -----------------------------
def _wq_display_name(bundle, model_path):
    """Use saved display name when present; otherwise use the PKL filename."""
    if isinstance(bundle, dict):
        saved = bundle.get("display_name")
        if saved:
            return str(saved)
    return model_path.stem


def _looks_like_wq_model(obj):
    if not isinstance(obj, dict):
        return False
    name = str(obj.get("model_name", ""))
    return (
        "Model1" in name
        or "Model2" in name
        or "Model3" in name
        or obj.get("artifact_type") == "wq_operational_model_bundle"
    )


def discover_wq_operational_models():
    """Scan Water Quality Operational_Active for compatible WQ bundles."""
    WQ_OPERATIONAL_MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    registry = {}
    logs = []

    for model_path in sorted(WQ_OPERATIONAL_MODELS_FOLDER.glob("*.pkl")):
        bundle, error = _safe_joblib_load(model_path)
        if error:
            logs.append(f"Skipped {model_path.name}: {error}")
            continue
        if not _looks_like_wq_model(bundle):
            logs.append(f"Skipped {model_path.name}: not a supported Water Quality bundle")
            continue

        display_name = _wq_display_name(bundle, model_path)
        base_name = display_name
        counter = 2
        while display_name in registry:
            display_name = f"{base_name}_{counter}"
            counter += 1

        registry[display_name] = str(model_path)
        logs.append(f"Loaded {display_name} | file={model_path.name}")

    if not registry:
        logs.append(f"No Water Quality model bundles found in {WQ_OPERATIONAL_MODELS_FOLDER}")

    return registry, logs


# -----------------------------
# Registry rescans
# -----------------------------
def rescan_pond_model_registry():
    global OPERATIONAL_MODELS
    global OPERATIONAL_LABEL_ENCODERS
    global OPERATIONAL_MODEL_METADATA
    global _OPERATIONAL_MODEL_CACHE

    (
        OPERATIONAL_MODELS,
        OPERATIONAL_LABEL_ENCODERS,
        OPERATIONAL_MODEL_METADATA,
        logs,
    ) = discover_pond_operational_models()
    _OPERATIONAL_MODEL_CACHE.clear()
    return logs


def rescan_wq_model_registry():
    global WQ_OPERATIONAL_MODELS
    global _WQ_OPERATIONAL_MODEL_CACHE

    WQ_OPERATIONAL_MODELS, logs = discover_wq_operational_models()
    _WQ_OPERATIONAL_MODEL_CACHE.clear()
    return logs


# -----------------------------
# Unique destination helpers
# -----------------------------
def _unique_drive_destination(base_name, folder):
    """Return a unique display name and file path without overwriting a model."""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_name)).strip("_.")
    clean = clean or "Model"
    candidate = clean
    counter = 2
    path = Path(folder) / f"{candidate}.pkl"
    while path.exists():
        candidate = f"{clean}_{counter}"
        path = Path(folder) / f"{candidate}.pkl"
        counter += 1
    return candidate, path


def _unique_session_destination(base_name, folder, occupied_names=None):
    """Create a unique session-only name/path without changing Drive."""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_name)).strip("_.") or "Model"
    occupied = set(occupied_names or [])
    Path(folder).mkdir(parents=True, exist_ok=True)

    candidate = clean
    counter = 2
    path = Path(folder) / f"{candidate}.pkl"
    while candidate in occupied or path.exists():
        candidate = f"{clean}_{counter}"
        path = Path(folder) / f"{candidate}.pkl"
        counter += 1
    return candidate, path


# -----------------------------
# Model choice lists
# -----------------------------
def _pond_model_choices():
    """Drive models first, followed by models deployed in this runtime."""
    return list(dict.fromkeys([
        *OPERATIONAL_MODELS.keys(),
        *_USER_TRAINED_MODELS.keys(),
    ]))


def _wq_model_choices():
    """Drive WQ models first, followed by models deployed in this runtime."""
    return list(dict.fromkeys([
        *WQ_OPERATIONAL_MODELS.keys(),
        *_USER_TRAINED_WQ_MODELS.keys(),
    ]))


# -----------------------------
# Model naming utility
# -----------------------------
def build_model_display_name(entry):
    """
    Build a concise, informative display name for a trained model,
    following the same convention as pre-existing operational models.
    e.g.:  SXGB_d8_lr003_nest2500_Acc87_Town2
           3S_XGB_CB_d10_nest400_Acc89_Town2Town4
           GUE_nest1000_d6_Acc81_Stats
           LR_C1_Acc75_Town2
    """
    mk = entry.get("model_kind", "")
    params = entry.get("params", {})
    datasets = entry.get("dataset", "")
    acc = entry.get("acc", 0) * 100
    exp_id = entry.get("experiment_id", 0)

    ds_parts = [os.path.splitext(d.strip())[0] for d in datasets.split(",")]
    def _abbr(s):
        s = re.sub(r'^Uploaded:\s*', '', s)
        return re.sub(r'[^a-zA-Z0-9]', '', s)[:10]
    if len(ds_parts) == 1:
        ds_abbr = _abbr(ds_parts[0])
    else:
        ds_abbr = "".join([_abbr(d)[:6] for d in ds_parts[:2]])

    acc_str = f"Acc{int(round(acc))}"

    if mk.startswith("Single XGBoost"):
        depth = int(params.get("x_max_depth", 8))
        lr = float(params.get("x_lr", 0.03))
        n = int(params.get("x_n_estimators", 2500))
        lr_str = f"{lr:.3f}".replace("0.", "").replace(".", "")
        name = f"SXGB_d{depth}_lr{lr_str}_nest{n}_{acc_str}"

    elif mk.startswith("Three-Stage"):
        s1 = params.get("s1", {}) if isinstance(params.get("s1"), dict) else {}
        depth = int(s1.get("depth", 10))
        n = int(s1.get("n", 400))
        name = f"3S_XGB_CB_d{depth}_nest{n}_{acc_str}"

    elif mk.startswith("Generalized Universal"):
        n = int(params.get("u_n_estimators", 1000))
        depth = int(params.get("u_max_depth", 6))
        lr = float(params.get("u_lr", 0.05))
        lr_str = f"{lr:.3f}".replace("0.", "").replace(".", "")
        name = f"GUE_d{depth}_lr{lr_str}_nest{n}_{acc_str}"

    elif mk.startswith("Custom Models"):
        tname = params.get("custom_template_name", "Custom")
        if tname == "Logistic Regression":
            c = params.get("custom_lr_C", 1.0)
            c_str = re.sub(r'\.0$', '', str(round(float(c), 2)))
            name = f"LR_C{c_str}_{acc_str}"
        elif tname == "SVC":
            kernel = params.get("custom_svc_kernel", "rbf")
            c = params.get("custom_svc_C", 1.0)
            c_str = re.sub(r'\.0$', '', str(round(float(c), 2)))
            name = f"SVC_{kernel}_C{c_str}_{acc_str}"
        elif tname == "KNN":
            k = int(params.get("custom_knn_k", 5))
            name = f"KNN_k{k}_{acc_str}"
        elif tname == "Random Forest":
            n = int(params.get("custom_rf_n", 300))
            depth = int(params.get("custom_rf_depth", 0))
            d_str = "Auto" if depth <= 0 else str(depth)
            name = f"RF_n{n}_d{d_str}_{acc_str}"
        elif tname == "XGBoost (simple)":
            depth = int(params.get("custom_cxgb_depth", 6))
            lr = float(params.get("custom_cxgb_lr", 0.05))
            n = int(params.get("custom_cxgb_n", 800))
            lr_str = f"{lr:.3f}".replace("0.", "").replace(".", "")
            name = f"CXGB_d{depth}_lr{lr_str}_nest{n}_{acc_str}"
        else:
            name = f"Custom_Exp{exp_id}_{acc_str}"
    else:
        name = f"Model_Exp{exp_id}_{acc_str}"

    if ds_abbr:
        name = f"{name}_{ds_abbr}"

    return name


# -----------------------------
# Model loading and caching
# -----------------------------
def load_pretrained_model(model_path):
    """Load and cache a trusted pre-trained model from Drive."""
    try:
        if not model_path or not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        key = str(model_path)
        if key in _OPERATIONAL_MODEL_CACHE:
            return _OPERATIONAL_MODEL_CACHE[key], None
        model = joblib.load(key)
        _OPERATIONAL_MODEL_CACHE[key] = model
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# -----------------------------
# Operational model class discovery
# -----------------------------
def _operational_model_path(model_name):
    """Resolve a static or user-deployed Operational model path."""
    if model_name in _USER_TRAINED_MODELS:
        return _USER_TRAINED_MODELS[model_name]
    return OPERATIONAL_MODELS.get(model_name)


def _normalise_model_class_values(values):
    """Return unique, normalized human-readable class names in model order."""
    cleaned = []
    for value in list(values):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")

        if isinstance(value, (int, np.integer)):
            continue
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            continue

        label = normalize_status(value)
        if label and label not in {"nan", "none", "null", ""} and label not in cleaned:
            cleaned.append(label)
    return cleaned


def extract_trained_classes_from_bundle(bundle, separate_encoder=None):
    """
    Extract the actual classes learned by an Operational model.

    Priority:
      1. Separate LabelEncoder
      2. LabelEncoder stored inside a bundle/object
      3. Explicit class-name metadata
      4. Estimator classes_ only when they are already human-readable labels
    """
    encoder_candidates = []
    explicit_class_candidates = []
    estimator_candidates = []

    if separate_encoder is not None:
        encoder_candidates.append(separate_encoder)

    if isinstance(bundle, dict):
        for key in ("labelencoder", "label_encoder", "encoder", "le"):
            candidate = bundle.get(key)
            if candidate is not None:
                encoder_candidates.append(candidate)

        for key in ("trained_classes", "class_names", "target_classes", "classes"):
            candidate = bundle.get(key)
            if candidate is not None:
                explicit_class_candidates.append(candidate)

        for key in (
            "xgb_model", "model", "rejuv_detector", "newf_detector",
            "persistextinct_clf", "classifier", "estimator"
        ):
            candidate = bundle.get(key)
            if candidate is not None:
                estimator_candidates.append(candidate)
    else:
        for attr in ("labelencoder", "label_encoder", "encoder", "le"):
            candidate = getattr(bundle, attr, None)
            if candidate is not None:
                encoder_candidates.append(candidate)
        estimator_candidates.append(bundle)

    for encoder in encoder_candidates:
        classes = getattr(encoder, "classes_", None)
        if classes is not None:
            cleaned = _normalise_model_class_values(classes)
            if cleaned:
                return cleaned

    for candidate in explicit_class_candidates:
        try:
            cleaned = _normalise_model_class_values(candidate)
        except TypeError:
            cleaned = []
        if cleaned:
            return cleaned

    for estimator in estimator_candidates:
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            cleaned = _normalise_model_class_values(classes)
            if cleaned:
                return cleaned

    return []


def get_operational_model_trained_classes(model_name):
    """Load the selected model and return only the classes it was trained on."""
    model_path = _operational_model_path(model_name)
    if not model_path:
        return [], f"Model path is not configured for '{model_name}'."

    bundle, error = load_pretrained_model(model_path)
    if error:
        return [], error

    separate_encoder = None
    encoder_path = OPERATIONAL_LABEL_ENCODERS.get(model_name)
    if encoder_path:
        separate_encoder, encoder_error = load_pretrained_model(encoder_path)
        if encoder_error:
            return [], f"Label encoder load error: {encoder_error}"

    classes = extract_trained_classes_from_bundle(bundle, separate_encoder)
    if not classes:
        return [], (
            "The selected model does not expose readable trained class names. "
            "Save a LabelEncoder as bundle['labelencoder'] or configure "
            "OPERATIONAL_LABEL_ENCODERS for this model."
        )

    return classes, None


# -----------------------------
# Pond deployment
# -----------------------------
def deploy_experiment_to_operational(state, exp_id):
    """
    Deploy an Advanced experiment into Operational mode for the current Colab
    runtime only. Nothing is written to Google Drive here.
    """
    global _USER_TRAINED_MODELS

    # Deferred import to avoid circular dependency
    from .ui.render import update_operational_model_ui

    history = (state or {}).get("history", [])
    disabled_save = gr.update(
        interactive=False,
        value="☁️ Save Deployed Model to Drive",
    )

    if not history:
        return (
            gr.update(), gr.update(), gr.update(),
            "<p style='color:#b45309;font-weight:600;'>⚠️ No experiments to deploy. Train a model first.</p>",
            None,
            disabled_save,
        )

    try:
        if exp_id is None or exp_id == "" or (
            isinstance(exp_id, (float, np.floating)) and np.isnan(exp_id)
        ):
            exp_id_int = 0
        else:
            exp_id_int = int(exp_id)
    except (TypeError, ValueError):
        exp_id_int = 0

    if exp_id_int > 0:
        target = next(
            (h for h in history if h["experiment_id"] == exp_id_int),
            None,
        )
        if target is None:
            return (
                gr.update(), gr.update(), gr.update(),
                f"<p style='color:red;'>❌ Experiment ID {exp_id_int} not found.</p>",
                None,
                disabled_save,
            )
    else:
        target = max(history, key=lambda h: h["acc"])

    original_bundle = target.get("bundle")
    if original_bundle is None:
        return (
            gr.update(), gr.update(), gr.update(),
            "<p style='color:red;'>❌ No model bundle in this experiment.</p>",
            None,
            disabled_save,
        )

    base_name = build_model_display_name(target)
    occupied = set(_pond_model_choices())
    display_name, filepath = _unique_session_destination(
        base_name,
        POND_SESSION_DEPLOYMENTS_FOLDER,
        occupied,
    )

    if isinstance(original_bundle, dict):
        bundle_to_save = dict(original_bundle)
    else:
        bundle_to_save = {"model": original_bundle}

    encoder = (
        bundle_to_save.get("labelencoder")
        or bundle_to_save.get("label_encoder")
        or bundle_to_save.get("le")
    )
    classes = (
        list(getattr(encoder, "classes_", []))
        if encoder is not None
        else list(bundle_to_save.get("trained_classes", []))
    )

    bundle_to_save.update({
        "artifact_type": "operational_model_bundle",
        "display_name": display_name,
        "trained_classes": classes,
        "classes": classes,
        "deployed_from": "Advanced Training UI — current session",
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": int(target.get("experiment_id", 0)),
        "training_accuracy": float(target.get("acc", 0.0)),
        "training_macro_f1": float(target.get("macro_f1", 0.0)),
        "saved_to_drive": False,
    })

    try:
        Path(POND_SESSION_DEPLOYMENTS_FOLDER).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle_to_save, str(filepath))
    except Exception as exc:
        return (
            gr.update(), gr.update(), gr.update(),
            f"<p style='color:red;'>❌ Could not deploy model in this session: {html.escape(str(exc))}</p>",
            None,
            disabled_save,
        )

    _USER_TRAINED_MODELS[display_name] = str(filepath)
    _OPERATIONAL_MODEL_CACHE.pop(str(filepath), None)

    description, class_update = update_operational_model_ui(display_name)
    choices = _pond_model_choices()
    acc = float(target.get("acc", 0.0)) * 100.0

    deployment_record = {
        "model_type": "pond",
        "display_name": display_name,
        "filepath": str(filepath),
        "saved_to_drive": False,
    }

    status_html = f"""
    <div style='background:linear-gradient(135deg,#dbeafe,#eff6ff);
                border-left:4px solid #3b82f6;border-radius:12px;padding:18px;margin-top:10px;'>
        <div style='font-size:15px;font-weight:700;color:#1e3a8a;margin-bottom:8px;'>
            🚀 Deployed to Operational for this session
        </div>
        <div style='font-size:13px;color:#1e40af;line-height:1.8;'>
            <b>Model:</b> {html.escape(display_name)}<br>
            <b>Accuracy:</b> {acc:.2f}%<br>
            <b>Persistence:</b> Not saved to Google Drive yet.
        </div>
        <div style='margin-top:10px;padding:9px 12px;background:rgba(59,130,246,0.10);
                    border-radius:8px;font-size:12px;color:#1e3a8a;'>
            The model is ready in the Operational tab now. Click
            <b>☁️ Save Deployed Model to Drive</b> below only when you want it
            to remain available after the Colab runtime ends.
        </div>
    </div>
    """

    return (
        gr.update(choices=choices, value=display_name),
        description,
        class_update,
        status_html,
        deployment_record,
        gr.update(
            interactive=True,
            value="☁️ Save Deployed Model to Drive",
        ),
    )


def save_deployed_pond_model_to_drive(deployment_record):
    """Persist the most recently session-deployed Pond model to Drive."""
    global _USER_TRAINED_MODELS

    # Deferred import to avoid circular dependency
    from .ui.render import update_operational_model_ui

    if not deployment_record:
        return (
            gr.update(), gr.update(), gr.update(),
            "<p style='color:#b45309;font-weight:600;'>⚠️ Deploy a model first.</p>",
            deployment_record,
            gr.update(interactive=False),
        )

    if deployment_record.get("saved_to_drive"):
        return (
            gr.update(), gr.update(), gr.update(),
            "<p style='color:#047857;font-weight:600;'>✅ This deployed model is already saved to Drive.</p>",
            deployment_record,
            gr.update(interactive=False, value="✅ Saved to Drive"),
        )

    source_path = Path(str(deployment_record.get("filepath", "")))
    old_display_name = str(deployment_record.get("display_name", "Model"))

    if not source_path.exists():
        return (
            gr.update(), gr.update(), gr.update(),
            "<p style='color:red;'>❌ The session model file is no longer available. Deploy it again.</p>",
            deployment_record,
            gr.update(interactive=True),
        )

    try:
        bundle = joblib.load(str(source_path))
        if not isinstance(bundle, dict):
            bundle = {"model": bundle}

        drive_display_name, drive_path = _unique_drive_destination(
            old_display_name,
            POND_OPERATIONAL_MODELS_FOLDER,
        )
        bundle.update({
            "artifact_type": "operational_model_bundle",
            "display_name": drive_display_name,
            "saved_to_drive": True,
            "saved_to_drive_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployed_from": "Advanced Training UI",
        })

        Path(POND_OPERATIONAL_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, str(drive_path))
    except Exception as exc:
        return (
            gr.update(), gr.update(), gr.update(),
            f"<p style='color:red;'>❌ Could not save model to Drive: {html.escape(str(exc))}</p>",
            deployment_record,
            gr.update(interactive=True),
        )

    _USER_TRAINED_MODELS.pop(old_display_name, None)
    rescan_pond_model_registry()

    choices = _pond_model_choices()
    description, class_update = update_operational_model_ui(drive_display_name)
    updated_record = {
        **deployment_record,
        "display_name": drive_display_name,
        "drive_filepath": str(drive_path),
        "saved_to_drive": True,
    }

    status_html = f"""
    <div style='background:linear-gradient(135deg,#d1fae5,#ecfdf5);
                border-left:4px solid #10b981;border-radius:12px;padding:18px;margin-top:10px;'>
        <div style='font-size:15px;font-weight:700;color:#065f46;margin-bottom:8px;'>
            ☁️ Model saved to Google Drive
        </div>
        <div style='font-size:13px;color:#064e3b;line-height:1.8;'>
            <b>Model:</b> {html.escape(drive_display_name)}<br>
            <b>File:</b> <code>{html.escape(drive_path.name)}</code><br>
            <b>Folder:</b> <code>{html.escape(str(POND_OPERATIONAL_MODELS_FOLDER))}</code>
        </div>
        <div style='margin-top:10px;font-size:12px;color:#047857;'>
            It will be discovered automatically the next time the application starts.
        </div>
    </div>
    """

    return (
        gr.update(choices=choices, value=drive_display_name),
        description,
        class_update,
        status_html,
        updated_record,
        gr.update(interactive=False, value="✅ Saved to Drive"),
    )


# -----------------------------
# WQ deployment
# -----------------------------
def _build_wq_drive_name(artifacts):
    """Build filenames using the same convention as the current WQ PKLs."""
    model_name = str(artifacts.get("model_name", "WQ_Model"))
    hp = artifacts.get("hyperparameters", {}) or {}
    metrics = artifacts.get("metrics", {}) or {}

    if "Model1" in model_name:
        k = int(hp.get("K", 0))
        pca_obj = artifacts.get("pca")
        pca_n = getattr(pca_obj, "n_components_", getattr(pca_obj, "n_components", 0))
        sil = float(metrics.get("silhouette", 0.0))
        ch = int(round(float(metrics.get("calinski_harabasz", 0.0))))
        return f"WQ_M1_PCAKMeans_K{k}_PCA{pca_n}_Sil{sil:.4f}_CH{ch}"

    if "Model2" in model_name:
        T = int(hp.get("T", 0))
        protos = int(hp.get("num_prototypes", len(artifacts.get("prototypes", []))))
        return f"WQ_M2_DTW_ShapeClassifier_T{T}_Protos{protos}"

    if "Model3" in model_name:
        k = int(hp.get("K", 0))
        pca_each = int(hp.get("pca_components_each", 0))
        sil = float(metrics.get("silhouette", 0.0))
        ch = int(round(float(metrics.get("calinski_harabasz", 0.0))))
        return f"WQ_M3_TripleStream_K{k}_PCA{pca_each}each_Sil{sil:.4f}_CH{ch}"

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "WQ_Model"


def deploy_wq_model_to_operational(state):
    """Deploy the latest WQ analysis for the current Colab runtime only."""
    global _USER_TRAINED_WQ_MODELS

    # Deferred import to avoid circular dependency
    from .ui.render import get_wq_model_details

    artifacts = (state or {}).get("wq_last_artifacts")
    disabled_save = gr.update(
        interactive=False,
        value="☁️ Save Deployed WQ Model to Drive",
    )

    if not artifacts:
        return (
            gr.update(), gr.update(),
            "<p style='color:#b45309;font-weight:600;margin-top:10px;'>⚠️ No analyzed WQ model to deploy. Run an analysis first.</p>",
            None,
            disabled_save,
        )

    base_name = _build_wq_drive_name(artifacts)
    display_name, filepath = _unique_session_destination(
        base_name,
        WQ_SESSION_DEPLOYMENTS_FOLDER,
        set(_wq_model_choices()),
    )

    bundle_to_save = dict(artifacts)
    bundle_to_save.update({
        "artifact_type": "wq_operational_model_bundle",
        "display_name": display_name,
        "deployed_from": "Water Quality Analysis UI — current session",
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "saved_to_drive": False,
    })

    try:
        Path(WQ_SESSION_DEPLOYMENTS_FOLDER).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle_to_save, str(filepath))
    except Exception as exc:
        return (
            gr.update(), gr.update(),
            f"<p style='color:red;margin-top:10px;'>❌ Session deployment error: {html.escape(str(exc))}</p>",
            None,
            disabled_save,
        )

    _USER_TRAINED_WQ_MODELS[display_name] = str(filepath)
    _WQ_OPERATIONAL_MODEL_CACHE.pop(str(filepath), None)
    choices = _wq_model_choices()
    description = get_wq_model_details(display_name)

    deployment_record = {
        "model_type": "water_quality",
        "display_name": display_name,
        "filepath": str(filepath),
        "saved_to_drive": False,
    }

    status_html = f"""
    <div style='background:linear-gradient(135deg,#dbeafe,#eff6ff);
                border-left:4px solid #3b82f6;border-radius:12px;padding:18px;margin-top:12px;'>
        <div style='font-size:15px;font-weight:700;color:#1e3a8a;margin-bottom:8px;'>
            🚀 WQ model deployed for this session
        </div>
        <div style='font-size:13px;color:#1e40af;line-height:1.8;'>
            <b>Model:</b> {html.escape(display_name)}<br>
            <b>Persistence:</b> Not saved to Google Drive yet.
        </div>
        <div style='margin-top:10px;padding:9px 12px;background:rgba(59,130,246,0.10);
                    border-radius:8px;font-size:12px;color:#1e3a8a;'>
            It is available in Operational Predict now. Use the cloud-save button
            below only when you want it to remain after the runtime ends.
        </div>
    </div>
    """

    return (
        gr.update(choices=choices, value=display_name),
        description,
        status_html,
        deployment_record,
        gr.update(
            interactive=True,
            value="☁️ Save Deployed WQ Model to Drive",
        ),
    )


def save_deployed_wq_model_to_drive(deployment_record):
    """Persist the most recently session-deployed WQ model to Drive."""
    global _USER_TRAINED_WQ_MODELS

    # Deferred import to avoid circular dependency
    from .ui.render import get_wq_model_details

    if not deployment_record:
        return (
            gr.update(), gr.update(),
            "<p style='color:#b45309;font-weight:600;'>⚠️ Deploy a WQ model first.</p>",
            deployment_record,
            gr.update(interactive=False),
        )

    if deployment_record.get("saved_to_drive"):
        return (
            gr.update(), gr.update(),
            "<p style='color:#047857;font-weight:600;'>✅ This WQ model is already saved to Drive.</p>",
            deployment_record,
            gr.update(interactive=False, value="✅ Saved to Drive"),
        )

    source_path = Path(str(deployment_record.get("filepath", "")))
    old_display_name = str(deployment_record.get("display_name", "WQ_Model"))

    if not source_path.exists():
        return (
            gr.update(), gr.update(),
            "<p style='color:red;'>❌ The session WQ model is no longer available. Deploy it again.</p>",
            deployment_record,
            gr.update(interactive=True),
        )

    try:
        bundle = joblib.load(str(source_path))
        if not isinstance(bundle, dict):
            raise ValueError("The deployed WQ artifact is not a dictionary bundle.")

        drive_display_name, drive_path = _unique_drive_destination(
            old_display_name,
            WQ_OPERATIONAL_MODELS_FOLDER,
        )
        bundle.update({
            "artifact_type": "wq_operational_model_bundle",
            "display_name": drive_display_name,
            "saved_to_drive": True,
            "saved_to_drive_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployed_from": "Water Quality Analysis UI",
        })

        Path(WQ_OPERATIONAL_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, str(drive_path))
    except Exception as exc:
        return (
            gr.update(), gr.update(),
            f"<p style='color:red;'>❌ Could not save WQ model to Drive: {html.escape(str(exc))}</p>",
            deployment_record,
            gr.update(interactive=True),
        )

    _USER_TRAINED_WQ_MODELS.pop(old_display_name, None)
    rescan_wq_model_registry()

    choices = _wq_model_choices()
    description = get_wq_model_details(drive_display_name)
    updated_record = {
        **deployment_record,
        "display_name": drive_display_name,
        "drive_filepath": str(drive_path),
        "saved_to_drive": True,
    }

    status_html = f"""
    <div style='background:linear-gradient(135deg,#d1fae5,#ecfdf5);
                border-left:4px solid #10b981;border-radius:12px;padding:18px;margin-top:12px;'>
        <div style='font-size:15px;font-weight:700;color:#065f46;margin-bottom:8px;'>
            ☁️ WQ model saved to Google Drive
        </div>
        <div style='font-size:13px;color:#064e3b;line-height:1.8;'>
            <b>Model:</b> {html.escape(drive_display_name)}<br>
            <b>File:</b> <code>{html.escape(drive_path.name)}</code><br>
            <b>Folder:</b> <code>{html.escape(str(WQ_OPERATIONAL_MODELS_FOLDER))}</code>
        </div>
    </div>
    """

    return (
        gr.update(choices=choices, value=drive_display_name),
        description,
        status_html,
        updated_record,
        gr.update(interactive=False, value="✅ Saved to Drive"),
    )
