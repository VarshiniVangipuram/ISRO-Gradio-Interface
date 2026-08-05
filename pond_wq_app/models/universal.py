"""
models/universal.py
===================
Universal Generalized Ensemble model using T-agnostic fractional features.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

from ..features.universal import extract_universal_features


def train_generalized_universal_model(
        df, labelcol, areacols,
        test_size=0.2, random_state=42,
        use_tempered_weights=True, alpha=0.6, cap=3.0,
        n_estimators=1000, learning_rate=0.05,
        max_depth=6, min_child_weight=1.0, gamma=0.0,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.0,
        early_stopping_rounds=50,
        use_gpu=False):
    """
    Train the Universal Generalized Ensemble using fractional/T-agnostic
    features. All important XGBoost settings are supplied by the Advanced UI.
    """
    feats = []
    for _, r in df.iterrows():
        d = extract_universal_features(r, areacols)
        d["label"] = r[labelcol]
        feats.append(d)

    fdf = pd.DataFrame(feats).replace([np.inf, -np.inf], 0).fillna(0)
    feature_names = list(fdf.drop(columns=["label"]).columns)
    X = fdf[feature_names].values.astype(float)
    y = fdf["label"].values

    le = LabelEncoder()
    yenc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, yenc,
        test_size=float(test_size),
        stratify=yenc,
        random_state=int(random_state)
    )

    sample_weight = None
    if bool(use_tempered_weights):
        sample_weight = compute_sample_weight(class_weight="balanced", y=ytr)
        sample_weight = np.minimum(
            sample_weight * float(alpha),
            float(cap)
        )

    early_stopping_rounds = int(early_stopping_rounds)
    model_kwargs = dict(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=int(random_state),
        tree_method="gpu_hist" if bool(use_gpu) else "hist",
    )
    if early_stopping_rounds > 0:
        model_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBClassifier(**model_kwargs)

    fit_kwargs = {
        "eval_set": [(Xte, yte)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(Xtr, ytr, **fit_kwargs)

    pred = model.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    macro = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred)
    report = classification_report(
        yte, pred, target_names=le.classes_, zero_division=0
    )
    classes = list(le.classes_)
    actual_counts = pd.Series(le.inverse_transform(yte)).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(le.inverse_transform(pred)).value_counts().reindex(classes, fill_value=0)

    params = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "use_tempered_weights": bool(use_tempered_weights),
        "alpha": float(alpha),
        "cap": float(cap),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "gamma": float(gamma),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "early_stopping_rounds": int(early_stopping_rounds),
        "use_gpu": bool(use_gpu),
    }

    artifacts = {
        "model_kind": "Universal Generalized Ensemble",
        "xgb_model": model,
        "labelencoder": le,
        "featurenames": feature_names,
        "selectedfeatures": feature_names,
        "is_universal": True,
        "pipeline": "universal",
        "T_original": len(areacols),
        "T": len(areacols),
        "trained_classes": list(le.classes_),
        "preprocessing_version": 3,
        "thresholds": {},
        "params": params,
    }

    return {
        "acc": acc,
        "macro_f1": macro,
        "cm": cm,
        "report": report,
        "classes": classes,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "artifacts": artifacts,
    }
