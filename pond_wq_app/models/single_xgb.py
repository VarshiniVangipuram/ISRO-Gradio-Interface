"""
models/single_xgb.py
====================
Single XGBoost model using 76-feature extraction.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

from ..features.fast import extractfeaturesfast


def train_single_xgb(df, labelcol, tscols,
                     test_size=0.2, random_state=42,
                     use_tempered_weights=True, alpha=0.6, cap=3.0,
                     n_estimators=2500, learning_rate=0.03,
                     max_depth=8, min_child_weight=1, gamma=0.0,
                     subsample=0.9, colsample_bytree=0.9,
                     reg_lambda=2.0, reg_alpha=0.5,
                     early_stopping_rounds=50,
                     use_gpu=False,
                     dataset_name=""):

    feats = []
    for _, r in df.iterrows():
        d = extractfeaturesfast(r, tscols)
        d["label"] = r[labelcol]
        feats.append(d)

    fdf = pd.DataFrame(feats).replace([np.inf, -np.inf], 0).fillna(0)
    X = fdf.drop(columns=["label"]).values
    y = fdf["label"].values

    le = LabelEncoder()
    yenc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, yenc, test_size=float(test_size), stratify=yenc, random_state=int(random_state)
    )

    sw = None
    if use_tempered_weights:
        sw = compute_sample_weight(class_weight="balanced", y=ytr)
        sw = np.minimum(sw * float(alpha), float(cap))

    # Town4 switch (optional but matches your original behavior)
    if "town4" in str(dataset_name).lower():
        max_depth = int(max_depth) if int(max_depth) <= 6 else 6
        min_child_weight = max(int(min_child_weight), 2)
        gamma = max(float(gamma), 0.5)

    model = xgb.XGBClassifier(
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
        tree_method="gpu_hist" if use_gpu else "hist",
        early_stopping_rounds=int(early_stopping_rounds)
    )

    if sw is None:
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    else:
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xte, yte)], verbose=False)

    pred = model.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    macro = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, target_names=le.classes_, zero_division=0)

    classes = list(le.classes_)
    actual_counts = pd.Series(le.inverse_transform(yte)).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(le.inverse_transform(pred)).value_counts().reindex(classes, fill_value=0)

    artifacts = {
        "model_kind": "Single XGBoost",
        "xgb_model": model,
        "labelencoder": le,
        "featurenames": list(fdf.drop(columns=["label"]).columns),
        "selectedfeatures": list(fdf.drop(columns=["label"]).columns),
        "T": len(tscols),
        "pipeline": "single_xgb_fast",
        "trained_classes": list(le.classes_),
        "preprocessing_version": 2,
        "thresholds": {},
        "params": {
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
            "use_tempered_weights": bool(use_tempered_weights),
            "alpha": float(alpha),
            "cap": float(cap),
            "use_gpu": bool(use_gpu),
        }
    }

    return {
        "acc": acc, "macro_f1": macro, "cm": cm, "report": report,
        "classes": classes, "actual_counts": actual_counts, "pred_counts": pred_counts,
        "artifacts": artifacts
    }
