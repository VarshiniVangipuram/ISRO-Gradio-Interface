"""
models/custom.py
================
Custom template model trainer using raw TS columns as features.
Supports Logistic Regression, SVC, KNN, Random Forest, and simple XGBoost.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


def train_custom_template(df, labelcol, tscols,
                          test_size=0.2, random_state=42,
                          template_name="Logistic Regression",
                          use_scaler=True,
                          use_class_weight=True,
                          lr_C=1.0, lr_max_iter=200,
                          svc_C=1.0, svc_kernel="rbf", svc_gamma="scale",
                          knn_k=5,
                          rf_n=300, rf_depth=0,
                          cxgb_n=800, cxgb_lr=0.05, cxgb_depth=6):

    X = df[tscols].values.astype(float)
    y = df[labelcol].values

    le = LabelEncoder()
    yenc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, yenc, test_size=float(test_size), stratify=yenc, random_state=int(random_state)
    )

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    class_weight = "balanced" if use_class_weight else None

    template_name = str(template_name)

    if template_name == "Logistic Regression":
        model = LogisticRegression(
            C=float(lr_C),
            max_iter=int(lr_max_iter),
            class_weight=class_weight,
            n_jobs=None,
            multi_class="auto"
        )
    elif template_name == "SVC":
        model = SVC(
            C=float(svc_C),
            kernel=str(svc_kernel),
            gamma=str(svc_gamma),
            class_weight=class_weight,
            probability=False
        )
    elif template_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=int(knn_k))
    elif template_name == "Random Forest":
        md = None if int(rf_depth) <= 0 else int(rf_depth)
        model = RandomForestClassifier(
            n_estimators=int(rf_n),
            max_depth=md,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
    elif template_name == "XGBoost (simple)":
        model = xgb.XGBClassifier(
            n_estimators=int(cxgb_n),
            learning_rate=float(cxgb_lr),
            max_depth=int(cxgb_depth),
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown custom template: {template_name}")

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    macro = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, target_names=le.classes_, zero_division=0)

    classes = list(le.classes_)
    actual_counts = pd.Series(le.inverse_transform(yte)).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(le.inverse_transform(pred)).value_counts().reindex(classes, fill_value=0)

    params = {
        "template_name": template_name,
        "use_scaler": bool(use_scaler),
        "use_class_weight": bool(use_class_weight),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "lr_C": float(lr_C),
        "lr_max_iter": int(lr_max_iter),
        "svc_C": float(svc_C),
        "svc_kernel": str(svc_kernel),
        "svc_gamma": str(svc_gamma),
        "knn_k": int(knn_k),
        "rf_n": int(rf_n),
        "rf_depth": int(rf_depth),
        "cxgb_n": int(cxgb_n),
        "cxgb_lr": float(cxgb_lr),
        "cxgb_depth": int(cxgb_depth),
    }

    artifacts = {
        "model_kind": "Custom Models (Templates)",
        "template_name": template_name,
        "model": model,
        "scaler": scaler,
        "labelencoder": le,
        "featurenames": list(tscols),
        "selectedfeatures": list(tscols),
        "T": len(tscols),
        "pipeline": "custom_raw_ts",
        "trained_classes": list(le.classes_),
        "preprocessing_version": 2,
        "thresholds": {},
        "params": params
    }

    return {
        "acc": acc, "macro_f1": macro, "cm": cm, "report": report,
        "classes": classes, "actual_counts": actual_counts, "pred_counts": pred_counts,
        "artifacts": artifacts
    }
