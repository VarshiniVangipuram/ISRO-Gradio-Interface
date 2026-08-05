"""
models/three_stage.py
=====================
Three-Stage cascaded ensemble model for pond classification.

Stage 1: Rejuvenated detector (XGBoost)
Stage 2: Newly-formed detector (XGBoost)
Stage 3: Persistent vs Extinct classifier (CatBoost)
"""

import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import ADASYN, SMOTE

import xgboost as xgb
from catboost import CatBoostClassifier

from ..features.discriminative import build_feature_frame_three_stage


def train_three_stage(df, areacols, labelcol, groups,
                      test_size=0.2, random_state=42,
                      use_adasyn=True, adasyn_neighbors=3,
                      auto_optimize_thresholds=True,
                      s1_n=400, s1_depth=10, s1_lr=0.05, s1_spw=15.0,
                      s2_n=400, s2_depth=10, s2_lr=0.05, s2_spw=12.0,
                      s3_iter=400, s3_depth=8, s3_lr=0.05,
                      init_rth=0.30, init_nth=0.35,
                      fs_threshold="mean"):

    featuredf, featurenames, X, y = build_feature_frame_three_stage(df, areacols, labelcol, groups)

    le = LabelEncoder()
    yenc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, yenc, test_size=float(test_size), random_state=int(random_state), stratify=yenc
    )

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    rfselector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfselector.fit(Xtr_s, ytr)
    selector = SelectFromModel(rfselector, prefit=True, threshold=str(fs_threshold))
    Xtr_sel = selector.transform(Xtr_s)
    Xte_sel = selector.transform(Xte_s)
    selected_feature_names = [featurenames[i] for i, ok in enumerate(selector.get_support()) if ok]

    if use_adasyn:
        def safe_oversample(X, y, n_neighbors, random_state=42):
            try:
                return ADASYN(
                    random_state=random_state,
                    n_neighbors=int(n_neighbors),
                    sampling_strategy="auto"
                ).fit_resample(X, y)
            except ValueError:
                try:
                    return ADASYN(
                        random_state=random_state,
                        n_neighbors=int(n_neighbors),
                        sampling_strategy="minority"
                    ).fit_resample(X, y)
                except ValueError:
                    try:
                        return SMOTE(
                            random_state=random_state,
                            k_neighbors=max(1, int(n_neighbors)-1),
                            sampling_strategy="auto"
                        ).fit_resample(X, y)
                    except ValueError:
                        return X, y

        Xtr_sel, ytr = safe_oversample(Xtr_sel, ytr, adasyn_neighbors)

    def stage_adasyn(Xs, ybin):
        if not use_adasyn:
            return Xs, ybin
        ada = ADASYN(random_state=42, n_neighbors=2)
        try:
            return ada.fit_resample(Xs, ybin)
        except Exception:
            return Xs, ybin

    idx_rejuv = int(np.where(le.classes_ == "rejuvenated")[0][0])
    idx_newf = int(np.where(le.classes_ == "newly formed")[0][0])
    idx_pers = int(np.where(le.classes_ == "persistent")[0][0])
    idx_ext  = int(np.where(le.classes_ == "extinct")[0][0])

    # Stage 1
    ytr_r = (ytr == idx_rejuv).astype(int)
    Xr_bal, yr_bal = stage_adasyn(Xtr_sel, ytr_r)
    rejuv_detector = xgb.XGBClassifier(
        n_estimators=int(s1_n), max_depth=int(s1_depth), learning_rate=float(s1_lr),
        scale_pos_weight=float(s1_spw),
        random_state=42, n_jobs=-1, eval_metric="logloss", objective="binary:logistic"
    )
    rejuv_detector.fit(Xr_bal, yr_bal)

    # Stage 2
    ytr_n = (ytr == idx_newf).astype(int)
    Xn_bal, yn_bal = stage_adasyn(Xtr_sel, ytr_n)
    newf_detector = xgb.XGBClassifier(
        n_estimators=int(s2_n), max_depth=int(s2_depth), learning_rate=float(s2_lr),
        scale_pos_weight=float(s2_spw),
        random_state=42, n_jobs=-1, eval_metric="logloss", objective="binary:logistic"
    )
    newf_detector.fit(Xn_bal, yn_bal)

    # Stage 3 (persistent vs extinct)
    mask_pe = np.isin(ytr, [idx_pers, idx_ext])
    Xtr_pe = Xtr_sel[mask_pe]
    ytr_pe = ytr[mask_pe]
    if use_adasyn:
        ada_pe = ADASYN(random_state=42, n_neighbors=2)
        try:
            Xtr_peb, ytr_peb = ada_pe.fit_resample(Xtr_pe, ytr_pe)
        except Exception:
            Xtr_peb, ytr_peb = Xtr_pe, ytr_pe
    else:
        Xtr_peb, ytr_peb = Xtr_pe, ytr_pe

    pe_clf = CatBoostClassifier(
        iterations=int(s3_iter), depth=int(s3_depth), learning_rate=float(s3_lr),
        random_seed=42, verbose=False
    )
    pe_clf.fit(Xtr_peb, ytr_peb)

    # Threshold optimization
    rprob = rejuv_detector.predict_proba(Xte_sel)[:, 1]
    nprob = newf_detector.predict_proba(Xte_sel)[:, 1]
    best_rth, best_nth = float(init_rth), float(init_nth)

    if auto_optimize_thresholds:
        best_f1 = -1.0
        for rth in np.arange(0.25, 0.45, 0.02):
            for nth in np.arange(0.30, 0.45, 0.02):
                tmp = []
                for i in range(len(Xte_sel)):
                    if rprob[i] >= rth:
                        tmp.append(idx_rejuv)
                    elif nprob[i] >= nth:
                        tmp.append(idx_newf)
                    else:
                        tmp.append(int(pe_clf.predict(Xte_sel[i:i+1])[0]))
                tmp = np.array(tmp, dtype=int)
                f1m = f1_score(yte, tmp, average="macro")
                if f1m > best_f1:
                    best_f1 = f1m
                    best_rth, best_nth = float(rth), float(nth)

    final_preds = []
    for i in range(len(Xte_sel)):
        if rprob[i] >= best_rth:
            final_preds.append(idx_rejuv)
        elif nprob[i] >= best_nth:
            final_preds.append(idx_newf)
        else:
            final_preds.append(int(pe_clf.predict(Xte_sel[i:i+1])[0]))
    final_preds = np.array(final_preds, dtype=int)

    acc = float(accuracy_score(yte, final_preds))
    macro = float(f1_score(yte, final_preds, average="macro"))
    cm = confusion_matrix(yte, final_preds)
    report = classification_report(yte, final_preds, target_names=le.classes_, zero_division=0)

    classes = list(le.classes_)
    actual_counts = pd.Series(le.inverse_transform(yte)).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(le.inverse_transform(final_preds)).value_counts().reindex(classes, fill_value=0)

    artifacts = {
        "model_kind": "Three-Stage Ensemble",
        "rejuv_detector": rejuv_detector,
        "newf_detector": newf_detector,
        "persistextinct_clf": pe_clf,
        "labelencoder": le,
        "scaler": scaler,
        "selector": selector,
        "featurenames": featurenames,
        "rfselector": rfselector,
        "T": len(areacols),
        "pipeline": "three_stage",
        "feature_groups": copy.deepcopy(groups),
        "trained_classes": list(le.classes_),
        "preprocessing_version": 2,
        "thresholds": {
            "rejuv": float(best_rth),
            "newf": float(best_nth)
        }
    }

    return {
        "acc": acc, "macro_f1": macro, "cm": cm, "report": report,
        "classes": classes, "actual_counts": actual_counts, "pred_counts": pred_counts,
        "artifacts": artifacts
    }
