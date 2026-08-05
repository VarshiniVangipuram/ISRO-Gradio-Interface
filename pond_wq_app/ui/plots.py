"""
ui/plots.py
===========
Matplotlib visualization functions for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion(cm, classes, acc):
    fig, ax = plt.subplots(figsize=(8,6))
    cm = np.array(cm)
    cm_pct = cm / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    ann = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm, annot=ann, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_feature_importance_rf(rfselector, all_feature_names, topk=20):
    importances = rfselector.feature_importances_
    idx = np.argsort(importances)[::-1][:topk]
    vals = importances[idx]
    names = [all_feature_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(len(vals))[::-1], vals[::-1], color="teal", alpha=0.85)
    ax.set_yticks(range(len(vals))[::-1])
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_title(f"Top {topk} Feature Importances (RF)")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_actual_vs_pred(actual_counts, pred_counts, acc):
    fig, ax = plt.subplots(figsize=(8,5))
    classes = list(actual_counts.index)
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width/2, actual_counts.values, width, label="Actual", alpha=0.85)
    ax.bar(x + width/2, pred_counts.values, width, label="Predicted", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_title(f"Actual vs Predicted (Accuracy {acc*100:.2f}%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig
