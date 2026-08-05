"""
ui/render.py
============
HTML rendering functions for the Gradio UI:
  - error_html, success_summary_html
  - render_performance_dashboard
  - render_classification_report
  - render_best_model_status
  - render_dataset_cards, render_wq_dataset_cards
  - render_wq_summary_html
  - get_model_details, get_wq_model_details
  - update_operational_model_ui
"""

import os
import html
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

from ..config import (
    OPERATIONAL_MODELS,
    OPERATIONAL_LABEL_ENCODERS,
    OPERATIONAL_MODEL_METADATA,
    WQ_OPERATIONAL_MODELS,
    _USER_TRAINED_MODELS,
    _USER_TRAINED_WQ_MODELS,
)
from ..registry import (
    _safe_joblib_load,
    load_pretrained_model,
    _operational_model_path,
    extract_trained_classes_from_bundle,
    get_operational_model_trained_classes,
)


# -----------------------------
# Validation UI
# -----------------------------
def error_html(message):
    """Generate error message HTML."""
    return f"""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                border-left: 4px solid #dc2626;
                border-radius: 12px;
                padding: 20px;
                margin: 16px 0;
                box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 24px;">❌</span>
            <h3 style="margin: 0; color: #991b1b; font-family: 'Poppins', sans-serif; font-weight: 600;">
                Validation Failed
            </h3>
        </div>
        <p style="margin: 8px 0 0 36px; color: #7f1d1d; font-size: 14px; line-height: 1.6;">
            {message}
        </p>
    </div>
    """


def success_summary_html(rows, label_col, ts_cols, T, classes):
    """Generate success summary HTML."""
    class_items = "".join([f"<li><strong>{cls}</strong>: {count}</li>" for cls, count in classes.items()])
    return f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                border-left: 4px solid #10b981;
                border-radius: 12px;
                padding: 20px;
                margin: 16px 0;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span style="font-size: 24px;">✅</span>
            <h3 style="margin: 0; color: #065f46; font-family: 'Poppins', sans-serif; font-weight: 600;">
                Dataset Validated Successfully
            </h3>
        </div>
        <div style="margin-left: 36px; color: #064e3b;">
            <p style="margin: 6px 0; font-size: 14px;"><strong>Rows accepted:</strong> {rows}</p>
            <p style="margin: 6px 0; font-size: 14px;"><strong>Label column:</strong> {label_col}</p>
            <p style="margin: 6px 0; font-size: 14px;"><strong>Time-series columns:</strong> {ts_cols}</p>
            <p style="margin: 6px 0; font-size: 14px;"><strong>Unified length:</strong> {T}</p>
            <p style="margin: 6px 0; font-size: 14px;"><strong>Classes found:</strong></p>
            <ul style="margin: 4px 0 0 20px; font-size: 14px;">
                {class_items}
            </ul>
        </div>
    </div>
    """


# -----------------------------
# Performance Dashboard
# -----------------------------
def render_performance_dashboard(exp_id, model_kind, dataset_name, acc, macro_f1,
                                 rows=None, ts_length=None, thresholds=None,
                                 template_name=None, features_selected=None):
    """Render a compact, theme-aware latest-run performance dashboard."""
    if acc >= 0.90:
        status = "Excellent"
        status_class = "status-excellent"
    elif acc >= 0.75:
        status = "Good"
        status_class = "status-good"
    else:
        status = "Review"
        status_class = "status-review"

    if acc >= 0.85 and macro_f1 >= 0.80:
        insight = "High accuracy with balanced performance across the selected classes."
    elif acc >= 0.75 and macro_f1 < acc - 0.10:
        insight = "Accuracy is strong, but the lower Macro F1 suggests uneven class performance."
    elif macro_f1 >= 0.75:
        insight = "Macro F1 indicates consistent performance across classes."
    else:
        insight = "Review class balance, feature settings, and model hyperparameters before deployment."

    detail_items = []
    if dataset_name:
        detail_items.append(("Dataset", html.escape(str(dataset_name))))
    if rows is not None:
        detail_items.append(("Samples", f"{int(rows):,}"))
    if ts_length is not None:
        detail_items.append(("Series length", html.escape(str(ts_length))))
    if features_selected is not None:
        detail_items.append(("Selected features", html.escape(str(features_selected))))
    if thresholds:
        threshold_text = ", ".join(f"{k}: {v:.2f}" for k, v in thresholds.items())
        detail_items.append(("Thresholds", html.escape(threshold_text)))
    if template_name:
        detail_items.append(("Template", html.escape(str(template_name))))

    details_html = "".join(
        f"<div><span>{label}</span><b>{value}</b></div>"
        for label, value in detail_items
    ) or "<div><span>Details</span><b>Not available</b></div>"

    return f"""
    <section class='training-summary-card'>
        <div class='training-summary-head'>
            <div>
                <span class='eyebrow'>LATEST RUN</span>
                <div class='training-summary-title'>Training complete</div>
                <div class='training-summary-subtitle'>Experiment {int(exp_id):02d}</div>
            </div>
            <span class='training-status {status_class}'>{status}</span>
        </div>

        <div class='training-model-line'>
            <span>Model</span><b>{html.escape(str(model_kind))}</b>
        </div>

        <div class='training-kpi-grid'>
            <div class='training-kpi training-kpi-primary'>
                <span>Accuracy</span><b>{acc * 100:.2f}%</b>
            </div>
            <div class='training-kpi training-kpi-secondary'>
                <span>Macro F1</span><b>{macro_f1 * 100:.2f}%</b>
            </div>
        </div>

        <div class='training-insight'>{html.escape(insight)}</div>

        <details class='training-details'>
            <summary>Technical details</summary>
            <div class='training-detail-grid'>{details_html}</div>
        </details>
    </section>
    """


# -----------------------------
# Classification Report
# -----------------------------
def render_classification_report(report_str, classes):
    """
    Generates a modern, structured classification report with KPI chips,
    per-class table, and F1-score visualizations.
    """
    try:
        lines = report_str.strip().split("\n")

        class_data = []
        for line in lines[2:-5]:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_name = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    support = int(parts[4])
                    class_data.append({
                        "class": class_name,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": support
                    })
                except (ValueError, IndexError):
                    continue

        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

        for line in lines[-5:]:
            if "accuracy" in line.lower():
                parts = line.split()
                try:
                    accuracy = float(parts[1])
                except (ValueError, IndexError):
                    pass
            elif "macro avg" in line.lower():
                parts = line.split()
                try:
                    macro_f1 = float(parts[3])
                except (ValueError, IndexError):
                    pass
            elif "weighted avg" in line.lower():
                parts = line.split()
                try:
                    weighted_f1 = float(parts[3])
                except (ValueError, IndexError):
                    pass

        class_data_sorted = sorted(class_data, key=lambda x: x["f1"])

        if class_data_sorted:
            min_f1 = class_data_sorted[0]["f1"]
            max_f1 = class_data_sorted[-1]["f1"]
        else:
            min_f1 = max_f1 = 0

        kpi_chips = f"""
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:24px;'>
            <div style='
                background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding:16px;
                border-radius:10px;
                color:white;
                text-align:center;
                box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
            '>
                <div style='font-size:11px; opacity:0.9; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>
                    Accuracy
                </div>
                <div style='font-size:28px; font-weight:700;'>
                    {accuracy*100:.2f}%
                </div>
            </div>

            <div style='
                background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding:16px;
                border-radius:10px;
                color:white;
                text-align:center;
                box-shadow: 0 3px 10px rgba(245, 87, 108, 0.3);
            '>
                <div style='font-size:11px; opacity:0.9; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>
                    Macro Avg F1
                </div>
                <div style='font-size:28px; font-weight:700;'>
                    {macro_f1*100:.2f}%
                </div>
            </div>

            <div style='
                background:linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding:16px;
                border-radius:10px;
                color:white;
                text-align:center;
                box-shadow: 0 3px 10px rgba(79, 172, 254, 0.3);
            '>
                <div style='font-size:11px; opacity:0.9; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>
                    Weighted Avg F1
                </div>
                <div style='font-size:28px; font-weight:700;'>
                    {weighted_f1*100:.2f}%
                </div>
            </div>
        </div>
        """

        table_rows = ""
        for item in class_data_sorted:
            if item["f1"] == min_f1 and min_f1 != max_f1:
                row_bg = "rgba(239, 68, 68, 0.08)"
            elif item["f1"] == max_f1 and min_f1 != max_f1:
                row_bg = "rgba(16, 185, 129, 0.08)"
            else:
                row_bg = "transparent"

            bar_width = item["f1"] * 100

            table_rows += f"""
            <tr style='background:{row_bg};'>
                <td style='padding:12px; font-weight:600; color:#1e293b;'>{html.escape(item["class"])}</td>
                <td style='padding:12px; text-align:center; color:#475569;'>{item["precision"]:.2f}</td>
                <td style='padding:12px; text-align:center; color:#475569;'>{item["recall"]:.2f}</td>
                <td style='padding:12px; text-align:center; color:#475569;'>{item["f1"]:.2f}</td>
                <td style='padding:12px; text-align:center; color:#64748b;'>{item["support"]}</td>
                <td style='padding:12px;'>
                    <div style='
                        background:#e2e8f0;
                        border-radius:4px;
                        height:20px;
                        position:relative;
                        overflow:hidden;
                    '>
                        <div style='
                            background:linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            height:100%;
                            width:{bar_width}%;
                            border-radius:4px;
                            transition:width 0.3s ease;
                        '></div>
                        <div style='
                            position:absolute;
                            top:0;
                            left:0;
                            right:0;
                            bottom:0;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-size:11px;
                            font-weight:600;
                            color:#1e293b;
                        '>
                            {item["f1"]*100:.1f}%
                        </div>
                    </div>
                </td>
            </tr>
            """

        report_html = f"""
        <div style='
            background:#ffffff;
            border-radius:16px;
            padding:24px;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
            border:1px solid #e2e8f0;
        '>
            <div style='font-size:20px; font-weight:700; color:#1e293b; font-family:Poppins, sans-serif; margin-bottom:20px;'>
                📊 Classification Report
            </div>

            {kpi_chips}

            <div style='margin-top:24px;'>
                <div style='font-size:14px; font-weight:600; color:#475569; margin-bottom:12px;'>
                    Per-Class Performance (sorted by F1 Score)
                </div>
                <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                    <thead>
                        <tr style='background:#f8fafc; border-bottom:2px solid #e2e8f0;'>
                            <th style='padding:12px; text-align:left; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>Class</th>
                            <th style='padding:12px; text-align:center; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>Precision</th>
                            <th style='padding:12px; text-align:center; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>Recall</th>
                            <th style='padding:12px; text-align:center; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>F1-Score</th>
                            <th style='padding:12px; text-align:center; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>Support</th>
                            <th style='padding:12px; text-align:center; font-weight:600; color:#64748b; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;'>F1 Visualization</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>

            <div style='margin-top:16px; padding:12px; background:#f8fafc; border-radius:8px; font-size:12px; color:#64748b;'>
                <strong>Note:</strong> Rows are sorted by F1-Score (lowest to highest).
                <span style='color:#ef4444;'>●</span> Lowest F1 highlighted in light red,
                <span style='color:#10b981;'>●</span> Highest F1 highlighted in light green.
            </div>
        </div>
        """

        return report_html

    except Exception as e:
        return f"""
        <div style='padding:20px; background:#fee2e2; border-radius:8px; color:#991b1b;'>
            <strong>Error parsing classification report:</strong> {str(e)}
        </div>
        """


# -----------------------------
# Best Model / Experiment History
# -----------------------------
def render_best_model_status(history):
    """Render compact, theme-aware experiment history cards."""
    if not history:
        return """
        <div class='advanced-empty-state'>
            <span class='advanced-empty-mark'>01</span>
            <div>
                <b>No training experiments yet</b>
                <small>Configure a model and run training to build the experiment history.</small>
            </div>
        </div>
        """

    best_acc = max(h["acc"] for h in history)
    cards = []

    for h in reversed(history):
        exp_id = h["experiment_id"]
        model_name = html.escape(str(h["model_kind"]))
        dataset_name = html.escape(str(h["dataset"]))
        accuracy = float(h["acc"]) * 100.0
        macro_f1 = float(h["macro_f1"]) * 100.0
        is_best = h["acc"] == best_acc

        cards.append(f"""
        <article class='experiment-card {'experiment-card-best' if is_best else ''}'>
            <div class='experiment-card-head'>
                <div>
                    <span class='experiment-id'>EXPERIMENT {exp_id:02d}</span>
                    <div class='experiment-model'>{model_name}</div>
                </div>
                {"<span class='experiment-best-badge'>BEST</span>" if is_best else ""}
            </div>
            <div class='experiment-dataset'>{dataset_name}</div>
            <div class='experiment-metrics'>
                <div><span>Accuracy</span><b>{accuracy:.2f}%</b></div>
                <div><span>Macro F1</span><b>{macro_f1:.2f}%</b></div>
            </div>
        </article>
        """)

    return f"""
    <div class='training-history-panel'>
        <div class='training-history-head'>
            <div>
                <span class='eyebrow'>EXPERIMENT TRACKER</span>
                <div class='training-history-title'>Training history</div>
            </div>
            <div class='training-history-summary'>
                <b>{len(history)}</b><span>runs</span>
                <b>{best_acc * 100:.2f}%</b><span>best</span>
            </div>
        </div>
        <div class='experiment-card-list'>{''.join(cards)}</div>
    </div>
    """


# -----------------------------
# Dataset Cards
# -----------------------------
def render_dataset_cards(preloaded_datasets):
    """
    Render dataset info as visually appealing cards with colors and icons.
    """
    html_cards = ""
    for name, meta in preloaded_datasets.items():
        df = meta["df"]
        rows = df.shape[0]
        cols = df.shape[1]
        ts_len = meta["meta"].get("T", "N/A")
        classes = df['label'].nunique() if 'label' in df.columns else "N/A"
        path = meta.get("path", "N/A")

        if rows > 5000:
            color = "#FF6B6B"
        elif rows > 2000:
            color = "#FFA94D"
        else:
            color = "#4DABF7"

        html_cards += f"""
        <div class='dataset-card' style='background-color:{color};'>
            <h3>{name}</h3>
            <p>🗂 Rows: <b>{rows}</b></p>
            <p>📊 Columns: <b>{cols}</b></p>
            <p>⏱ TS Length: <b>{ts_len}</b></p>
            <p>🎯 Classes: <b>{classes}</b></p>
            <p style='font-size:12px;'>ℹ️ Path: {path}</p>
        </div>
        """

    return f"""
    <style>
    .dataset-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: flex-start;
    }}
    .dataset-card {{
        flex: 0 1 200px;
        border-radius: 12px;
        padding: 12px;
        color: white;
        font-family: Arial, sans-serif;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }}
    .dataset-card:hover {{
        transform: scale(1.05);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }}
    .dataset-card h3 {{
        margin: 0 0 8px 0;
        font-size: 16px;
        text-align: center;
    }}
    .dataset-card p {{
        margin: 4px 0;
        font-size: 13px;
        text-align: center;
    }}
    </style>
    <div class="dataset-container">
        {html_cards}
    </div>
    """


def render_wq_dataset_cards(wq_preloaded_datasets):
    """
    Render WQ dataset info as visually appealing cards with colors and icons.
    """
    html_cards = ""
    for name, data in wq_preloaded_datasets.items():
        df_wq = data.get("df_wq")
        df_wsa = data.get("df_wsa")
        ndci_cols = data.get("ndci_cols", [])
        ndti_cols = data.get("ndti_cols", [])
        wsa_cols = data.get("wsa_cols", [])

        rows_wq = len(df_wq) if df_wq is not None else 0
        rows_wsa = len(df_wsa) if df_wsa is not None else 0

        if rows_wq > 5000:
            color = "#FF6B6B"
        elif rows_wq > 2000:
            color = "#FFA94D"
        else:
            color = "#4DABF7"

        html_cards += f"""
        <div class='dataset-card' style='background-color:{color};'>
            <h3>{name}</h3>
            <p>🗂 WQ Rows: <b>{rows_wq}</b></p>
            <p>🗂 WSA Rows: <b>{rows_wsa}</b></p>
            <p>📊 NDCI Cols: <b>{len(ndci_cols)}</b></p>
            <p>📊 NDTI Cols: <b>{len(ndti_cols)}</b></p>
            <p>📊 WSA Cols: <b>{len(wsa_cols)}</b></p>
        </div>
        """

    return f"""
    <style>
    .dataset-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: flex-start;
    }}
    .dataset-card {{
        flex: 0 1 200px;
        border-radius: 12px;
        padding: 12px;
        color: white;
        font-family: Arial, sans-serif;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }}
    .dataset-card:hover {{
        transform: scale(1.05);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }}
    .dataset-card h3 {{
        margin: 0 0 8px 0;
        font-size: 16px;
        text-align: center;
    }}
    .dataset-card p {{
        margin: 4px 0;
        font-size: 13px;
        text-align: center;
    }}
    </style>
    <div class="dataset-container">
        {html_cards}
    </div>
    """


# -----------------------------
# WQ Result Rendering
# -----------------------------
def render_wq_summary_html(df, scores=None, model_name=""):
    """Generate a styled HTML summary card for WQ results."""
    typology_counts = df['Typology'].value_counts().to_dict() if 'Typology' in df.columns else {}
    total = len(df)

    typ_rows = ""
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
    for i, (cls, cnt) in enumerate(typology_counts.items()):
        pct = cnt / total * 100 if total else 0
        c = colors[i % len(colors)]
        typ_rows += f"""
        <tr>
            <td style='padding:10px 14px;'><span style='display:inline-block;width:10px;height:10px;background:{c};border-radius:50%;margin-right:8px;'></span>{html.escape(str(cls))}</td>
            <td style='padding:10px 14px;text-align:center;font-weight:600;'>{cnt}</td>
            <td style='padding:10px 14px;text-align:center;'>{pct:.1f}%</td>
        </tr>"""

    score_html = ""
    if scores:
        sil = scores.get('final_sil', 0)
        ch = scores.get('final_ch', 0)
        k = scores.get('final_k', 0)
        pca_var = scores.get('pca_variance', [])
        pca_total = sum(pca_var) * 100 if pca_var else 0
        score_html = f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;'>
            <div style='background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;padding:16px;color:white;text-align:center;'>
                <div style='font-size:11px;opacity:0.9;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>K Clusters</div>
                <div style='font-size:28px;font-weight:700;'>{k}</div>
            </div>
            <div style='background:linear-gradient(135deg,#f093fb,#f5576c);border-radius:12px;padding:16px;color:white;text-align:center;'>
                <div style='font-size:11px;opacity:0.9;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>Silhouette</div>
                <div style='font-size:28px;font-weight:700;'>{sil:.4f}</div>
            </div>
            <div style='background:linear-gradient(135deg,#4facfe,#00f2fe);border-radius:12px;padding:16px;color:white;text-align:center;'>
                <div style='font-size:11px;opacity:0.9;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>PCA Variance</div>
                <div style='font-size:28px;font-weight:700;'>{pca_total:.1f}%</div>
            </div>
        </div>"""

    return f"""
    <div style='background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:24px;box-shadow:0 4px 12px rgba(0,0,0,0.08);'>
        <div style='font-size:20px;font-weight:700;color:#1e293b;font-family:Poppins,sans-serif;margin-bottom:16px;'>
            💧 {html.escape(model_name) if model_name else 'Water Quality Analysis'} Complete
        </div>
        <div style='font-size:13px;color:#64748b;margin-bottom:16px;'>
            Analyzed {total} water bodies
        </div>
        {score_html}
        <div style='font-weight:600;margin-bottom:10px;font-size:14px;color:#475569;'>Typology Distribution</div>
        <table style='width:100%;border-collapse:collapse;font-size:13px;'>
            <thead><tr style='background:#f8fafc;border-bottom:2px solid #e2e8f0;'>
                <th style='padding:10px 14px;text-align:left;color:#64748b;font-size:11px;text-transform:uppercase;'>Class</th>
                <th style='padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;'>Count</th>
                <th style='padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;'>Percentage</th>
            </tr></thead>
            <tbody>{typ_rows}</tbody>
        </table>
    </div>
    """


# -----------------------------
# Operational Model Details
# -----------------------------
def get_model_details(model_name):
    """Render a compact, theme-aware card for any Pond model discovered in Drive."""
    if not model_name:
        return (
            "<div class='mini-empty'>"
            "<span class='mini-empty-icon'>🧠</span>"
            "<div><b>No model available</b><small>Add a compatible model to Operational_Active and refresh.</small></div>"
            "</div>"
        )

    known = {
        "USO_XGB_d6_lr003_nest2000_Acc89": (
            "Universal Shape XGBoost",
            "13 length-independent shape features",
            "Legacy model + paired LabelEncoder",
            "🌐",
        ),
        "USO_5_XGB_d6_lr003_nest1500_Acc86": (
            "Universal 5-Class XGBoost",
            "8 length-independent shape features",
            "Legacy model + paired LabelEncoder",
            "🌍",
        ),
    }

    filepath = _operational_model_path(model_name)
    if not filepath:
        return (
            "<div class='inline-alert inline-alert-error'>"
            f"Model not found: {html.escape(str(model_name))}"
            "</div>"
        )

    bundle, error = load_pretrained_model(filepath)
    if error:
        return (
            "<div class='inline-alert inline-alert-error'>"
            f"Could not load model details: {html.escape(str(error))}"
            "</div>"
        )

    metadata = OPERATIONAL_MODEL_METADATA.get(model_name, {})
    pipeline = metadata.get("pipeline") or (
        bundle.get("pipeline") if isinstance(bundle, dict) else None
    ) or "bundle inference"

    if model_name in known:
        title, architecture, source, icon = known[model_name]
    else:
        title = str(
            bundle.get("model_kind", "User-Trained Operational Model")
            if isinstance(bundle, dict)
            else type(bundle).__name__
        )
        architecture = str(bundle.get("architecture", f"Pipeline: {pipeline}")) if isinstance(bundle, dict) else f"Pipeline: {pipeline}"
        source = str(
            bundle.get("deployed_from", "Drive Operational_Active folder")
            if isinstance(bundle, dict)
            else "Drive Operational_Active folder"
        )
        icon = "🧠"

    trained_classes = extract_trained_classes_from_bundle(bundle)
    if not trained_classes:
        encoder_path = OPERATIONAL_LABEL_ENCODERS.get(model_name)
        encoder = None
        if encoder_path:
            encoder, _ = load_pretrained_model(encoder_path)
        trained_classes = extract_trained_classes_from_bundle(bundle, encoder)

    classes = trained_classes or []
    class_pills = "".join(
        f"<span class='class-pill'>{html.escape(str(cls))}</span>"
        for cls in classes
    ) or "<span class='class-pill class-pill-muted'>Classes unavailable</span>"

    accuracy = bundle.get("training_accuracy") if isinstance(bundle, dict) else None
    accuracy_text = f"{float(accuracy) * 100:.2f}%" if accuracy is not None else "Stored model"

    return f"""
    <div class='model-profile'>
        <div class='model-profile-top'>
            <div class='model-avatar'>{icon}</div>
            <div class='model-profile-heading'>
                <div class='model-profile-title'>{html.escape(title)}</div>
                <div class='model-profile-name'>{html.escape(str(model_name))}</div>
            </div>
            <span class='model-state-badge'>READY</span>
        </div>
        <div class='model-facts'>
            <div><span>Pipeline</span><b>{html.escape(str(pipeline))}</b></div>
            <div><span>Training score</span><b>{html.escape(accuracy_text)}</b></div>
        </div>
        <div class='model-profile-copy'>{html.escape(str(architecture))}</div>
        <div class='class-pill-row'>{class_pills}</div>
        <div class='model-profile-foot'>
            <span>{html.escape(str(source))}</span>
            <span>{html.escape(Path(filepath).name)}</span>
        </div>
    </div>
    """


def get_wq_model_details(model_name):
    """Generate a compact, theme-aware WQ model card from the scanned Drive bundle."""
    if not model_name:
        return (
            "<div class='mini-empty'>"
            "<span class='mini-empty-icon'>💧</span>"
            "<div><b>No WQ model available</b><small>Add a compatible model and refresh.</small></div>"
            "</div>"
        )

    filepath = WQ_OPERATIONAL_MODELS.get(model_name) or _USER_TRAINED_WQ_MODELS.get(model_name)
    if not filepath:
        return (
            "<div class='inline-alert inline-alert-error'>"
            f"Model not found: {html.escape(str(model_name))}"
            "</div>"
        )

    bundle, error = _safe_joblib_load(filepath)
    if error or not isinstance(bundle, dict):
        return (
            "<div class='inline-alert inline-alert-error'>"
            f"Could not read model details: {html.escape(str(error))}"
            "</div>"
        )

    internal_name = str(bundle.get("model_name", model_name))
    source = str(bundle.get("deployed_from", "Drive Operational_Active folder"))
    hp = bundle.get("hyperparameters", {}) or {}
    metrics = bundle.get("metrics", {}) or {}

    if "Model1" in internal_name:
        summary = "PCA + KMeans water-quality clustering"
        icon = "🧭"
    elif "Model2" in internal_name:
        summary = "DTW temporal shape classification"
        icon = "〰️"
    elif "Model3" in internal_name:
        summary = "Triple-stream WSA + NDCI + NDTI clustering"
        icon = "🔀"
    else:
        summary = internal_name
        icon = "💧"

    hp_items = list(hp.items())[:4]
    metric_items = list(metrics.items())[:4]
    hp_pills = "".join(
        f"<span class='class-pill'>{html.escape(str(k))}: {html.escape(str(v))}</span>"
        for k, v in hp_items
    ) or "<span class='class-pill class-pill-muted'>Saved configuration</span>"
    metric_pills = "".join(
        f"<span class='metric-pill'>{html.escape(str(k))}: "
        f"{float(v):.4f}</span>" if isinstance(v, (int, float, np.number))
        else f"<span class='metric-pill'>{html.escape(str(k))}: {html.escape(str(v))}</span>"
        for k, v in metric_items
    ) or "<span class='metric-pill metric-pill-muted'>Metrics not recorded</span>"

    return f"""
    <div class='model-profile model-profile-water'>
        <div class='model-profile-top'>
            <div class='model-avatar'>{icon}</div>
            <div class='model-profile-heading'>
                <div class='model-profile-title'>{html.escape(str(model_name))}</div>
                <div class='model-profile-name'>{html.escape(summary)}</div>
            </div>
            <span class='model-state-badge'>READY</span>
        </div>
        <div class='class-pill-row'>{metric_pills}</div>
        <div class='class-pill-row'>{hp_pills}</div>
        <div class='model-profile-foot'>
            <span>{html.escape(source)}</span>
            <span>{html.escape(Path(filepath).name)}</span>
        </div>
    </div>
    """


def update_operational_model_ui(model_name):
    """
    Update both the model description and Operational class selector.

    The checkbox choices always come from the selected model, never from the
    uploaded Operational dataset.
    """
    description = get_model_details(model_name)
    classes, error = get_operational_model_trained_classes(model_name)

    if error:
        class_card = f"""
        <div style='margin-top:10px; padding:10px 12px; border-radius:8px;
                    background:#fff7ed; border-left:4px solid #f97316;
                    color:#9a3412; font-size:12px;'>
            ⚠️ Could not read trained classes: {html.escape(str(error))}
        </div>
        """
        return description + class_card, gr.update(
            choices=[],
            value=[],
            interactive=False
        )

    class_list = ", ".join(html.escape(str(c)) for c in classes)
    class_card = f"""
    <div style='margin-top:10px; padding:10px 12px; border-radius:8px;
                background:#ecfdf5; border-left:4px solid #10b981;
                color:#065f46; font-size:12px;'>
        <b>Model-trained classes:</b> {class_list}<br>
        Only these classes are available in Operational mode.
    </div>
    """
    return description + class_card, gr.update(
        choices=classes,
        value=classes,
        interactive=True
    )
