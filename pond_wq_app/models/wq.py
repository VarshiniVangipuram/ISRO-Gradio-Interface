"""
models/wq.py
============
Water Quality assessment models:
  - Model 1: PCA + KMeans + TS + Resilience
  - Model 2: DTW Shape Classifier
  - Model 3: Triple-Stream (WSA + NDCI + NDTI) PCA + KMeans
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# ---------- MODEL 1: PCA + KMeans + TS + Resilience ----------

def wq_model1_run(df_wq, ndci_cols, ndti_cols,
                  n_clusters=3, k_range_max=10, pca_components=4,
                  bloom_threshold=0.2):
    """Run PCA + KMeans clustering with TS and Resilience features."""
    df = df_wq.copy()
    all_temporal = ndci_cols + ndti_cols
    X_raw = df[all_temporal].fillna(df[all_temporal].mean())

    # Feature engineering
    df['NDCI_Mean'] = df[ndci_cols].mean(axis=1)
    df['NDTI_Mean'] = df[ndti_cols].mean(axis=1)
    df['NDCI_CV']   = df[ndci_cols].std(axis=1) / (df['NDCI_Mean'].abs() + 1e-6)
    df['NDCI_STD']  = df[ndci_cols].std(axis=1)

    df['TS_Amplitude'] = df[ndci_cols].apply(lambda r: np.max(r.values) - np.min(r.values), axis=1)
    df['Bloom_Freq']   = df[ndci_cols].apply(lambda r: int(np.sum(r.values > float(bloom_threshold))), axis=1)
    df['Resilience_Score'] = 1.0 / (df['NDCI_STD'] + df[ndti_cols].std(axis=1) + 1e-6)

    # PCA on temporal
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X_raw)
    n_comp = min(int(pca_components), X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Hybrid feature set
    behavioral = ['NDCI_Mean', 'NDTI_Mean', 'NDCI_CV']
    scaler_beh = StandardScaler()
    X_beh = scaler_beh.fit_transform(df[behavioral])
    X_final = np.hstack((X_pca, X_beh))

    # K-search
    k_max = min(int(k_range_max), len(df) - 1)
    k_range = range(2, k_max + 1)
    inertia_vals, sil_vals, ch_vals = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
        lbl = km.fit_predict(X_final)
        inertia_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(X_final, lbl))
        ch_vals.append(calinski_harabasz_score(X_final, lbl))

    # Final clustering
    final_k = min(int(n_clusters), len(df) - 1)
    model = KMeans(n_clusters=final_k, init='k-means++', n_init=20, random_state=42)
    df['Cluster'] = model.fit_predict(X_final)

    # Typology labels – standardized WQ-I to WQ-IV scale
    rank = df.groupby('Cluster')['NDCI_Mean'].mean().sort_values().index.tolist()
    eco_map = {}
    eco_names = [
        "WQ-I: Pristine (Oligotrophic)",
        "WQ-II: Moderate (Mesotrophic)",
        "WQ-III: Stressed (Eutrophic)",
        "WQ-IV: Critical (Hyper-Eutrophic)",
        "WQ-IV: Critical (Severely Degraded)"
    ]
    for i, idx in enumerate(rank):
        eco_map[idx] = eco_names[min(i, len(eco_names)-1)]
    df['Typology'] = df['Cluster'].map(eco_map)

    # TS pattern classification – standardized labels
    ts_score = df['TS_Amplitude'] + df['Bloom_Freq']
    try:
        df['TS_Class'] = pd.qcut(ts_score, q=3,
                                 labels=["TS-Stable: Low Variability",
                                         "TS-Seasonal: Moderate Bloom",
                                         "TS-Acute: High Bloom Frequency"],
                                 duplicates='drop')
    except Exception:
        df['TS_Class'] = "TS-Stable: Low Variability"

    # Resilience classification – standardized labels
    try:
        df['Resilience_Class'] = pd.qcut(df['Resilience_Score'], q=3,
                                         labels=["Resilience-Low: Vulnerable",
                                                 "Resilience-Med: Moderate",
                                                 "Resilience-High: Stable"],
                                         duplicates='drop')
    except Exception:
        df['Resilience_Class'] = "Resilience-Med: Moderate"

    # ---- FIGURES ----
    figs = {}

    # Elbow + Silhouette
    fig_es, axes_es = plt.subplots(1, 2, figsize=(12, 4))
    axes_es[0].plot(list(k_range), inertia_vals, marker='o', color='navy', linestyle='--')
    axes_es[0].set_title('Elbow Method'); axes_es[0].set_xlabel('K'); axes_es[0].set_ylabel('Inertia')
    axes_es[1].plot(list(k_range), sil_vals, marker='s', color='darkorange')
    axes_es[1].set_title('Silhouette Score'); axes_es[1].set_xlabel('K'); axes_es[1].set_ylabel('Score')
    plt.tight_layout(); figs['elbow_sil'] = fig_es

    # CH Score
    fig_ch, ax_ch = plt.subplots(figsize=(6, 4))
    ax_ch.plot(list(k_range), ch_vals, marker='^', color='green')
    ax_ch.set_title('Calinski-Harabasz Score'); ax_ch.set_xlabel('K'); ax_ch.set_ylabel('CH Score')
    plt.tight_layout(); figs['ch_score'] = fig_ch

    # Behavioral Space scatter
    fig_bs, ax_bs = plt.subplots(figsize=(8, 5))
    for typ in df['Typology'].unique():
        sub = df[df['Typology'] == typ]
        ax_bs.scatter(sub['NDCI_Mean'], sub['NDCI_CV'], label=typ, alpha=0.7, s=40)
    ax_bs.set_xlabel('NDCI Mean'); ax_bs.set_ylabel('NDCI CV')
    ax_bs.set_title('Behavioral Space (Typology)'); ax_bs.legend(fontsize=8)
    plt.tight_layout(); figs['behavioral'] = fig_bs

    # Temporal Fingerprints
    fig_tf, ax_tf = plt.subplots(figsize=(10, 4))
    for typ in df['Typology'].unique():
        mean_ts = df[df['Typology'] == typ][ndci_cols].mean()
        ax_tf.plot(mean_ts.values, label=typ, marker='.', markersize=4)
    ax_tf.set_title('Temporal Fingerprints (Mean NDCI by Typology)')
    ax_tf.set_xlabel('Time Step'); ax_tf.set_ylabel('Mean NDCI'); ax_tf.legend(fontsize=8)
    plt.tight_layout(); figs['temporal'] = fig_tf

    # TS Class count
    fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
    ts_counts = df['TS_Class'].value_counts()
    ax_ts.bar(ts_counts.index.astype(str), ts_counts.values, color=['#4CAF50','#FFC107','#F44336'])
    ax_ts.set_title('Time Series Pattern Distribution'); ax_ts.set_ylabel('Count')
    plt.xticks(rotation=15); plt.tight_layout(); figs['ts_class'] = fig_ts

    # Resilience count
    fig_rc, ax_rc = plt.subplots(figsize=(6, 4))
    rc_counts = df['Resilience_Class'].value_counts()
    ax_rc.bar(rc_counts.index.astype(str), rc_counts.values, color=['#e74c3c','#f39c12','#2ecc71'])
    ax_rc.set_title('Resilience Stability Distribution'); ax_rc.set_ylabel('Count')
    plt.xticks(rotation=15); plt.tight_layout(); figs['resilience'] = fig_rc

    scores = {
        'k_range': list(k_range), 'inertia': inertia_vals,
        'silhouette': sil_vals, 'ch': ch_vals,
        'final_k': final_k,
        'final_sil': silhouette_score(X_final, df['Cluster'].values),
        'final_ch': calinski_harabasz_score(X_final, df['Cluster'].values),
        'pca_variance': pca.explained_variance_ratio_.tolist(),
    }

    artifacts = {
        "model_name": f"Model1: PCA + KMeans (K={final_k})",
        "scaler_pca": scaler_pca,
        "pca": pca,
        "behavioral_features": behavioral,
        "scaler_beh": scaler_beh,
        "kmeans": model,
        "hyperparameters": {
            "K": final_k,
            "bloom_threshold": float(bloom_threshold)
        },
        "metrics": {
            "silhouette": scores["final_sil"],
            "calinski_harabasz": scores["final_ch"],
            "pca_variance_pct": sum(pca.explained_variance_ratio_) * 100.0
        }
    }

    return df, scores, figs, artifacts


# ---------- MODEL 2: DTW Temporal Shape Classifier ----------

def dtw_distance(ts1, ts2):
    """Compute DTW distance between two 1D time series."""
    n, m = len(ts1), len(ts2)
    dtw_mat = np.full((n+1, m+1), np.inf)
    dtw_mat[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(ts1[i-1] - ts2[j-1])
            dtw_mat[i, j] = cost + min(dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1])
    return dtw_mat[n, m]


def wq_model2_run(df_result, ndci_cols):
    """Run DTW shape classification on existing WQ result dataframe."""
    df = df_result.copy()
    df[ndci_cols] = df[ndci_cols].fillna(df[ndci_cols].mean())
    ts_matrix = df[ndci_cols].values
    T = ts_matrix.shape[1]
    x = np.arange(T)

    # Prototypes
    early_proto = np.exp(-0.5 * ((x - 2) / 2) ** 2)
    mid_proto   = np.exp(-0.5 * ((x - T//2) / 2) ** 2)
    late_proto  = np.exp(-0.5 * ((x - (T-2)) / 2) ** 2)
    osc_proto   = np.sin(x / 2) + 1
    prototypes  = np.vstack([early_proto, mid_proto, late_proto, osc_proto])

    labels = []
    for ts in ts_matrix:
        dists = [dtw_distance(ts, p) for p in prototypes]
        labels.append(np.argmin(dists))

    # Standardized bloom timing labels
    shape_map = {
        0: "Bloom-A: Early Season",
        1: "Bloom-B: Mid Season",
        2: "Bloom-C: Late Season",
        3: "Bloom-D: Oscillatory"
    }
    df['Shape_Regime'] = [shape_map[i] for i in labels]

    # Cross-tabs
    ct1 = pd.crosstab(df.get('Typology', pd.Series(dtype=str)), df['Shape_Regime'])
    ct2 = pd.crosstab(df.get('Resilience_Class', pd.Series(dtype=str)), df['Shape_Regime'])

    idx_combo = [df.get('Typology', pd.Series(dtype=str)), df.get('Resilience_Class', pd.Series(dtype=str))]
    try:
        ct3 = pd.crosstab(idx_combo, df['Shape_Regime'])
    except Exception:
        ct3 = pd.DataFrame()

    figs = {}

    # Heatmap (counts)
    if not ct3.empty:
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        sns.heatmap(ct3, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax_hm)
        ax_hm.set_title("Hydro-Ecological Behaviour Matrix (Counts)")
        ax_hm.set_ylabel("Typology + Stability"); ax_hm.set_xlabel("Bloom Timing Regime")
        plt.tight_layout(); figs['heatmap_counts'] = fig_hm

        # Heatmap (normalized)
        ct3_norm = ct3.div(ct3.sum(axis=1) + 1e-9, axis=0)
        fig_hn, ax_hn = plt.subplots(figsize=(10, 6))
        sns.heatmap(ct3_norm, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, ax=ax_hn)
        ax_hn.set_title("Hydro-Ecological Behaviour Matrix (Row %)")
        plt.tight_layout(); figs['heatmap_norm'] = fig_hn

    # Stacked bar
    if not ct1.empty:
        fig_sb, ax_sb = plt.subplots(figsize=(10, 5))
        ct1.plot(kind='bar', stacked=True, colormap='Set2', ax=ax_sb)
        ax_sb.set_title("Bloom Timing by Typology"); ax_sb.set_ylabel("Count")
        plt.xticks(rotation=30); plt.tight_layout(); figs['stacked_bar'] = fig_sb

    # Shape regime distribution
    fig_sd, ax_sd = plt.subplots(figsize=(7, 4))
    sr_counts = df['Shape_Regime'].value_counts()
    ax_sd.bar(sr_counts.index, sr_counts.values, color=['#3498db','#e67e22','#9b59b6','#1abc9c'])
    ax_sd.set_title('Shape Regime Distribution'); ax_sd.set_ylabel('Count')
    plt.xticks(rotation=20); plt.tight_layout(); figs['shape_dist'] = fig_sd

    artifacts = {
        "model_name": "Model2: DTW Shape Classifier",
        "prototypes": prototypes,
        "shape_map": shape_map,
        "hyperparameters": {
            "T": T,
            "num_prototypes": len(prototypes)
        }
    }

    return df, figs, artifacts


# ---------- MODEL 3: WSA + NDCI + NDTI Triple-Stream ----------

def wq_model3_run(df_wq, df_wsa, ndci_cols, ndti_cols, wsa_cols,
                  n_clusters=4, pca_components_each=2):
    """Run Triple-Stream PCA + KMeans."""
    df_wsa_c = df_wsa.copy()
    df_wq_c  = df_wq.copy()

    # Align by ID if present
    if 'ID' in df_wsa_c.columns and 'ID' in df_wq_c.columns:
        df_wsa_c = df_wsa_c.set_index('ID')
        df_wq_c  = df_wq_c.set_index('ID')
        common_ids = df_wsa_c.index.intersection(df_wq_c.index)
    else:
        common_ids = range(min(len(df_wsa_c), len(df_wq_c)))
        df_wsa_c.index = range(len(df_wsa_c))
        df_wq_c.index  = range(len(df_wq_c))
        common_ids = df_wsa_c.index.intersection(df_wq_c.index)

    X_wsa_raw  = df_wsa_c.loc[common_ids, wsa_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_ndci_raw = df_wq_c.loc[common_ids, ndci_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_ndti_raw = df_wq_c.loc[common_ids, ndti_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    df_final = pd.DataFrame(index=common_ids)
    df_final['WSA_Mean']  = np.mean(X_wsa_raw, axis=1)
    df_final['WSA_CV']    = np.std(X_wsa_raw, axis=1) / (df_final['WSA_Mean'].abs() + 1e-6)
    df_final['NDCI_Mean'] = np.mean(X_ndci_raw, axis=1)
    df_final['NDCI_CV']   = np.std(X_ndci_raw, axis=1) / (np.abs(df_final['NDCI_Mean']) + 1e-6)
    df_final['NDTI_Mean'] = np.mean(X_ndti_raw, axis=1)
    df_final['NDTI_CV']   = np.std(X_ndti_raw, axis=1) / (np.abs(df_final['NDTI_Mean']) + 1e-6)

    scaler_ndci = StandardScaler()
    n_comp = min(int(pca_components_each), X_ndci_raw.shape[1])
    pca_ndci = PCA(n_components=n_comp, random_state=42)
    X_pca_ndci = pca_ndci.fit_transform(scaler_ndci.fit_transform(X_ndci_raw))

    scaler_ndti = StandardScaler()
    pca_ndti = PCA(n_components=n_comp, random_state=42)
    X_pca_ndti = pca_ndti.fit_transform(scaler_ndti.fit_transform(X_ndti_raw))

    scaler_wsa = StandardScaler()
    n_comp_wsa = min(int(pca_components_each), X_wsa_raw.shape[1])
    pca_wsa = PCA(n_components=n_comp_wsa, random_state=42)
    X_pca_wsa = pca_wsa.fit_transform(scaler_wsa.fit_transform(X_wsa_raw))

    beh_feats = ['WSA_Mean', 'WSA_CV', 'NDCI_Mean', 'NDCI_CV', 'NDTI_Mean', 'NDTI_CV']
    scaler_beh = StandardScaler()
    X_beh = scaler_beh.fit_transform(df_final[beh_feats])
    X_clustering = np.hstack((X_pca_ndci, X_pca_ndti, X_pca_wsa, X_beh))

    final_k = min(int(n_clusters), len(df_final) - 1)
    model = KMeans(n_clusters=final_k, init='k-means++', n_init=20, random_state=42)
    df_final['Cluster'] = model.fit_predict(X_clustering)

    # Standardized WQ-I to WQ-IV typology labels (rule-based)
    def label_ecology(row):
        if row['NDCI_Mean'] > 0.3 and row['NDTI_Mean'] > 0.1:
            return "WQ-IV: Critical (Algal + Turbid)"
        elif row['NDCI_Mean'] > 0.3:
            return "WQ-III: Stressed (Algal Bloom)"
        elif row['NDTI_Mean'] > 0.1:
            return "WQ-II: Moderate (Turbid/Sediment)"
        else:
            return "WQ-I: Pristine (Clear/Stable)"
    df_final['Typology'] = df_final.apply(label_ecology, axis=1)

    figs = {}

    # Scatter: Algae-Turbidity-Area
    fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
    for typ in df_final['Typology'].unique():
        sub = df_final[df_final['Typology'] == typ]
        ax_sc.scatter(sub['NDTI_Mean'], sub['NDCI_Mean'], s=sub['WSA_Mean'].clip(5, 500),
                      alpha=0.5, label=typ)
    ax_sc.set_xlabel('Mean Turbidity (NDTI)'); ax_sc.set_ylabel('Mean Chlorophyll (NDCI)')
    ax_sc.set_title('Ecological Space: Turbidity vs Algae vs Lake Size')
    ax_sc.legend(fontsize=8); plt.tight_layout(); figs['scatter'] = fig_sc

    # WSA Boxplot by Typology
    fig_bx, ax_bx = plt.subplots(figsize=(8, 5))
    typs = df_final['Typology'].unique()
    data_box = [df_final[df_final['Typology'] == t]['WSA_Mean'].values for t in typs]
    ax_bx.boxplot(data_box, labels=[t[:20] for t in typs])
    ax_bx.set_title('Lake Size by Typology'); ax_bx.set_ylabel('WSA Mean')
    plt.xticks(rotation=20); plt.tight_layout(); figs['boxplot'] = fig_bx

    # Typology distribution
    fig_td, ax_td = plt.subplots(figsize=(8, 4))
    typ_counts = df_final['Typology'].value_counts()
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    ax_td.bar(range(len(typ_counts)), typ_counts.values, color=colors[:len(typ_counts)])
    ax_td.set_xticks(range(len(typ_counts)))
    ax_td.set_xticklabels(typ_counts.index, rotation=20, fontsize=8)
    ax_td.set_title('Typology Distribution'); ax_td.set_ylabel('Count')
    plt.tight_layout(); figs['typology_dist'] = fig_td

    artifacts = {
        "model_name": f"Model3: Triple-Stream (WSA+NDCI+NDTI) (K={final_k})",
        "scaler_ndci": scaler_ndci,
        "scaler_ndti": scaler_ndti,
        "scaler_wsa": scaler_wsa,
        "pca_ndci": pca_ndci,
        "pca_ndti": pca_ndti,
        "pca_wsa": pca_wsa,
        "scaler_beh": scaler_beh,
        "kmeans": model,
        "behavioral_features": beh_feats,
        "hyperparameters": {
            "K": final_k,
            "pca_components_each": int(pca_components_each)
        },
        "metrics": {
            "silhouette": float(silhouette_score(X_clustering, df_final['Cluster'].values)),
            "calinski_harabasz": float(calinski_harabasz_score(X_clustering, df_final['Cluster'].values))
        }
    }

    return df_final, figs, artifacts
