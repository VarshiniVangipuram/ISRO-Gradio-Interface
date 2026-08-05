# Multi-Temporal Surface Water Body Classification & Water Quality Assessment Framework

An end-to-end machine learning platform for classifying seasonal ponds from multi-temporal satellite-derived time-series data and assessing surface water quality using remote-sensing indices. Built with a Gradio web interface and designed to run in Google Colab with Google Drive model persistence.

---

## Table of Contents

- [Overview](#overview)
- [Business Use Case](#business-use-case)
- [Project Architecture](#project-architecture)
  - [1. Pond Classification Module](#1-pond-classification-module)
  - [2. Water Quality Assessment Module](#2-water-quality-assessment-module)
- [ML Models](#ml-models)
  - [Pond Classification Models](#pond-classification-models)
  - [Water Quality Models](#water-quality-models)
- [Data Pipeline](#data-pipeline)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Overview

This project addresses two interconnected problems in surface-water resource management:

1. **Pond Classification** — Given a time-series of surface-water area measurements (derived from satellite imagery over many time steps), classify each water body into one of four ecological trajectories: **Persistent**, **Rejuvenated**, **Newly Formed**, or **Extinct**.

2. **Water Quality Assessment** — Using remote-sensing indices (NDCI for chlorophyll/algal blooms, NDTI for turbidity, and Water Surface Area), cluster water bodies into ecological quality typologies ranging from **Pristine (WQ-I)** to **Critical (WQ-IV)**.

The entire system is exposed through a single Gradio web application with a modern, card-based UI featuring two main tabs (Pond Classification and Water Quality), each with sub-tabs for **Advanced training/experimentation** and **Operational prediction** using pre-saved models.

---

## Business Use Case

### Why this matters

Seasonal and semi-permanent water bodies are critical for agriculture, drinking water, biodiversity, and groundwater recharge in semi-arid regions. However, tracking their long-term behaviour and water quality at scale is impractical through field surveys alone.

### Applications

- **Government & Water Resource Agencies** (e.g., ISRO, state water boards): Monitor thousands of ponds across districts using satellite time-series data without manual field visits. Identify ponds that are drying up (extinct), recovering (rejuvenated), or newly formed — enabling targeted restoration or conservation action.

- **Environmental Monitoring**: Detect ecological stress early. The water quality module flags water bodies shifting toward eutrophic or hyper-eutrophic conditions based on algal bloom frequency, turbidity trends, and resilience scores — supporting early intervention before irreversible degradation.

- **Urban Planning & Land Use**: Newly formed and extinct ponds signal land-use changes, encroachment, or hydrological shifts. Planners can use these classifications to inform zoning, protection, or restoration decisions.

- **Agricultural Advisory**: Persistent and rejuvenated ponds indicate reliable water availability for irrigation. Extinct ponds signal risk for dependent farming communities.

- **Research & Policy**: The framework provides reproducible, model-driven classifications that can be compared across regions and time periods, supporting evidence-based water governance policy.

### Key Value Propositions

| Capability | Benefit |
|---|---|
| Automated 4-class pond trajectory classification | Replaces manual survey with scalable satellite-based monitoring |
| Three-Stage cascaded ensemble | Handles severe class imbalance (rare classes like "rejuvenated") with high recall |
| T-agnostic Universal model | Works across datasets with different numbers of time steps — no manual alignment needed |
| Unsupervised water quality clustering | No labelled water-quality data required; discovers natural ecological groupings |
| Model deployment & persistence | Trained models are saved to Google Drive and can be loaded for operational batch prediction |
| Interactive Gradio UI | Non-technical users can train, evaluate, deploy, and predict without writing code |

---

## Project Architecture

The application is built as a modular Python package (`pond_wq_app/`) with a thin entry point (`app.py`). The high-level structure is:

```
Gradio App (demo)
├── 🌊 Pond Classification Analysis (outer tab)
│   ├── Advanced (inner tab) — Training & Experimentation
│   │   ├── Step 1: Dataset selection (preloaded from Drive or custom upload)
│   │   ├── Step 2: Class selection (rejuvenated, persistent, newly formed, extinct)
│   │   ├── Step 3: Model architecture selection
│   │   ├── Model-specific hyperparameter configuration
│   │   ├── Train → Evaluate → Deploy → Save to Drive
│   │   └── Experiment history with undo/revert
│   └── Operational (inner tab) — Batch Prediction with saved models
│       ├── Upload dataset
│       ├── Select saved model from Drive
│       ├── Run prediction
│       └── Download predictions CSV
│
└── 💧 Water Quality Assessment (outer tab)
    ├── Advanced (inner tab) — Unsupervised Analysis
    │   ├── Load Excel (WQ + WSA sheets)
    │   ├── Run Model 1 / Model 2 / Model 3
    │   ├── Visualizations (elbow, silhouette, behavioral space, temporal fingerprints)
    │   └── Download combined results CSV
    └── Operational (inner tab) — Prediction with saved WQ models
```

### Model Registry & Persistence

- **Operational models** are stored as `.pkl` bundles in Google Drive under `Best_Models/Pond_Classification_Models/Operational_Active/` and `Best_Models/Water_Quality_Models/Operational_Active/`.
- On startup, the app scans these folders and auto-discovers compatible model bundles.
- Models trained during a session are first deployed to session-only storage. The user must explicitly click **"Save to Drive"** to persist them.
- Each model bundle is a self-contained dictionary including the estimator(s), label encoder, scaler, feature selector, feature names, preprocessing pipeline version, hyperparameters, and thresholds.

---

## ML Models

### Pond Classification Models

The system supports five model architectures, each selectable from the UI with model-specific hyperparameters:

#### 1. Three-Stage Cascade Ensemble (XGBoost + XGBoost + CatBoost)

A cascaded binary-classifier architecture designed to handle severe class imbalance. Each stage uses ADASYN oversampling and threshold optimization.

```
Input features (up to 11 feature families)
    │
    ├── StandardScaler → RandomForest feature selection (SelectFromModel)
    │
    ├── Stage 1: XGBoost binary — "Is this pond Rejuvenated?"
    │       └── If probability ≥ threshold_r → classify as "rejuvenated"
    │
    ├── Stage 2: XGBoost binary — "Is this pond Newly Formed?"
    │       └── If probability ≥ threshold_n → classify as "newly formed"
    │
    └── Stage 3: CatBoost binary — "Persistent vs Extinct"
            └── Remaining samples classified by CatBoost
```

**Feature families** (toggleable, 11 groups):
- Basic statistics (mean, median, std, max, min, range, CV)
- Water presence (zero count, zero %, nonzero count, nonzero %)
- Temporal quarters (Q1–Q4 means)
- Trend analysis (overall slope on nonzero values)
- Rejuvenation signals (min location ratio, before/after min means, recovery ratio, recovery slope)
- V-shape indicators (first/second half means, middle section, V-shape ratio)
- Zero patterns (max consecutive zeros, zeros at end)
- Variability (total area changes, mean change)
- Peaks (number of peaks, peak distance)
- Advanced distribution (skewness, kurtosis, percentiles, IQR, start/end ratio, volatility, mean return)
- Discriminative (early water flag, starts-from-zero flag, late period growth)

**Class imbalance handling**: ADASYN oversampling per stage, `scale_pos_weight` tuning, and grid-search threshold optimization (maximizing macro-F1).

#### 2. Single XGBoost (76-feature)

A single multi-class XGBoost classifier using a rich 76-feature extraction (`extractfeaturesfast`) covering basic stats, water presence, temporal quarters, trend, rejuvenation, V-shape, zero patterns, variability, stability, peaks, distribution stats, start/end ratios, returns, thirds, zero span, and discriminative flags.

- Tempered class weights (balanced sample weights × alpha, capped)
- Early stopping with validation set
- GPU support (`tree_method="gpu_hist"`)

#### 3. Universal Generalized Ensemble (T-agnostic)

Uses fractional/normalized features (`extract_universal_features`) that are independent of the time-series length T. This model can be trained on one dataset and applied to another with a different number of time steps — the key advantage for cross-region generalization.

- Normalized time axis [0, 1] for all slope calculations
- Fractional windowing (quarters by proportion, not fixed indices)
- Scale-agnostic statistics (CV, stability index, fractional zero patterns)

#### 4. Shape-Only Models (13-feature and 8-feature)

Lightweight models using only shape-describing features after per-sample min-max normalization. Two variants:
- **13-feature** (`extract_shape_only_features`): zero structure, transitions, consecutive zeros, volatility, peaks, early/late ratios, monotonicity, U-shape detection
- **8-feature** (`extract_shape_features_8`): compact subset for fast inference

These are the legacy "USO" (Universal Shape Only) models pre-stored on Drive.

#### 5. Custom Model Templates

Baseline classifiers using raw `TS_###` columns as features (no engineered features):
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost (simple)

Each with optional StandardScaler and balanced class weights.

### Water Quality Models

Three unsupervised models that run sequentially, accumulating results into a master DataFrame:

#### Model 1: PCA + KMeans + Temporal Stability + Resilience

```
NDCI + NDTI temporal columns
    │
    ├── Feature engineering: NDCI mean, NDTI mean, NDCI CV, NDCI std,
    │   TS amplitude, bloom frequency, resilience score
    │
    ├── PCA on scaled temporal data (reduces dimensionality)
    │
    ├── Hybrid feature space: PCA components + behavioral features
    │
    ├── K-search: Elbow, Silhouette, Calinski-Harabasz scores
    │
    └── Final KMeans clustering → Typology (WQ-I to WQ-IV)
                                    → TS Class (Stable / Seasonal / Acute)
                                    → Resilience Class (Low / Med / High)
```

**Typology scale**:
- **WQ-I: Pristine (Oligotrophic)** — low NDCI, stable
- **WQ-II: Moderate (Mesotrophic)** — moderate nutrient load
- **WQ-III: Stressed (Eutrophic)** — high algal activity
- **WQ-IV: Critical (Hyper-Eutrophic)** — severe degradation

#### Model 2: DTW Temporal Shape Classifier

Uses Dynamic Time Warping to classify each water body's NDCI temporal profile against four prototype shapes:

| Prototype | Label |
|---|---|
| Early-season Gaussian peak | Bloom-A: Early Season |
| Mid-season Gaussian peak | Bloom-B: Mid Season |
| Late-season Gaussian peak | Bloom-C: Late Season |
| Sinusoidal oscillation | Bloom-D: Oscillatory |

Produces cross-tabulations (Typology × Shape Regime, Resilience × Shape Regime) and heatmap visualizations.

#### Model 3: Triple-Stream PCA + KMeans (WSA + NDCI + NDTI)

Aligns Water Surface Area (WSA) data with water quality indices by ID, then applies independent PCA to each stream before concatenating into a unified clustering space.

```
WSA stream  →  StandardScaler  →  PCA  ─┐
NDCI stream →  StandardScaler  →  PCA  ─┤  →  hstack  →  KMeans  →  Typology
NDTI stream →  StandardScaler  →  PCA  ─┘         ↑
                                          + 6 behavioral features
                                            (WSA/NDCI/NDTI mean & CV)
```

**Rule-based typology**:
- NDCI > 0.3 AND NDTI > 0.1 → **WQ-IV: Critical (Algal + Turbid)**
- NDCI > 0.3 → **WQ-III: Stressed (Algal Bloom)**
- NDTI > 0.1 → **WQ-II: Moderate (Turbid/Sediment)**
- Otherwise → **WQ-I: Pristine (Clear/Stable)**

---

## Data Pipeline

### Pond Classification Data Flow

```
Raw CSV/Excel (col_1_sum, col_2_sum, ..., final_status)
    │
    ├── Auto-detect label column (finalstatus / status / value-based)
    ├── Normalize status labels (extint→extinct, rejuv→rejuvenated, etc.)
    ├── Auto-detect time-series columns (col_*_sum pattern or all numeric)
    ├── Remove rows with NaN/empty labels
    ├── Resample all series to unified length T (linear interpolation)
    │
    └── Unified format: [ID, TS_000, TS_001, ..., TS_{T-1}, label]
            │
            ├── Model-specific feature extraction
            ├── Train/test split (stratified)
            ├── Scaling + feature selection (model-dependent)
            ├── Class imbalance handling (ADASYN / tempered weights)
            ├── Training
            ├── Evaluation (accuracy, macro-F1, confusion matrix, classification report)
            │
            └── Deploy → Session storage → (optional) Save to Google Drive
```

### Water Quality Data Flow

```
Excel file with 2 sheets:
  Sheet 0 (WSA): ID, col_1_sum, col_2_sum, ... (water surface area time-series)
  Sheet 1 (WQ):  ID, NDCI_1, NDCI_2, ..., NDTI_1, NDTI_2, ... (water quality indices)
    │
    ├── Auto-detect NDCI, NDTI, WSA column groups
    │
    ├── Model 1: PCA + KMeans → Typology + TS_Class + Resilience_Class
    ├── Model 2: DTW shape classification → Shape_Regime (requires Model 1)
    ├── Model 3: Triple-stream clustering → Typology (independent)
    │
    └── Master DataFrame accumulates all model results (M1_*, M2_*, M3_*)
        → Export to CSV
```

---

## Technology Stack

| Category | Tools |
|---|---|
| **UI Framework** | Gradio (Blocks API with custom CSS/JS) |
| **ML Libraries** | XGBoost, CatBoost, scikit-learn (RandomForest, SVC, KNN, LogisticRegression, KMeans, PCA, StandardScaler, SelectFromModel) |
| **Imbalanced Learning** | imbalanced-learn (ADASYN, SMOTE) |
| **Time-Series Analysis** | Dynamic Time Warping (custom implementation), scipy.stats |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Model Persistence** | joblib (`.pkl` bundles) |
| **Environment** | Google Colab + Google Drive |
| **Data Formats** | CSV, Excel (.xlsx) |

---

## Repository Structure

```
.
├── app.py                # Main entry point — boot, discovery, launch
├── colab_launcher.py     # Thin Colab cell (paste & run)
├── interface.py          # Original monolithic script (kept for reference)
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
└── pond_wq_app/          # Modular application package
    ├── __init__.py
    ├── config.py          # Paths, constants, global registries, UI markdown
    ├── state.py           # State management, experiment history, undo
    ├── registry.py        # Model discovery, deployment, persistence, caching
    ├── controllers.py     # Orchestration: run_experiment, operational_predict,
    │                       # wq_run_analysis, wq_operational_predict, toggles,
    │                       # refresh helpers, init_state, download_best_model
    │
    ├── data/              # Data loading & validation
    │   ├── __init__.py
    │   ├── loader.py       # CSV/Excel reading, preprocessing, label normalization
    │   ├── drive_utils.py  # Google Drive download helpers, filename resolution
    │   ├── validation.py   # Dataset validation (training & operational)
    │   └── wq_loader.py    # WQ Excel loading, column detection, autoload
    │
    ├── features/          # Feature engineering
    │   ├── __init__.py
    │   ├── shape.py        # Shape-only features (13 & 8 feature variants)
    │   ├── fast.py         # 76-feature fast extraction
    │   ├── discriminative.py # Discriminative features with group gating
    │   ├── universal.py    # T-agnostic universal features
    │   └── utils.py        # Shared feature utilities
    │
    ├── models/            # ML model implementations
    │   ├── __init__.py
    │   ├── three_stage.py  # Three-Stage cascaded ensemble (XGB+XGB+CatBoost)
    │   ├── single_xgb.py   # Single XGBoost classifier
    │   ├── universal.py    # Generalized Universal ensemble (T-agnostic)
    │   ├── custom.py       # Custom template models (LR, SVC, KNN, RF, XGB)
    │   └── wq.py           # Water Quality models (PCA+KMeans, DTW, Triple-Stream)
    │
    ├── ui/                # Gradio UI layer
    │   ├── __init__.py
    │   ├── css.py          # Custom CSS (~2,600 lines)
    │   ├── js.py           # Dropdown portal & slider sync JavaScript
    │   ├── render.py       # HTML rendering functions (dashboards, reports, cards)
    │   ├── plots.py        # Matplotlib/seaborn plot generators
    │   └── layout.py       # Gradio Blocks builder + event wiring
    │
    └── tests/             # Unit tests
        └── __init__.py
```

> **Note**: Pre-trained model `.pkl` files and datasets are stored on Google Drive and are not included in this repository. The application auto-discovers models from the Drive folders on startup. To use with your own data, place model bundles in the configured Drive paths or upload datasets directly through the UI.

---

## How to Run

### In Google Colab (recommended)

1. Upload the project folder to your Google Drive at `MyDrive/Gradio Interface/`.
2. Open a new Colab notebook and run:

```python
# Option A: Use the launcher script
%run "/content/drive/MyDrive/Gradio Interface/colab_launcher.py"

# Option B: Manual launch
!pip -q install gradio pandas numpy scikit-learn imbalanced-learn joblib xgboost catboost openpyxl matplotlib seaborn scipy statsmodels

from google.colab import drive
drive.mount("/content/drive")

import sys, os
sys.path.insert(0, "/content/drive/MyDrive/Gradio Interface")
os.chdir("/content/drive/MyDrive/Gradio Interface")

from app import main
main()
```

3. Click the generated Gradio public URL to open the web interface.

### Prerequisites

- Google Drive folder structure:
  - `MyDrive/Pond Classification/Data/` — pond datasets (CSV/Excel)
  - `MyDrive/Water_Quality_Exports/` — water quality datasets (Excel)
  - `MyDrive/Best_Models/Pond_Classification_Models/Operational_Active/` — saved pond models
  - `MyDrive/Best_Models/Water_Quality_Models/Operational_Active/` — saved WQ models
- The app gracefully handles missing folders — it starts with empty registries and allows dataset upload through the UI.

### Installing Dependencies Locally

```bash
pip install gradio pandas numpy scikit-learn imbalanced-learn joblib xgboost catboost openpyxl matplotlib seaborn scipy statsmodels
```

When running outside Colab, Google Drive mount is automatically skipped and local `datasets/` folders are used as fallback.
