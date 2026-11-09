"""
config.py
=========
Central configuration: paths, constants, and global registries.

All Colab-specific paths and shared constants live here so that every
other module imports from a single source of truth.
"""

import tempfile
from pathlib import Path

# -----------------------------
# DATA + DRIVE MODEL FOLDERS
# -----------------------------

# Project root = the folder containing the pond_wq_app package. Datasets and
# model folders are expected to live INSIDE this root so the whole project is
# a single, movable Drive folder. The old top-level MyDrive locations are kept
# as fallbacks so nothing breaks during the move.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVE_ROOT = Path("/content/drive/MyDrive")


def _resolve_folder(*candidates):
    """Return the first candidate directory that exists, else the first one."""
    for candidate in candidates:
        if Path(candidate).is_dir():
            return Path(candidate)
    return Path(candidates[0])


DATASET_FOLDER = str(
    _resolve_folder(
        PROJECT_ROOT / "Data" / "Pond Classification" / "Data",
        PROJECT_ROOT / "Data",
        DRIVE_ROOT / "Pond Classification" / "Data",
    )
)

BEST_MODELS_ROOT = _resolve_folder(
    PROJECT_ROOT / "Best_Models",
    DRIVE_ROOT / "Best_Models",
)
POND_OPERATIONAL_MODELS_FOLDER = (
    BEST_MODELS_ROOT / "Pond_Classification_Models" / "Operational_Active"
)
WQ_OPERATIONAL_MODELS_FOLDER = (
    BEST_MODELS_ROOT / "Water_Quality_Models" / "Operational_Active"
)

# Deploying from the UI is session-only. Models are first written to local
# Colab runtime storage and become persistent only when the user explicitly
# clicks the separate "Save Deployed Model to Drive" button.
SESSION_DEPLOYMENTS_ROOT = Path("/content/session_deployed_models")
POND_SESSION_DEPLOYMENTS_FOLDER = SESSION_DEPLOYMENTS_ROOT / "Pond_Classification"
WQ_SESSION_DEPLOYMENTS_FOLDER = SESSION_DEPLOYMENTS_ROOT / "Water_Quality"

# Kept for compatibility with older helper functions.
TRAINED_MODELS_FOLDER = str(POND_SESSION_DEPLOYMENTS_FOLDER)
WQ_TRAINED_MODELS_FOLDER = str(WQ_SESSION_DEPLOYMENTS_FOLDER)

# Prediction/analysis downloads are generated in runtime storage only.
# Gradio only serves output files located in allowed_paths, the current
# working directory, or the system temp directory. Keeping downloads under
# tempfile.gettempdir() makes them servable regardless of how launch()
# is invoked in the notebook.
GENERATED_RESULTS_FOLDER = Path(tempfile.gettempdir()) / "generated_results"
GENERATED_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Persistent results archive inside the project folder. Every generated CSV is
# written here as well as to the temp folder, so downloads are both durable on
# Drive and servable by Gradio. Best-effort: failures must not break the app.
RESULTS_FOLDER = PROJECT_ROOT / "Results"
try:
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# =============================
# WATER QUALITY CONFIGURATION
# =============================
WQ_DATASET_FOLDER = str(
    _resolve_folder(
        PROJECT_ROOT / "Data" / "Water_Quality_Exports",
        PROJECT_ROOT / "Water_Quality_Exports",
        DRIVE_ROOT / "Water_Quality_Exports",
    )
)
WQ_DATASET_PATH = str(Path(WQ_DATASET_FOLDER) / "HYD_WSA_WQ_Feb2026.xlsx")
WQ_SHEET_WQ = 1
WQ_SHEET_WSA = 0

# -----------------------------
# CLASS LABELS
# -----------------------------

# All known class names (superset used for auto-detection only)
ALL_KNOWN_CLASSES = ["rejuvenated", "persistent", "newly formed", "extinct"]

# Default target classes used when none are explicitly selected
DEFAULT_TARGET_CLASSES = ["rejuvenated", "persistent", "newly formed", "extinct"]

# -----------------------------
# RUNTIME REGISTRIES
# -----------------------------
# These dictionaries are populated at boot time by scanning Google Drive
# and are updated when the user deploys or saves models.

OPERATIONAL_MODELS = {}
OPERATIONAL_LABEL_ENCODERS = {}
OPERATIONAL_MODEL_METADATA = {}
WQ_OPERATIONAL_MODELS = {}

# These dictionaries only mark models deployed during the current session.
# Persistent discovery is handled by the Drive folder registries above.
_USER_TRAINED_MODELS = {}
_USER_TRAINED_WQ_MODELS = {}

# Cache loaded models to avoid repeatedly reading large PKL files.
_OPERATIONAL_MODEL_CACHE = {}
_WQ_OPERATIONAL_MODEL_CACHE = {}

# Cache the original filename returned by Google Drive download responses.
_DRIVE_URL_FILENAME_CACHE = {}

# -----------------------------
# UI MARKDOWN CONSTANTS
# -----------------------------
DATASET_REQUIREMENTS_MD = """
### 📋 Dataset Requirements

Your dataset must include the following:

#### 1️⃣ Label Column (OPTIONAL in Operational Mode)
- **Status Column**: Confirming ground truth (e.g., `finalstatus`).
- **MANDATORY for Training**: Required in the **Workspace** tab to train or evaluate models.
- **OPTIONAL for Prediction**: In the **Operational** tab, you can upload data without this column to get pure predictions (metrics/plots will simply be hidden).
- **Accepted names**: `finalstatus`, `final status`, `final_status`, `final-status`.
- **Accepted classes**: `rejuvenated`, `persistent`, `newly formed`, `extinct` (variants like `rejuv`, `stable` are auto-mapped).

#### 2️⃣ ID Column (OPTIONAL)
- Column name: `ID` or `id`. Auto-generated if missing.

#### 3️⃣ Time-Series Numeric Columns (MANDATORY)
- **Minimum 5 numeric columns** required.
- Each row represents one entity's temporal sequence.
- Supports any naming (e.g., `2019`, `2020` or `col_01`, `col_02`).
- All temporal values must be numeric.

---
#### 💡 Pro-Tip
For the best results in **Operational Mode**, use the same column format (temporal range and order) as the dataset the model was originally trained on.
"""
