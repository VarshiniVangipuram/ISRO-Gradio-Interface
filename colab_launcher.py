#!/usr/bin/env python3
"""
Colab launcher — paste this entire cell into a Google Colab notebook.

It installs dependencies, clones/pulls the project, and launches the Gradio app.
"""

# ─── 1. Install dependencies ─────────────────────────────────────────────
import subprocess, sys

def _pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

_pip_install(
    "gradio", "pandas", "numpy", "scikit-learn", "imbalanced-learn",
    "joblib", "xgboost", "catboost", "openpyxl", "matplotlib",
    "seaborn", "scipy", "statsmodels",
)

# ─── 2. Mount Google Drive ────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Google Colab. Skipping Drive mount.")

# ─── 3. Add project to sys.path ───────────────────────────────────────────
# Resolve the project root from this file's own location so the launcher
# works no matter which Drive folder copy it is run from. If you paste this
# script into a notebook cell instead of running it via %run, set
# _PROJECT_DIR_OVERRIDE to the project folder path.
import os, sys

_PROJECT_DIR_OVERRIDE = None  # e.g. "/content/drive/MyDrive/Complete_Pond_Classification_Final"

if _PROJECT_DIR_OVERRIDE:
    _PROJECT_DIR = _PROJECT_DIR_OVERRIDE
else:
    try:
        _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is undefined when pasted into a notebook cell
        _PROJECT_DIR = os.getcwd()

if not os.path.isdir(_PROJECT_DIR):
    # Fallback: assume the notebook is opened from the project root
    _PROJECT_DIR = os.getcwd()

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

os.chdir(_PROJECT_DIR)
print(f"Working directory: {_PROJECT_DIR}")

# ─── 4. Launch the app ────────────────────────────────────────────────────
from app import main
main()
