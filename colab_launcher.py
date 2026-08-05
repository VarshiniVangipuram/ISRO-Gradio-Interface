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
# If the project lives on Google Drive, adjust the path below.
import os, sys

_PROJECT_DIR = "/content/drive/MyDrive/Gradio Interface"
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
