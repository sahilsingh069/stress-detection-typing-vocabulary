"""
Central configuration for the Stress Detection project.

Edit the PATHS section once you have downloaded the real datasets.
Everything else (features, models, evaluation) reads from here.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# PATHS  -- edit these after downloading the datasets
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# SWELL-KW: put the computer-interaction / feature CSV here after registering at
#   http://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html
SWELL_PATH = DATA_DIR / "swell_kw.csv"

# Dreaddit: dreaddit-train.csv / dreaddit-test.csv from the public release
DREADDIT_TRAIN_PATH = DATA_DIR / "dreaddit-train.csv"
DREADDIT_TEST_PATH = DATA_DIR / "dreaddit-test.csv"

# The Kaggle "Stress Detection by Keystroke, App & Mouse Changes" archive,
# extracted so that this folder contains "user 1/", "user 2/"
KEYSTROKE_KAGGLE_DIR = DATA_DIR / "Data"

# --------------------------------------------------------------------------
# LABELS
# --------------------------------------------------------------------------
# Canonical 3-class target used across both pipelines.
STRESS_CLASSES = ["Low", "Medium", "High"]

# SWELL-KW ships condition labels; map them onto our 3-class scheme.
SWELL_CONDITION_MAP = {
    "no stress": "Low",
    "nostress": "Low",
    "n": "Low",
    "neutral": "Low",
    "interruption": "Medium",
    "i": "Medium",
    "time pressure": "High",
    "timepressure": "High",
    "t": "High",
}

# Dreaddit is binary (0 = not stressed, 1 = stressed).
DREADDIT_LABEL_MAP = {0: "Low", 1: "High"}

# Kaggle keystroke dataset self-report values -> 3-class
KAGGLE_STRESS_MAP = {
    "F_Great": "Low",
    "F_Good": "Low",
    "Neutral": "Medium",
    "S_Stressed": "High",
    "V_Stressed": "High",
}

# --------------------------------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# How to deal with class imbalance: "none" | "class_weight" | "smote"
IMBALANCE_STRATEGY = "class_weight"

# Text vectorisation
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)

# Keystroke session windowing (seconds) when building sessions from raw logs
SESSION_WINDOW_SECONDS = 300
LONG_PAUSE_THRESHOLD_MS = 2000
