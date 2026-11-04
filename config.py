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
# (csv, tsv or xlsx all work; rename the file or edit this path)
SWELL_PATH = DATA_DIR / "swell_kw.csv"

# Own collected data: sessions_*.csv + keystrokes_*.csv from the collector
OWN_DIR = DATA_DIR / "own"

# Trained models used by the demo web app (app/server.py)
DEMO_DIR = OUTPUT_DIR / "demo"

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
    # The rest/relax periods ("R") involve no work, so there is no typing
    # behaviour to learn from; they are deliberately left unmapped (dropped).
}

# Binary view of SWELL-KW: neutral work vs. either stressor.
SWELL_BINARY_MAP = {"Low": "NoStress", "Medium": "Stress", "High": "Stress"}

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

# Sessions (a copy task, a chat answer, a demo input) with fewer key presses
# than this are dropped from the typing stream: too few keys for stable timing.
MIN_PRESSES_PER_SESSION = 10

# --------------------------------------------------------------------------
# OWN PAIRED DATA - which columns the models may see
# --------------------------------------------------------------------------
# Typing features computed in Python from keystrokes_*.csv. An explicit
# allow-list, not a deny-list: anything else in sessions_*.csv (block,
# condition, time limit, duration, self-ratings, raw counts) either IS the
# label or is fixed by the timer, and would leak the answer.
#   n_keystrokes is excluded: answer length is capped by the timer.
OWN_TYPING_FEATURES = [
    "typing_speed_cpm", "mean_flight_ms", "std_flight_ms", "median_flight_ms",
    "backspace_rate", "special_key_rate", "long_pause_rate",
    "rhythm_variability", "mean_dwell_ms", "std_dwell_ms",
]

# Linguistic features dropped for own data: word_count measures the timer
# (pressured answers are cut off), not the participant's stress.
PAIRED_EXCLUDED_TEXT_FEATURES = ["word_count"]

# Task types whose typed text is the participant's own words. Copy tasks
# reproduce a fixed passage, so their text carries no vocabulary signal.
# ("free" is the v1 collector's free-writing task, "chat" is v2.)
OWN_TEXT_TASKS = ["chat", "free"]

# Participant-grouped cross-validation folds (capped by participant count)
GROUP_CV_FOLDS = 5
