"""
Dataset loaders.

Every loader returns a pandas DataFrame with a column named 'stress_label'
holding values from config.STRESS_CLASSES.

The loaders are deliberately tolerant about column names, because the exact
column headers in SWELL-KW / Dreaddit vary between the raw and preprocessed
releases. If a file is missing, the loader raises FileNotFoundError with the
exact path it looked at, so you know what to download and where to put it.

Use `load_synthetic_*` to develop and test the pipeline before the real
datasets are in place.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ==========================================================================
# helpers
# ==========================================================================
def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column whose lowercased name matches a candidate."""
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # partial match fallback
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def _normalise_swell_label(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return config.SWELL_CONDITION_MAP.get(key)


def _read_any(path: Path) -> pd.DataFrame:
    """Read csv / tsv / xlsx based on suffix."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at: {path}\n"
            f"Download it and place it there, or edit the path in config.py"
        )
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


# ==========================================================================
# SWELL-KW  (typing / computer-interaction stream)
# ==========================================================================
def load_swell(path: Path | None = None) -> pd.DataFrame:
    """
    Load the SWELL-KW computer-interaction feature file.

    Returns a DataFrame of numeric behaviour features plus 'stress_label'.
    """
    path = Path(path or config.SWELL_PATH)
    df = _read_any(path)

    label_col = _find_column(
        df, ["condition", "stress_label", "stress", "label", "class", "cond"]
    )
    if label_col is None:
        raise ValueError(
            f"Could not find a condition/label column in {path.name}. "
            f"Columns found: {list(df.columns)[:25]}"
        )

    df = df.copy()
    df["stress_label"] = df[label_col].map(_normalise_swell_label)

    unmapped = df["stress_label"].isna().sum()
    if unmapped:
        warnings.warn(
            f"{unmapped} rows had condition values not in SWELL_CONDITION_MAP "
            f"and were dropped. Unique raw values: "
            f"{df.loc[df['stress_label'].isna(), label_col].unique()[:10]}"
        )
    df = df.dropna(subset=["stress_label"])

    # drop obvious identifier columns so they don't leak into the model
    drop_like = ["subject", "participant", "id", "session", "datetime", "time"]
    to_drop = [
        c
        for c in df.columns
        if c != "stress_label"
        and any(k in c.lower() for k in drop_like)
    ]
    df = df.drop(columns=to_drop + [label_col], errors="ignore")

    return df.reset_index(drop=True)


# ==========================================================================
# Dreaddit  (text / vocabulary stream)
# ==========================================================================
def load_dreaddit(
    train_path: Path | None = None, test_path: Path | None = None
) -> pd.DataFrame:
    """
    Load Dreaddit and return a DataFrame with columns ['text', 'stress_label'].
    Train and test files are concatenated; we re-split later ourselves.
    """
    train_path = Path(train_path or config.DREADDIT_TRAIN_PATH)
    frames = [_read_any(train_path)]

    test_path = Path(test_path or config.DREADDIT_TEST_PATH)
    if test_path.exists():
        frames.append(_read_any(test_path))

    df = pd.concat(frames, ignore_index=True)

    text_col = _find_column(df, ["text", "post", "body", "content", "sentence"])
    label_col = _find_column(df, ["label", "stress", "target", "class"])
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not locate text/label columns in Dreaddit. "
            f"Columns found: {list(df.columns)[:25]}"
        )

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "stress_label": df[label_col]
            .astype(int)
            .map(config.DREADDIT_LABEL_MAP),
        }
    )
    return out.dropna(subset=["stress_label"]).reset_index(drop=True)


# ==========================================================================
# Kaggle keystroke dataset (raw event logs -> sessions)
# ==========================================================================
def load_kaggle_keystrokes(root: Path | None = None) -> pd.DataFrame:
    """
    Load the Kaggle 'Stress Detection by Keystroke, App & Mouse Changes'
    dataset. Returns raw keystroke events joined to the nearest self-report,
    with columns: user, timestamp, key, event, stress_label. When the source
    supplies both press and release timestamps, each input keystroke becomes
    two events so downstream feature extraction can calculate dwell time.

    Feature aggregation into sessions happens in features.typing_features.
    """
    root = Path(root or config.KEYSTROKE_KAGGLE_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Expected extracted Kaggle data at: {root}")

    frames = []
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ks_path = user_dir / "keystrokes.tsv"
        cond_path = user_dir / "usercondition.tsv"
        if not ks_path.exists() or not cond_path.exists():
            continue

        ks = pd.read_csv(ks_path, sep="\t")
        cond = pd.read_csv(cond_path, sep="\t")

        ks.columns = [c.strip() for c in ks.columns]
        cond.columns = [c.strip() for c in cond.columns]

        press_col = _find_column(
            ks, ["press_time", "press time", "pressed_time", "press"]
        )
        release_col = _find_column(
            ks,
            ["release_time", "release time", "relase_time", "relase time", "release", "relase"],
        )
        ks_time = _find_column(ks, ["time", "timestamp", "datetime"])
        cond_time = _find_column(cond, ["time", "timestamp", "datetime"])
        cond_stress = _find_column(cond, ["stress_val", "stress", "label"])
        key_col = _find_column(ks, ["key"])
        event_col = _find_column(ks, ["event", "action", "type"])

        if not all([press_col or ks_time, cond_time, cond_stress, key_col]):
            warnings.warn(f"Skipping {user_dir.name}: unexpected columns")
            continue

        cond[cond_time] = pd.to_datetime(cond[cond_time], errors="coerce")
        cond = cond.dropna(subset=[cond_time]).sort_values(cond_time)

        # The Kaggle files contain one row per key with separate press/release
        # timestamps. Preserve both; treating them as a single event throws
        # away dwell time, one of this project's intended core features.
        event_frames = []
        if press_col:
            event_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(ks[press_col], errors="coerce"),
                        "key": ks[key_col],
                        "event": "press",
                    }
                )
            )
        elif ks_time:
            event_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(ks[ks_time], errors="coerce"),
                        "key": ks[key_col],
                        "event": ks[event_col] if event_col else "press",
                    }
                )
            )
        if release_col:
            event_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(ks[release_col], errors="coerce"),
                        "key": ks[key_col],
                        "event": "release",
                    }
                )
            )

        events = pd.concat(event_frames, ignore_index=True)
        events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

        cond["stress_label"] = cond[cond_stress].map(config.KAGGLE_STRESS_MAP)
        cond = cond.dropna(subset=["stress_label"])
        if cond.empty or events.empty:
            continue

        # Attach each event to the closest self-report within 30 minutes.
        merged = pd.merge_asof(
            events,
            cond[[cond_time, "stress_label"]],
            left_on="timestamp",
            right_on=cond_time,
            direction="nearest",
            tolerance=pd.Timedelta(minutes=30),
        ).dropna(subset=["stress_label"])

        merged["user"] = user_dir.name
        frames.append(
            merged[["user", "timestamp", "key", "event", "stress_label"]]
        )

    if not frames:
        raise ValueError(f"No usable user folders found under {root}")

    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# Synthetic fallbacks -- lets you run the whole pipeline before the real
# datasets arrive. NEVER report results from these as project results.
# ==========================================================================
def load_synthetic_typing(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic typing-behaviour features with a learnable stress signal."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(config.STRESS_CLASSES, size=n, p=[0.45, 0.35, 0.20])
    shift = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
    s = np.array([shift[l] for l in labels])

    df = pd.DataFrame(
        {
            "mean_dwell_ms": rng.normal(95 - 6 * s, 14),
            "std_dwell_ms": rng.normal(22 + 5 * s, 5),
            "mean_flight_ms": rng.normal(170 + 25 * s, 30),
            "std_flight_ms": rng.normal(60 + 18 * s, 12),
            "typing_speed_cpm": rng.normal(260 - 30 * s, 40),
            "backspace_rate": rng.normal(0.05 + 0.03 * s, 0.015).clip(0, 1),
            "long_pause_rate": rng.normal(0.02 + 0.02 * s, 0.01).clip(0, 1),
            "rhythm_variability": rng.normal(0.30 + 0.12 * s, 0.06),
            "stress_label": labels,
        }
    )
    return df


def load_synthetic_text(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic stress-labelled text for smoke-testing the NLP pipeline."""
    rng = np.random.default_rng(seed)
    stressed = [
        "i am completely overwhelmed and cannot cope with this deadline",
        "everything is falling apart and i feel anxious all the time",
        "i can't sleep, my mind keeps racing about work problems",
        "so much pressure right now, i feel like i am failing badly",
        "exhausted, worried and honestly close to breaking point",
    ]
    calm = [
        "had a relaxed weekend with family and enjoyed the weather",
        "work is going smoothly and i finished everything on time",
        "feeling calm and pretty happy with how things are going",
        "nice quiet evening, read a book and went to bed early",
        "everything is under control, no complaints at all today",
    ]
    labels = rng.choice(["Low", "High"], size=n, p=[0.55, 0.45])
    texts = [
        rng.choice(calm if l == "Low" else stressed) for l in labels
    ]
    return pd.DataFrame({"text": texts, "stress_label": labels})


# ==========================================================================
# Own collected data (from collector/typing_collector.html)
# ==========================================================================
def load_own_sessions(path: Path | None = None) -> pd.DataFrame:
    """
    Load sessions_*.csv exported by the collector tool.

    This is the PAIRED dataset: every row has typing-behaviour features AND
    the typed text AND one stress label, which is what makes early fusion
    possible. Pass the result to features.typing_features.prepare_feature_frame
    for the typing stream, or use the 'text' column for the vocabulary stream.
    """
    path = Path(path or (config.DATA_DIR / "own"))
    files = (
        sorted(path.glob("sessions_*.csv")) if path.is_dir() else [path]
    )
    if not files:
        raise FileNotFoundError(
            f"No sessions_*.csv found in {path}. Export them from "
            f"collector/typing_collector.html first."
        )

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    required = {"participant_id", "session_id", "typed_text", "self_report_stress"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Collected-session file is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    df = df.rename(columns={"typed_text": "text",
                            "self_report_stress": "stress_label"})
    df["stress_label"] = df["stress_label"].astype(str).str.strip().str.title()
    df = df[df["stress_label"].isin(config.STRESS_CLASSES)]
    print(f"  loaded {len(df)} sessions from {len(files)} file(s), "
          f"{df['participant_id'].nunique()} participants")
    return df.reset_index(drop=True)


def load_own_keystrokes(path: Path | None = None) -> pd.DataFrame:
    """
    Load keystrokes_*.csv exported by the collector tool and reshape it into
    the event format expected by features.typing_features.features_from_events.
    """
    path = Path(path or (config.DATA_DIR / "own"))
    files = (
        sorted(path.glob("keystrokes_*.csv")) if path.is_dir() else [path]
    )
    if not files:
        raise FileNotFoundError(f"No keystrokes_*.csv found in {path}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    required = {"participant_id", "session_id", "event", "key", "t_ms", "stress_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Collected-keystroke file is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    df = df.rename(columns={"event": "ev"})
    # Each export's timestamps start at zero. A session-specific user key keeps
    # different tasks from being accidentally aggregated into one 1970 window.
    df["user"] = (
        df["participant_id"].astype(str) + "__" + df["session_id"].astype(str)
    )
    # t_ms is relative to session start -> synthesise an absolute timestamp
    df["timestamp"] = pd.to_datetime(df["t_ms"], unit="ms", origin="unix")
    df["event"] = df["ev"].map({"down": "press", "up": "release"})
    return df[["user", "timestamp", "key", "event", "stress_label"]]
