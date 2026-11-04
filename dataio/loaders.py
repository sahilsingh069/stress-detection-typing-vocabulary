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
# SWELL-KW (Koldijk et al., 2014) logs 25 office workers writing reports and
# e-mails under three conditions: neutral (N), interruptions (I) and time
# pressure (T), plus rest periods (R). Its feature release has one row per
# minute with computer-interaction features named Sn* (SnKeyStrokes,
# SnErrorKeys, SnMouseDistance, SnAppChange, ...), alongside facial,
# posture and heart-rate features and questionnaire answers.
#
# Only the computer-interaction features belong to this project's typing
# stream. Questionnaire columns (Stress, Valence, MentalEffort, NASA-TLX ...)
# are the participant's own ratings, i.e. the label in another form, so they
# are never used as inputs.

SWELL_PARTICIPANT_CANDIDATES = ["pp", "participant", "subject", "subject_id",
                                "subjectid", "user", "person"]
SWELL_CONDITION_CANDIDATES = ["condition", "c", "cond", "stress_label",
                              "label", "class"]

# Keyword hints for computer-interaction columns when they are not Sn*-prefixed
SWELL_INTERACTION_HINTS = ["mouse", "click", "wheel", "drag", "key", "char",
                           "space", "app", "tab", "shortcut", "direction",
                           "error", "special"]

# Questionnaire / self-report columns: the label in disguise, never features
SWELL_SELF_REPORT_COLUMNS = {
    "stress", "valence", "arousal", "dominance", "mentaleffort",
    "mentaldemand", "physicaldemand", "temporaldemand", "effort",
    "performance", "frustration", "nasa_tlx", "rsme",
}


def _exact_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def swell_interaction_columns(df: pd.DataFrame) -> list[str]:
    """The computer-interaction feature columns of a SWELL-KW table."""
    cols = [c for c in df.columns if str(c).startswith("Sn")]
    if not cols:
        cols = [
            c for c in df.columns
            if any(h in str(c).lower() for h in SWELL_INTERACTION_HINTS)
        ]
    return [c for c in cols if str(c).lower() not in SWELL_SELF_REPORT_COLUMNS]


def load_swell(
    path: Path | None = None,
    frame: pd.DataFrame | None = None,
    feature_set: str = "interaction",
) -> pd.DataFrame:
    """
    Load the SWELL-KW feature file.

    Returns the selected numeric features plus 'participant' (for grouped
    evaluation; it is never used as a feature) and 'stress_label'
    (N -> Low, I -> Medium, T -> High; rest periods dropped).

    feature_set:
        "interaction"  computer-interaction features only (default; the
                       typing-behaviour modality of this project)
        "all_numeric"  every numeric column except identifiers and
                       questionnaire answers
    Pass `frame` instead of `path` to load an in-memory table.
    """
    if frame is None:
        path = Path(path or config.SWELL_PATH)
        df = _read_any(path)
        name = path.name
    else:
        df = frame
        name = "<frame>"
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    label_col = _exact_column(df, SWELL_CONDITION_CANDIDATES) or _find_column(
        df, ["condition"]
    )
    if label_col is None:
        raise ValueError(
            f"Could not find a condition/label column in {name}. "
            f"Columns found: {list(df.columns)[:25]}"
        )
    pp_col = _exact_column(df, SWELL_PARTICIPANT_CANDIDATES)

    labels = df[label_col].map(_normalise_swell_label)
    unmapped = labels.isna().sum()
    if unmapped:
        warnings.warn(
            f"{unmapped} rows had condition values not in SWELL_CONDITION_MAP "
            f"(e.g. rest periods) and were dropped. Unique raw values: "
            f"{df.loc[labels.isna(), label_col].astype(str).unique()[:10]}"
        )

    if feature_set == "interaction":
        feat_cols = swell_interaction_columns(df)
        if not feat_cols:
            hrv = [c for c in df.columns if c.upper() in {"MEAN_RR", "RMSSD", "HR", "SDRR"}]
            hint = (
                " This looks like the heart-rate (HRV) derivative of SWELL, "
                "which has no typing or mouse data."
                if hrv else ""
            )
            raise ValueError(
                f"No computer-interaction columns (Sn*, mouse, key, app ...) "
                f"found in {name}.{hint} Use the SWELL-KW computer-interaction "
                f"feature file, or pass feature_set='all_numeric' to use "
                f"whatever numeric columns exist."
            )
    elif feature_set == "all_numeric":
        drop_like = ["subject", "participant", "id", "session", "datetime",
                     "time", "block", "blok"]
        feat_cols = [
            c for c in df.columns
            if c not in {label_col, pp_col}
            and str(c).lower() not in SWELL_SELF_REPORT_COLUMNS
            and not any(k in str(c).lower() for k in drop_like)
        ]
    else:
        raise ValueError("feature_set must be 'interaction' or 'all_numeric'")

    out = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.notna().any()]
    out["participant"] = (
        df[pp_col].astype(str) if pp_col else "unknown"
    )
    out["stress_label"] = labels
    if pp_col is None:
        warnings.warn(
            "No participant column found: grouped (cross-person) evaluation "
            "is impossible and results will be optimistic."
        )
    return out.dropna(subset=["stress_label"]).reset_index(drop=True)


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
    # One Reddit post can give several rows. Kept so evaluation can hold out
    # whole posts; never a model input (text pipelines select their columns).
    if "post_id" in df.columns:
        out["post_id"] = df["post_id"].astype(str)
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
        # stroke_id ties each press to its own release, so dwell time is exact
        # even though the keys themselves are masked as "$".
        event_frames = []
        stroke_id = np.arange(len(ks))
        if press_col:
            event_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(ks[press_col], errors="coerce"),
                        "key": ks[key_col],
                        "event": "press",
                        "stroke_id": stroke_id,
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
                        "stroke_id": stroke_id,
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
        cols = ["user", "timestamp", "key", "event", "stress_label"]
        if "stroke_id" in merged.columns:
            cols.append("stroke_id")
        frames.append(merged[cols])

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
def _own_files(path: Path | None, pattern: str) -> list[Path]:
    path = Path(path or config.OWN_DIR)
    files = sorted(path.glob(pattern)) if path.is_dir() else [path]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(
            f"No {pattern} found in {path}. Export them from "
            f"collector/typing_collector.html and put them there."
        )
    return files


def _read_own_csvs(files: list[Path]) -> pd.DataFrame:
    # keep_default_na=False: a participant may genuinely type "NA" or "null",
    # and the key column holds values like " " (space) that must survive.
    return pd.concat(
        [pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files],
        ignore_index=True,
    )


def own_data_available(path: Path | None = None) -> bool:
    path = Path(path or config.OWN_DIR)
    return (
        path.is_dir()
        and any(path.glob("sessions_*.csv"))
        and any(path.glob("keystrokes_*.csv"))
    )


def load_own_sessions(path: Path | None = None) -> pd.DataFrame:
    """
    Load sessions_*.csv exported by the collector tool.

    One row per copy task or chat answer, with the typed text ('text'), the
    block's self-reported stress ('stress_label') and the browser-computed
    summary statistics. For modelling prefer `load_own_paired`, which
    recomputes typing features from the raw keystrokes in Python.
    """
    files = _own_files(path, "sessions_*.csv")
    df = _read_own_csvs(files)
    required = {"participant_id", "session_id", "typed_text", "self_report_stress"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Collected-session file is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    df = df.drop_duplicates(subset=["session_id"], keep="last")
    df = df.rename(columns={"typed_text": "text",
                            "self_report_stress": "stress_label"})
    df["text"] = df["text"].fillna("").astype(str)
    df["participant_id"] = df["participant_id"].astype(str)
    df["stress_label"] = df["stress_label"].astype(str).str.strip().str.title()
    df = df[df["stress_label"].isin(config.STRESS_CLASSES)]
    if "task_type" not in df.columns:
        df["task_type"] = "free"
    print(f"  loaded {len(df)} sessions from {len(files)} file(s), "
          f"{df['participant_id'].nunique()} participants")
    return df.reset_index(drop=True)


def _own_events(df: pd.DataFrame) -> pd.DataFrame:
    """Collector keystroke rows -> event format used by typing_features."""
    df = df.copy()
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce")
    df = df.dropna(subset=["t_ms"])
    df["participant_id"] = df["participant_id"].astype(str)
    df["session_id"] = df["session_id"].astype(str)
    # t_ms is relative to the session start -> synthesise an absolute timestamp
    df["timestamp"] = pd.to_datetime(df["t_ms"], unit="ms", origin="unix")
    df["event"] = df["event"].map({"down": "press", "up": "release"})
    df["key"] = df["key"].fillna("").astype(str)
    if "code" in df.columns:
        df["code"] = df["code"].fillna("").astype(str)
    return df.dropna(subset=["event"])


def load_own_keystrokes(path: Path | None = None) -> pd.DataFrame:
    """
    Load keystrokes_*.csv exported by the collector tool and reshape it into
    the event format expected by features.typing_features.features_from_events.
    """
    files = _own_files(path, "keystrokes_*.csv")
    df = _read_own_csvs(files)
    required = {"participant_id", "session_id", "event", "key", "t_ms", "stress_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Collected-keystroke file is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    df = _own_events(df)
    # Each export's timestamps start at zero. A session-specific user key keeps
    # different tasks from being accidentally aggregated into one 1970 window.
    df["user"] = df["participant_id"] + "__" + df["session_id"]
    cols = ["user", "participant_id", "session_id", "timestamp", "key", "event",
            "stress_label"]
    if "code" in df.columns:
        cols.insert(5, "code")
    return df[cols].reset_index(drop=True)


def build_paired_frame(
    sessions: pd.DataFrame,
    keystrokes: pd.DataFrame,
    min_presses: int | None = None,
) -> pd.DataFrame:
    """
    Join collector sessions with typing features computed from their raw
    keystrokes. Accepts the raw CSV shapes (as read from disk).

    Returns one row per session with: participant_id, session_id, condition,
    task_type, prompt_id, text, stress_label and the typing feature columns
    of features.typing_features. Sessions without enough key presses are
    dropped (they have no usable typing measurement).
    """
    from features.typing_features import features_per_session

    min_presses = min_presses or config.MIN_PRESSES_PER_SESSION
    events = _own_events(keystrokes)
    feats = features_per_session(
        events.drop(columns=["stress_label"], errors="ignore"),
        session_col="session_id", min_presses=min_presses,
    )
    if feats.empty:
        raise ValueError("No session had enough key presses to measure.")
    feats = feats.drop(columns=["stress_label"], errors="ignore")

    meta_cols = [c for c in ["participant_id", "session_id", "condition",
                             "task_type", "turn", "prompt_id", "text",
                             "stress_label", "self_report_intensity",
                             "self_report_pressure"] if c in sessions.columns]
    meta = sessions[meta_cols].copy()
    meta["session_id"] = meta["session_id"].astype(str)
    out = meta.merge(feats, on="session_id", how="inner")
    return out.reset_index(drop=True)


def load_own_paired(path: Path | None = None) -> pd.DataFrame:
    """Load both collector exports and return `build_paired_frame` output."""
    sessions = load_own_sessions(path)
    keys = _read_own_csvs(_own_files(path, "keystrokes_*.csv"))
    paired = build_paired_frame(sessions, keys)
    print(f"  {len(paired)} sessions have typing + text "
          f"({len(sessions) - len(paired)} dropped: too few key presses)")
    return paired


# ==========================================================================
# Synthetic stand-ins in the exact on-disk formats of the real sources.
# They exist to exercise every code path before real data exists.
# NEVER report results from these as project results: the signal is planted.
# ==========================================================================
_SYN_NEUTRAL = [
    "i went to the market and then cooked dinner with my roommate",
    "usually i wake up around seven and have tea before class",
    "this weekend i might visit my cousin and watch a movie",
    "i like the park near my house because it is quiet in the evening",
    "i take the bus to college and it takes about twenty minutes",
    "we had rice and dal at home and it was good",
    "this week i worked on a small python project for a course",
    "i play badminton with friends on most saturdays",
]
_SYN_CALM = ["it was relaxing", "i enjoyed it", "nothing much really, pretty calm",
             "it was nice and easy", "felt good overall"]
_SYN_STRESS = ["but i am so behind on everything", "honestly i am exhausted and worried",
               "i have a deadline and i cannot focus", "i feel rushed and stressed",
               "too much pressure right now"]


def load_synthetic_paired(
    n_participants: int = 16, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic (sessions, keystrokes) frames in the collector's v2 CSV format:
    per participant 2 copy tasks + 8 chat answers, rated per block.
    Each person gets their own typing speed, so a model must generalise
    across people - the same difficulty the real data has.
    """
    rng = np.random.default_rng(seed)
    passage = ("The library on the north side of campus stays open until "
               "midnight during term time.")
    session_rows, key_rows = [], []
    levels = {"Low": 0.0, "Medium": 1.0, "High": 2.0}

    for p in range(1, n_participants + 1):
        pid = f"SYN{p:02d}"
        base_flight = rng.normal(190, 40)          # person-specific speed
        base_dwell = rng.normal(95, 15)
        for block, (cond, task) in enumerate(
            [("relaxed", "copy"), ("relaxed", "chat"),
             ("pressured", "copy"), ("pressured", "chat")], start=1,
        ):
            probs = [0.55, 0.35, 0.10] if cond == "relaxed" else [0.10, 0.40, 0.50]
            label = rng.choice(config.STRESS_CLASSES, p=probs)
            s = levels[label]
            turns = [0] if task == "copy" else [1, 2, 3, 4]
            for turn in turns:
                sid = f"{pid}_B{block}" + (f"_T{turn}" if turn else "")
                if task == "copy":
                    text = passage
                else:
                    tail = _SYN_STRESS if rng.random() < 0.25 + 0.3 * s else _SYN_CALM
                    text = f"{rng.choice(_SYN_NEUTRAL)} {rng.choice(tail)}"
                t = 0.0
                n_back = 0
                for ch in text:
                    if rng.random() < 0.02 + 0.03 * s:      # typo + correction
                        for k, code in (("x", "KeyX"), ("Backspace", "Backspace")):
                            t += max(30, rng.normal(base_flight * (1 - 0.12 * s), 45))
                            d = max(20, rng.normal(base_dwell - 6 * s, 12))
                            key_rows += [(pid, sid, cond, task, turn, "down", k, code, t, ""),
                                         (pid, sid, cond, task, turn, "up", k, code, t + d, d)]
                        n_back += 1
                    gap = rng.normal(base_flight * (1 - 0.12 * s), 45 + 15 * s)
                    if rng.random() < 0.01 + 0.02 * s:
                        gap += rng.uniform(2000, 4000)       # hesitation
                    t += max(30, gap)
                    d = max(20, rng.normal(base_dwell - 6 * s, 12))
                    code = "Space" if ch == " " else f"Key{ch.upper()}"
                    key_rows += [(pid, sid, cond, task, turn, "down", ch, code, t, ""),
                                 (pid, sid, cond, task, turn, "up", ch, code, t + d, d)]
                session_rows.append({
                    "participant_id": pid, "session_id": sid, "block": block,
                    "condition": cond, "task_type": task, "turn": turn,
                    "prompt_id": "P1" if task == "copy" else f"Q{turn}",
                    "duration_s": round(t / 1000, 2),
                    "self_report_stress": label,
                    "self_report_intensity": int(1 + 2 * s),
                    "self_report_pressure": int(1 + 2 * s),
                    "typed_text": text,
                })

    sessions = pd.DataFrame(session_rows)
    keys = pd.DataFrame(key_rows, columns=[
        "participant_id", "session_id", "condition", "task_type", "turn",
        "event", "key", "code", "t_ms", "dwell_ms"])
    keys = keys.merge(
        sessions[["session_id", "self_report_stress"]], on="session_id"
    ).rename(columns={"self_report_stress": "stress_label"})
    return sessions, keys


def load_synthetic_paired_frame(n_participants: int = 16, seed: int = 42) -> pd.DataFrame:
    """`load_synthetic_paired` passed through the same path as real data."""
    sessions, keys = load_synthetic_paired(n_participants, seed)
    sessions = sessions.rename(columns={"typed_text": "text",
                                        "self_report_stress": "stress_label"})
    return build_paired_frame(sessions, keys)


def load_synthetic_swell_raw(
    n_participants: int = 25, minutes: int = 40, seed: int = 42,
) -> pd.DataFrame:
    """
    A table shaped like the SWELL-KW per-minute feature file (PP, Condition,
    Sn* interaction features, plus a questionnaire 'Stress' column that the
    loader must exclude). Pass to load_swell(frame=...).
    """
    rng = np.random.default_rng(seed)
    rows = []
    shift = {"N": 0.0, "I": 1.0, "T": 2.0, "R": 0.0}
    for p in range(1, n_participants + 1):
        person = rng.normal(1.0, 0.25)
        for cond in ["N", "I", "T", "R"]:
            for _ in range(minutes if cond != "R" else 5):
                s = shift[cond]
                rows.append({
                    "PP": f"PP{p}", "Condition": cond,
                    "SnKeyStrokes": max(0, rng.normal(110 * person + 12 * s, 30)),
                    "SnChars": max(0, rng.normal(90 * person + 8 * s, 25)),
                    "SnErrorKeys": max(0, rng.normal(6 + 2.0 * s, 3)),
                    "SnSpecialKeys": max(0, rng.normal(9 + 1.0 * s, 4)),
                    "SnMouseAct": max(0, rng.normal(40 * person + 6 * (s == 1), 12)),
                    "SnMouseDistance": max(0, rng.normal(3000 * person, 800)),
                    "SnLeftClicked": max(0, rng.normal(6 + 3 * (s == 1), 3)),
                    "SnAppChange": max(0, rng.normal(0.6 + 1.2 * (s == 1), 0.6)),
                    "SnTabfocusChange": max(0, rng.normal(0.4 + 0.8 * (s == 1), 0.5)),
                    "Stress": rng.normal(2 + 2 * s, 1),      # questionnaire: must be excluded
                })
    return pd.DataFrame(rows)
