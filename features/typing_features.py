"""
Typing-behaviour feature extraction.

Two entry points:

1. `features_from_events(df)` - build session-level features from RAW keystroke
   event logs (timestamp, key, event). Use this for the Kaggle keystroke
   dataset or for any data you collect yourself.

2. `prepare_feature_frame(df)` - take an ALREADY-featurised table (e.g. the
   SWELL-KW preprocessed feature file) and just clean/select numeric columns.

Features computed (these map directly onto Objective 1 in the synopsis):
  mean/std dwell time      - how long keys are held
  mean/std flight time     - gap between consecutive keystrokes
  typing_speed_cpm         - characters per minute
  backspace_rate           - corrections per keystroke
  long_pause_rate          - fraction of gaps over LONG_PAUSE_THRESHOLD_MS
  rhythm_variability       - coefficient of variation of inter-key intervals
  special_key_rate         - modifier / navigation key usage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

SPECIAL_KEYS = {
    "ctrl_l", "ctrl_r", "shift", "shift_r", "shift_l", "alt_l", "alt_gr",
    "caps_lock", "tab", "num_lock", "arrow_key", "delete", "esc",
}
CORRECTION_KEYS = {"backspace", "delete"}


# --------------------------------------------------------------------------
# 1. raw events -> session features
# --------------------------------------------------------------------------
def features_from_events(
    events: pd.DataFrame,
    window_seconds: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate raw keystroke events into fixed-length session windows.

    Expected columns: timestamp, key, stress_label
    Optional columns: user, event ('press'/'release' - enables dwell time)
    """
    window_seconds = window_seconds or config.SESSION_WINDOW_SECONDS
    df = events.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if "user" not in df.columns:
        df["user"] = "user_0"

    df["window"] = df["timestamp"].dt.floor(f"{window_seconds}s")

    rows = []
    for (user, window), grp in df.groupby(["user", "window"], sort=True):
        if "event" in grp.columns:
            event_names = grp["event"].astype(str).str.lower()
            press_count = event_names.str.contains("press|down", regex=True).sum()
            # Some sources provide only one event type; retain their rows.
            if press_count == 0:
                press_count = len(grp)
        else:
            press_count = len(grp)
        if press_count < 20:       # ignore near-empty keystroke windows
            continue
        rows.append(_window_features(grp, user, window))

    out = pd.DataFrame(rows)
    return out.dropna(subset=["stress_label"]).reset_index(drop=True)


def _window_features(grp: pd.DataFrame, user, window) -> dict:
    grp = grp.sort_values("timestamp")
    # A press/release log contains two records per key. Flight time, typing
    # speed and correction rate must use presses only; including releases
    # would turn key-hold duration into an apparent inter-key interval.
    if "event" in grp.columns:
        event_names = grp["event"].astype(str).str.lower()
        presses = grp[event_names.str.contains("press|down", regex=True)]
    else:
        presses = grp
    if presses.empty:
        presses = grp

    # Timedelta arithmetic is robust to pandas' internal datetime resolution
    # (nanoseconds in older versions, microseconds in newer versions).
    ts = (presses["timestamp"] - presses["timestamp"].iloc[0]).dt.total_seconds() * 1000
    keys = presses["key"].astype(str).str.lower()

    # inter-key intervals (flight time proxy when only press events exist)
    gaps = np.diff(ts.to_numpy())
    gaps = gaps[(gaps >= 0) & (gaps < 60_000)]           # drop idle jumps >60s

    duration_min = max((ts.iloc[-1] - ts.iloc[0]) / 60_000.0, 1e-6)
    n = len(presses)

    feat = {
        "user": user,
        "window_start": window,
        "n_keystrokes": n,
        "typing_speed_cpm": n / duration_min,
        "mean_flight_ms": float(np.mean(gaps)) if gaps.size else np.nan,
        "std_flight_ms": float(np.std(gaps)) if gaps.size else np.nan,
        "median_flight_ms": float(np.median(gaps)) if gaps.size else np.nan,
        "backspace_rate": float(keys.isin(CORRECTION_KEYS).mean()),
        "special_key_rate": float(keys.isin(SPECIAL_KEYS).mean()),
        "long_pause_rate": (
            float(np.mean(gaps > config.LONG_PAUSE_THRESHOLD_MS))
            if gaps.size else np.nan
        ),
        "rhythm_variability": (
            float(np.std(gaps) / np.mean(gaps))
            if gaps.size and np.mean(gaps) > 0 else np.nan
        ),
    }

    # dwell time only computable when press/release pairs are present
    if "event" in grp.columns and grp["event"].nunique() > 1:
        dwell = _dwell_times(grp)
        feat["mean_dwell_ms"] = float(np.mean(dwell)) if dwell else np.nan
        feat["std_dwell_ms"] = float(np.std(dwell)) if dwell else np.nan

    # majority stress label in the window
    feat["stress_label"] = grp["stress_label"].mode().iloc[0]
    return feat


def _dwell_times(grp: pd.DataFrame) -> list[float]:
    """Match press->release per key to get hold durations in ms."""
    pending: dict[str, float] = {}
    dwell: list[float] = []
    for _, row in grp.iterrows():
        key = str(row["key"]).lower()
        t = pd.Timestamp(row["timestamp"])
        ev = str(row.get("event", "")).lower()
        if "press" in ev or "down" in ev:
            pending[key] = t
        elif ("release" in ev or "up" in ev) and key in pending:
            d = (t - pending.pop(key)).total_seconds() * 1000
            if 0 <= d < 2000:
                dwell.append(d)
    return dwell


# --------------------------------------------------------------------------
# 2. already-featurised table -> clean numeric matrix
# --------------------------------------------------------------------------
def prepare_feature_frame(
    df: pd.DataFrame,
    excluded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a featurised DataFrame into (X numeric features, y labels),
    dropping non-numeric and identifier columns and imputing NaNs.
    """
    if "stress_label" not in df.columns:
        raise ValueError("DataFrame must contain a 'stress_label' column")

    y = df["stress_label"].astype(str)
    X = df.drop(columns=["stress_label"] + (excluded_columns or []), errors="ignore")

    drop_like = ["user", "window_start", "subject", "participant",
                 "session", "timestamp", "datetime", "id"]
    X = X.drop(
        columns=[c for c in X.columns
                 if any(k == c.lower() or k in c.lower() for k in drop_like)],
        errors="ignore",
    )

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found.")

    return X, y
