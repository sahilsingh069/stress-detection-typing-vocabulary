"""
Typing-behaviour feature extraction.

Three entry points:

1. `features_from_events(df)` - build fixed-length window features from RAW
   keystroke event logs (timestamp, key, event). Use this for continuous logs
   such as the Kaggle keystroke dataset.

2. `features_per_session(df)` - the same features, but one row per already
   delimited session (one copy task or one chat answer from the collector, or
   one text typed into the demo app). No windowing.

3. `prepare_feature_frame(df)` - take an ALREADY-featurised table (e.g. the
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
    # Kaggle / pynput-style names
    "ctrl_l", "ctrl_r", "shift", "shift_r", "shift_l", "alt_l", "alt_gr",
    "caps_lock", "tab", "num_lock", "arrow_key", "delete", "esc",
    # browser KeyboardEvent.key names (collector and demo app), lowercased
    "control", "alt", "altgraph", "meta", "capslock", "escape", "numlock",
    "arrowleft", "arrowright", "arrowup", "arrowdown", "home", "end",
    "pageup", "pagedown",
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


def features_per_session(
    events: pd.DataFrame,
    session_col: str = "session_id",
    min_presses: int = 10,
) -> pd.DataFrame:
    """
    One feature row per session. Expected columns: session_col, timestamp,
    key, event; optional: code (physical key, preferred for dwell matching),
    stress_label.

    Sessions with fewer than `min_presses` key presses are dropped: the
    timing statistics of a handful of keys are noise.
    """
    df = events.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "stress_label" not in df.columns:
        df["stress_label"] = np.nan

    rows = []
    for sid, grp in df.groupby(session_col, sort=True):
        presses = _press_mask(grp).sum()
        if presses < min_presses:
            continue
        feat = _window_features(grp, user=sid, window=grp["timestamp"].min())
        feat[session_col] = feat.pop("user")
        feat.pop("window_start")
        rows.append(feat)
    return pd.DataFrame(rows)


def _press_mask(grp: pd.DataFrame) -> pd.Series:
    if "event" not in grp.columns:
        return pd.Series(True, index=grp.index)
    names = grp["event"].astype(str).str.lower()
    mask = names.str.contains("press|down", regex=True)
    # Some sources provide only one event type; treat every row as a press.
    return mask if mask.any() else pd.Series(True, index=grp.index)


def _window_features(grp: pd.DataFrame, user, window) -> dict:
    grp = grp.sort_values("timestamp")
    # A press/release log contains two records per key. Flight time, typing
    # speed and correction rate must use presses only; including releases
    # would turn key-hold duration into an apparent inter-key interval.
    presses = grp[_press_mask(grp)]

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

    # majority stress label in the window (absent when predicting live)
    labels = grp["stress_label"].dropna() if "stress_label" in grp else []
    feat["stress_label"] = labels.mode().iloc[0] if len(labels) else np.nan
    return feat


def _dwell_times(grp: pd.DataFrame) -> list[float]:
    """
    Match press->release per key to get hold durations in ms.

    Pairing, most exact first:
      stroke_id  the source recorded press and release of one keystroke
                 together (Kaggle): exact, even with masked keys
      code       physical key (collector, demo): the character `key` can
                 change between press and release (Shift+1 is pressed as "1"
                 and may be released as "!")
      key        fallback
    """
    if "stroke_id" in grp.columns and grp["stroke_id"].notna().all():
        ev = grp["event"].astype(str).str.lower()
        t = pd.to_datetime(grp["timestamp"])
        press = t[ev.str.contains("press|down")].groupby(grp["stroke_id"]).min()
        release = t[ev.str.contains("release|up")].groupby(grp["stroke_id"]).max()
        d = (release - press).dropna().dt.total_seconds() * 1000
        return d[(d >= 0) & (d < 2000)].tolist()

    ident = grp["code"] if "code" in grp.columns else grp["key"]
    pending: dict[str, pd.Timestamp] = {}
    dwell: list[float] = []
    for k, t, ev in zip(
        ident.astype(str).str.lower(),
        pd.to_datetime(grp["timestamp"]),
        grp["event"].astype(str).str.lower(),
    ):
        if "press" in ev or "down" in ev:
            pending.setdefault(k, t)       # ignore auto-repeat presses
        elif ("release" in ev or "up" in ev) and k in pending:
            d = (t - pending.pop(k)).total_seconds() * 1000
            if 0 <= d < 2000:
                dwell.append(d)
    return dwell


# --------------------------------------------------------------------------
# richer timing features (press times only, so they also work without dwell)
# --------------------------------------------------------------------------
def rich_features(timestamps: pd.Series, keys: pd.Series) -> dict:
    """
    Distribution of key gaps rather than only its average, plus bursts, pauses
    before words and corrections. Gaps come from one session in time order.
    """
    t = (pd.to_datetime(timestamps) - pd.to_datetime(timestamps).iloc[0]).dt.total_seconds().to_numpy() * 1000
    k = keys.astype(str).str.lower().to_numpy()
    gap = np.diff(t)
    keep = (gap >= 0) & (gap < 60_000)
    gaps = gap[keep]
    before = k[1:][keep]             # the key each gap leads to
    after = k[:-1][keep]             # the key each gap follows
    typing = gaps[gaps < 2000]
    out: dict = {}
    if typing.size:
        for q in (10, 25, 50, 75, 90):
            out[f"gap_p{q}"] = float(np.percentile(typing, q))
        out["gap_iqr"] = out["gap_p75"] - out["gap_p25"]
    for ms in (500, 1000, 2000, 5000):
        out[f"pause_over_{ms}"] = float((gaps > ms).mean()) if gaps.size else np.nan
    pauses = np.flatnonzero(gaps > 2000)
    runs = np.diff(np.concatenate(([-1], pauses, [gaps.size]))) if gaps.size else np.array([])
    out["burst_keys_mean"] = float(runs.mean()) if runs.size else np.nan
    active_min = typing.sum() / 60_000 if typing.size else np.nan
    out["active_speed_cpm"] = float(typing.size / active_min) if typing.size and active_min > 0 else np.nan
    letter = np.array([len(a) == 1 and a.isalpha() and len(b) == 1 and b.isalpha()
                       for a, b in zip(after, before)], dtype=bool)
    within = gaps[letter & (gaps < 2000)]
    out["within_word_gap"] = float(np.median(within)) if within.size else np.nan
    word_start = gaps[(after == " ") & (gaps < 5000)]
    out["word_start_gap"] = float(np.median(word_start)) if word_start.size else np.nan
    to_back = gaps[(before == "backspace") & (after != "backspace") & (gaps < 5000)]
    out["before_correction_gap"] = float(np.median(to_back)) if to_back.size else np.nan
    out["backspace_rate"] = float(np.mean(k == "backspace"))
    starts = np.sum((k[1:] == "backspace") & (k[:-1] != "backspace"))
    out["correction_run_len"] = float(np.sum(k == "backspace") / starts) if starts else 0.0
    out["delete_rate"] = float(np.mean(k == "delete"))
    out["navigation_rate"] = float(np.mean(k == "arrow_key"))
    out["shortcut_rate"] = float(np.mean(k == "ctrl_l"))
    return out


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
