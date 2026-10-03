"""
Interaction features beyond the keyboard: mouse dynamics, idle periods and
application switching.

These come from the Kaggle keystroke dataset's companion files, which sit
alongside keystrokes.tsv but were unused in the first pass. Adding them
measurably improved the typing pipeline (macro F1 0.354 -> 0.438 on 3-class,
0.487 -> 0.540 on binary), because stress shows up in idle patterns and task
switching as well as in typing rhythm.

SWELL-KW logs the same families of signal, so the same feature names carry
over once that dataset is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def _floor_col(df: pd.DataFrame, col: str, window_s: int) -> pd.DataFrame:
    df = df.copy()
    df["_t"] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["_t"])
    df["window_start"] = df["_t"].dt.floor(f"{window_s}s")
    return df


def interaction_features_for_user(
    user_dir: Path, window_seconds: int | None = None
) -> pd.DataFrame:
    """Aggregate one participant's mouse / idle / app logs into windows."""
    window_seconds = window_seconds or config.SESSION_WINDOW_SECONDS
    parts: list[pd.DataFrame] = []

    # --- idle periods -----------------------------------------------------
    p = user_dir / "inactivity.tsv"
    if p.exists():
        d = pd.read_csv(p, sep="\t")
        d.columns = [c.strip() for c in d.columns]
        tcol = next((c for c in d.columns if "Stopped" in c or "Time" in c), None)
        dcol = next((c for c in d.columns if "Duration" in c), None)
        if tcol and dcol:
            d = _floor_col(d, tcol, window_seconds)
            parts.append(
                d.groupby("window_start").agg(
                    idle_count=(dcol, "size"),
                    idle_mean_s=(dcol, "mean"),
                    idle_max_s=(dcol, "max"),
                    idle_total_s=(dcol, "sum"),
                ).reset_index()
            )

    # --- mouse movement speed --------------------------------------------
    p = user_dir / "mouse_mov_speeds.tsv"
    if p.exists():
        d = pd.read_csv(p, sep="\t")
        d.columns = [c.strip() for c in d.columns]
        scol = next((c for c in d.columns if "Speed" in c), None)
        if scol and "Time" in d.columns:
            d = _floor_col(d, "Time", window_seconds)
            parts.append(
                d.groupby("window_start").agg(
                    mouse_speed_mean=(scol, "mean"),
                    mouse_speed_std=(scol, "std"),
                    mouse_speed_max=(scol, "max"),
                    mouse_events=(scol, "size"),
                ).reset_index()
            )

    # --- application switching -------------------------------------------
    p = user_dir / "activewindows.tsv"
    if p.exists():
        d = pd.read_csv(p, sep="\t")
        d.columns = [c.strip() for c in d.columns]
        if "Time" in d.columns and "App_Name" in d.columns:
            d = _floor_col(d, "Time", window_seconds)
            agg = {"app_distinct": ("App_Name", "nunique"),
                   "app_events": ("App_Name", "size")}
            if "BG_App_Cnt" in d.columns:
                agg["bg_app_mean"] = ("BG_App_Cnt", "mean")
            parts.append(d.groupby("window_start").agg(**agg).reset_index())

    if not parts:
        return pd.DataFrame(columns=["window_start"])

    out = parts[0]
    for extra in parts[1:]:
        out = out.merge(extra, on="window_start", how="outer")
    out["user"] = user_dir.name
    return out


def load_interaction_features(
    root: Path | None = None, window_seconds: int | None = None
) -> pd.DataFrame:
    """Build interaction features for every participant folder under `root`."""
    root = Path(root or config.KEYSTROKE_KAGGLE_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Expected extracted data at: {root}")

    frames = [
        interaction_features_for_user(d, window_seconds)
        for d in sorted(p for p in root.iterdir() if p.is_dir())
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["user", "window_start"])
    return pd.concat(frames, ignore_index=True)


def merge_with_typing(
    typing_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join interaction features onto keystroke session windows.
    Windows with no mouse/app activity get 0 (genuinely no activity).
    """
    if interaction_df.empty:
        return typing_df

    t = typing_df.copy()
    t["window_start"] = pd.to_datetime(t["window_start"])
    i = interaction_df.copy()
    i["window_start"] = pd.to_datetime(i["window_start"])

    merged = t.merge(i, on=["user", "window_start"], how="left")
    new_cols = [c for c in i.columns if c not in ("user", "window_start")]
    merged[new_cols] = merged[new_cols].fillna(0.0)
    return merged
