"""
Typing features for one burst, computed the same way in Python and in the extension
(extension/engine.js). Only key-press times and backspaces are used: SWELL-KW has no key
releases, and it logs modifier keys differently from a browser.
"""

from __future__ import annotations

import numpy as np

from features.keystroke_dynamics import rhythm_features

MAX_BURST_GAP_MS = 2000
MIN_BURST_KEYS = 15
MAX_BURST_KEYS = 400
MODIFIERS = {"shift", "control", "alt", "altgraph", "meta", "capslock", "fn"}   # browser only, dropped live
FEATURES = ["log_speed", "log_median_gap", "log_mean_gap", "gap_cv", "backspace_rate", "rhythm_step_median",
            "rhythm_step_iqr", "rhythm_lag1", "rhythm_drift", "rhythm_entropy", "rhythm_fast_share"]


def burst_features(times_ms, is_backspace) -> dict:
    times = np.asarray(times_ms, dtype=float)
    gaps = np.diff(times)
    gaps = gaps[gaps > 0]
    duration = max(times[-1] - times[0], 1.0)
    out = {"log_speed": float(np.log(len(times) / (duration / 60_000))),
           "log_median_gap": float(np.log(np.median(gaps))) if gaps.size else np.nan,
           "log_mean_gap": float(np.log(gaps.mean())) if gaps.size else np.nan,
           "gap_cv": float(gaps.std() / gaps.mean()) if gaps.size else np.nan,
           "backspace_rate": float(np.mean(is_backspace))}
    out.update(rhythm_features(gaps))
    return out


def split_bursts(times_ms) -> list[tuple[int, int]]:
    """[start, end) index pairs: a pause ends a burst and long bursts are cut, as live."""
    times = np.asarray(times_ms, dtype=float)
    spans, start = [], 0
    for i in range(1, len(times) + 1):
        if i == len(times) or times[i] - times[i - 1] > MAX_BURST_GAP_MS or i - start >= MAX_BURST_KEYS:
            if i - start >= MIN_BURST_KEYS:
                spans.append((start, i))
            start = i
    return spans
