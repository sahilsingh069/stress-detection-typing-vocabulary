"""
Timing features that depend less on what was typed.

A gap between 't' and 'h' is naturally shorter than one between 'q' and 'z', so the mix
of letters changes raw speed. These features compare each gap with what is usual for that
letter pair, and describe how gaps change from one key to the next. No labels are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIR_COUNT = 5
MAX_GAP_MS = 2000


def _pairs(keys: np.ndarray, gaps: np.ndarray) -> pd.DataFrame:
    a, b = keys[:-1], keys[1:]
    letter = np.array([len(x) == 1 and x.isalpha() and len(y) == 1 and y.isalpha() for x, y in zip(a, b)], dtype=bool)
    ok = letter & (gaps > 0) & (gaps < MAX_GAP_MS)
    return pd.DataFrame({"pair": pd.Series(a[ok]).astype(str) + pd.Series(b[ok]).astype(str), "gap": gaps[ok]})


def pair_table(sequences) -> pd.Series:
    """Median gap per letter pair, from any number of (keys, gaps_ms) sequences."""
    rows = [_pairs(np.asarray(k, dtype=str), np.asarray(g, dtype=float)) for k, g in sequences]
    all_rows = pd.concat(rows, ignore_index=True)
    grouped = all_rows.groupby("pair")["gap"]
    return grouped.median()[grouped.size() >= MIN_PAIR_COUNT]


def pair_features(keys, gaps_ms, table: pd.Series, prefix: str) -> dict:
    """How much slower than usual this stretch is, for the letter pairs it contains."""
    rows = _pairs(np.asarray(keys, dtype=str), np.asarray(gaps_ms, dtype=float))
    usual = rows["pair"].map(table)
    ratio = np.log(rows["gap"][usual.notna()] / usual[usual.notna()]).to_numpy()
    if ratio.size < 5:
        return {f"{prefix}_{n}": np.nan for n in ("median", "mean", "std", "p90")}
    return {f"{prefix}_median": float(np.median(ratio)), f"{prefix}_mean": float(ratio.mean()),
            f"{prefix}_std": float(ratio.std()), f"{prefix}_p90": float(np.percentile(ratio, 90))}


def rhythm_features(gaps_ms) -> dict:
    """How gaps change from key to key: steadiness, memory, drift and spread."""
    gaps = np.asarray(gaps_ms, dtype=float)
    gaps = gaps[(gaps > 0) & (gaps < MAX_GAP_MS)]
    names = ("step_median", "step_iqr", "lag1", "drift", "entropy", "fast_share")
    if gaps.size < 10:
        return {f"rhythm_{n}": np.nan for n in names}
    log = np.log(gaps)
    step = np.abs(np.diff(log))
    centred = log - log.mean()
    denom = float((centred ** 2).sum())
    counts, _ = np.histogram(log, bins=10)
    share = counts[counts > 0] / counts.sum()
    return {"rhythm_step_median": float(np.median(step)),
            "rhythm_step_iqr": float(np.percentile(step, 75) - np.percentile(step, 25)),
            "rhythm_lag1": float((centred[:-1] * centred[1:]).sum() / denom) if denom > 1e-12 else 0.0,
            "rhythm_drift": float(np.polyfit(np.arange(log.size), log, 1)[0] * log.size),
            "rhythm_entropy": float(-(share * np.log(share)).sum()),
            "rhythm_fast_share": float((gaps < 120).mean())}
