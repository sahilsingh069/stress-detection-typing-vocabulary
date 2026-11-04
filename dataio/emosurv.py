"""
EmoSurv (Maalej and Kallel, 2020): keystrokes from about 80 people, on physical
keyboards, typing a fixed passage and free text in a neutral state and again after
an emotion-eliciting video (anger, happiness, calmness or sadness).

The labels are EMOTIONS, not stress. Anger vs. calm is the closest proxy for stress
(negative and activated vs. positive and settled), and it stays a proxy.

Traps in this dataset, found before any model was run:
  - Neutral is always first, so neutral vs. emotion also compares early with late.
  - Every emotion has its own fixed passage and neutral's is the longest. A model
    can recognise the passage instead of the emotion, so free text is the primary
    data and fixed text is only reported as confounded.
  - `answer` is blank for all neutral rows, and in free text for all anger and calm
    rows, so it reveals the label. It is never read.
  - `TotTime` and raw counts depend on the passage length, so they are not features.
  - Rows without a key-up are auto-repeats (a held backspace) or corrupted timing,
    with values near +-1.58e12. Repeats are not presses; their time is kept.
  - keyCode was stored as a character (comma as a fraction sign, numpad '.' as 'n'),
    so rebuilt text is approximate.

Source: IEEE DataPort, doi:10.21227/eae6-pk42. Non-commercial research use, no
redistribution, so the files must stay out of the repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from features import typing_features

FILES = {"fixed": "Fixed Text Typing Dataset.csv", "free": "Free Text Typing Dataset.csv"}
EMOTIONS = {"N": "neutral", "A": "anger", "H": "happiness", "C": "calmness", "S": "sadness"}
MIN_BLOCK_PRESSES = 20
MAX_VALID_MS = 300_000
_KEYS = {"\\b": "backspace", "\\u0010": "shift", "\\u0011": "ctrl_l", "\\u0014": "caps_lock",
         "\\t": "tab", "\\r": "enter", "\\u0012": "alt_l"}
_ARROWS = {"%", "&", "'", "("}   # keyCodes 37-40 stored as characters
_PUNCT = {"¼": ",", "¾": ".", "ß": "!"}
_USE = ["userId", "emotionIndex", "index", "keyCode", "keyUp", "D1U1", "D1D2", "D1D3"]


def available() -> bool:
    return all((config.DATA_DIR / f).exists() for f in FILES.values())


def _ms(column: pd.Series) -> pd.Series:
    """Decimal commas become points; corrupted timestamps become NaN."""
    v = pd.to_numeric(column.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return v.where((v >= 0) & (v < MAX_VALID_MS))


def load_keys(kind: str) -> pd.DataFrame:
    """One row per logged key, in typing order. `answer` is deliberately not loaded."""
    raw = pd.read_csv(config.DATA_DIR / FILES[kind], sep=";", engine="python", quoting=3,
                      keep_default_na=False, dtype=str)
    raw = raw.rename(columns={"userid": "userId"})[_USE]
    out = pd.DataFrame({
        "participant_id": "E" + raw["userId"],
        "emotion": raw["emotionIndex"],
        "order": pd.to_numeric(raw["index"], errors="coerce"),
        "raw_key": raw["keyCode"],
        "is_press": raw["keyUp"].str.strip().astype(bool),
        "dwell_ms": _ms(raw["D1U1"]),
        "gap_ms": _ms(raw["D1D2"]),
        "tri_ms": _ms(raw["D1D3"]),
    })
    out = out[out["emotion"].isin(EMOTIONS)]
    out["key"] = [_KEYS.get(k, "arrow_key" if k in _ARROWS else k) for k in out["raw_key"]]
    return out.sort_values(["participant_id", "emotion", "order"]).reset_index(drop=True)


def rebuild_text(keys: pd.Series) -> str:
    """The words that remained, applying backspaces. Approximate (see module notes)."""
    chars: list[str] = []
    for k in keys:
        if k == "backspace":
            if chars:
                chars.pop()
        elif k in _PUNCT:
            chars.append(_PUNCT[k])
        elif len(k) == 1 and k.isprintable():
            chars.append(k)
    return "".join(chars)


def block_features(block: pd.DataFrame) -> dict:
    """Timing, hold and correction features. Rates and times only, never counts."""
    # A pseudo-clock from the gaps, so dropping auto-repeats keeps the time they took.
    clock = block["gap_ms"].fillna(0).cumsum().shift(fill_value=0)
    press = block["is_press"].to_numpy()
    stamps = pd.Series(pd.to_datetime(clock[press].to_numpy(), unit="ms"))
    keys = block.loc[press, "key"].reset_index(drop=True)
    out = typing_features.rich_features(stamps, keys)

    gaps = stamps.diff().dt.total_seconds().dropna().to_numpy() * 1000
    typing = gaps[(gaps >= 0) & (gaps < 2000)]
    dwell = block.loc[press, "dwell_ms"].dropna()
    dwell = dwell[dwell < 2000]
    out.update({
        "dwell_mean": dwell.mean(), "dwell_std": dwell.std(), "dwell_median": dwell.median(),
        "gap_mean": typing.mean() if typing.size else np.nan,
        "gap_std": typing.std() if typing.size else np.nan,
        "rhythm_variability": typing.std() / typing.mean() if typing.size and typing.mean() > 0 else np.nan,
        "gap_max_log": np.log1p(gaps.max()) if gaps.size else np.nan,
        "pause_over_10000": float((gaps > 10_000).mean()) if gaps.size else np.nan,
        "trigraph_median": block.loc[press, "tri_ms"].median(),
        "special_key_rate": float(keys.isin(["shift", "ctrl_l", "caps_lock", "alt_l", "tab"]).mean()),
    })
    return out


def load_blocks(kind: str = "free") -> pd.DataFrame:
    """One row per person, emotion and text type: participant_id, emotion, text, features."""
    rows = []
    for (person, emotion), block in load_keys(kind).groupby(["participant_id", "emotion"], sort=True):
        if block["is_press"].sum() < MIN_BLOCK_PRESSES:
            continue
        rows.append({"participant_id": person, "emotion": emotion, "text_type": kind,
                     "text": rebuild_text(block["key"]), **block_features(block)})
    return pd.DataFrame(rows)


def feature_columns(blocks: pd.DataFrame) -> list[str]:
    return [c for c in blocks.columns if c not in ("participant_id", "emotion", "text_type", "text")]


def from_baseline(blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Each emotion block minus that person's own neutral block. A real system could do
    this: type a little when calm, then compare. Neutral rows are removed because they
    would all be zero, which would give the label away.
    """
    cols = feature_columns(blocks)
    base = blocks[blocks["emotion"] == "N"].set_index("participant_id")[cols]
    out = blocks[(blocks["emotion"] != "N") & blocks["participant_id"].isin(base.index)].copy()
    out[cols] = out[cols].to_numpy(dtype=float) - base.loc[out["participant_id"], cols].to_numpy(dtype=float)
    return out.reset_index(drop=True)
