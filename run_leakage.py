"""
Why published scores are far above ours: the same data, scored the way many papers
score it and the way this project does. Labelled numbers from the leaky rows are
NOT results of this project; they exist to measure the gap.

    leaky, overlapping   random 10-fold over overlapping windows
    leaky                random 10-fold over separate windows (a person is on both sides)
    honest               separate windows, whole people held out

The models are Random Forests with no tuning, as in the papers compared with. Key
composition features let a model read which passage was typed.

    python run_leakage.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

import config
from dataio import emosurv, swell_raw
from features import typing_features

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "leakage"
SEED = config.RANDOM_STATE
LETTERS = list("abcdefghijklmnopqrstuvwxyz ")


def windows(frame: pd.DataFrame, by: list[str], size: int, step: int, make) -> pd.DataFrame:
    """Sliding windows over key presses, within one person's block."""
    rows = []
    for names, block in frame.groupby(by, sort=True):
        for start in range(0, len(block) - size + 1, step):
            feats = make(block.iloc[start:start + size])
            if feats is not None:
                rows.append({"person": names[0], "label": names[1], **feats})
    return pd.DataFrame(rows)


def score(table: pd.DataFrame, columns: list[str], people_held_out: bool) -> dict:
    X = table[columns].fillna(table[columns].median()).fillna(0)
    y, g = table["label"].to_numpy(), table["person"].to_numpy()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED) if people_held_out
          else StratifiedKFold(10, shuffle=True, random_state=SEED))
    pred = cross_val_predict(rf, X, y, groups=g if people_held_out else None, cv=cv)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "majority_acc": pd.Series(y).value_counts(normalize=True).max(),
            "n_windows": len(y), "n_people": len(set(g))}


def emosurv_features(window: pd.DataFrame) -> dict | None:
    if window["is_press"].sum() < 10:      # nearly all auto-repeats from a held key
        return None
    out = emosurv.block_features(window)
    for ch in LETTERS:
        out[f"share_{ch}"] = float((window["key"] == ch).mean())
    return out


def swell_features(window: pd.DataFrame) -> dict:
    return typing_features.rich_features(window["timestamp"], window["key"])


def run() -> pd.DataFrame:
    rows = []

    print("SWELL-KW raw logs, neutral vs time pressure ...", flush=True)
    keys = swell_raw.windowed_word_keys().sort_values(["participant_id", "block", "timestamp"])
    keys = keys[keys["condition"].isin(["N", "T"])]
    make = lambda step: windows(keys, ["participant_id", "condition"], 100, step, swell_features)
    apart, overlap = make(100), make(20)
    cols = [c for c in apart.columns if c not in ("person", "label")]
    for name, table, grouped in [("leaky, overlapping", overlap, False), ("leaky", apart, False),
                                 ("honest", apart, True)]:
        rows.append({"data": "SWELL-KW, neutral vs time pressure", "features": "timing",
                     "evaluation": name, **score(table, cols, grouped)})
        print(rows[-1], flush=True)

    print("EmoSurv, five states ...", flush=True)
    for kind in ("fixed", "free"):
        k = emosurv.load_keys(kind)
        apart = windows(k, ["participant_id", "emotion"], 30, 30, emosurv_features)
        timing = [c for c in apart.columns if c not in ("person", "label") and not c.startswith("share_")]
        everything = timing + [f"share_{c}" for c in LETTERS]
        if kind == "fixed":
            overlap = windows(k, ["participant_id", "emotion"], 30, 5, emosurv_features)
            rows.append({"data": "EmoSurv fixed text, five states", "features": "timing + keys",
                         "evaluation": "leaky, overlapping", **score(overlap, everything, False)})
            print(rows[-1], flush=True)
        for feats, cols_ in [("timing + keys", everything), ("timing", timing)]:
            for name, grouped in [("leaky", False), ("honest", True)]:
                rows.append({"data": f"EmoSurv {kind} text, five states", "features": feats,
                             "evaluation": name, **score(apart, cols_, grouped)})
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    table = run().round(3)
    table.to_csv(OUT / "leakage_gap.csv", index=False)
    (OUT / "report_leakage.md").write_text(
        "# The same data, scored two ways\n\nLeaky rows are not results of this project.\n\n"
        + table.to_markdown(index=False) + "\n")
    print(table.to_string(index=False))
