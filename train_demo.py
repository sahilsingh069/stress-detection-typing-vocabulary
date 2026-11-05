"""
Train the models served by the demo web app (app/server.py).

    python train_demo.py                    # best available real data
    python train_demo.py --source synthetic # try the app before collecting data

Model sources, in order of preference:

    typing  own collected sessions  ->  SWELL-KW bursts (25 people, conditions as
            labels)  ->  Kaggle keystroke windows (2 people). Only own data is
            validated for live input.
    text    own chat answers (when there are enough)  ->  Dreaddit
    early   own collected sessions only (needs paired data)

Everything is written to outputs/demo/ with a meta.json that records where
each model came from, so the app can show it next to every prediction.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import pandas as pd

import config
from dataio import loaders, swell_raw
from features import burst_features, text_features, typing_features
from modeling import classifiers

warnings.filterwarnings("ignore")

MIN_OWN_TEXT_ANSWERS = 150   # below this, Dreaddit's 3.5k posts train a better text model


def _ling(texts: pd.Series, drop: list[str]) -> pd.DataFrame:
    return text_features.build_text_features(texts).drop(columns=drop, errors="ignore")


def train_text(texts: pd.Series, y: pd.Series, drop: list[str]):
    ling = _ling(texts, drop)
    X = pd.concat([texts.reset_index(drop=True).rename("text"), ling], axis=1)
    pipe = classifiers.build_text_pipeline(
        classifiers.get_estimators()["LogisticRegression"], list(ling.columns))
    pipe.fit(X, y.reset_index(drop=True).astype(str))
    return pipe, list(ling.columns)


def train_typing(X: pd.DataFrame, y: pd.Series):
    pipe = classifiers.build_typing_pipeline(classifiers.get_estimators()["RandomForest"])
    pipe.fit(X, y.astype(str))
    return pipe


def train_early(df: pd.DataFrame, typing_cols: list[str]):
    ling = _ling(df["text"], config.PAIRED_EXCLUDED_TEXT_FEATURES)
    X = pd.concat([df[["text"]].reset_index(drop=True), ling,
                   df[typing_cols].reset_index(drop=True)], axis=1)
    pipe = classifiers.build_early_fusion_pipeline(
        classifiers.get_estimators()["LogisticRegression"], list(ling.columns), typing_cols)
    pipe.fit(X, df["stress_label"].astype(str).reset_index(drop=True))
    return pipe, list(ling.columns)


def main(source: str = "auto", weight: float = 0.25) -> dict:
    out = config.DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fusion_weight_typing": weight, "models": {}}

    paired = None
    if source == "synthetic":
        print("Using SYNTHETIC paired sessions - predictions will be meaningless.")
        paired = loaders.load_synthetic_paired_frame()
    elif loaders.own_data_available():
        paired = loaders.load_own_paired()

    typing_cols = list(config.OWN_TYPING_FEATURES)

    # ---- typing ----------------------------------------------------------
    if paired is not None:
        tsrc = "synthetic" if source == "synthetic" else "own"
        model = train_typing(paired[typing_cols], paired["stress_label"])
        n = len(paired)
    elif swell_raw.ULOG_DIR.exists() and any(swell_raw.ULOG_DIR.glob("*.xml")):
        print("No own data: typing model uses SWELL-KW bursts.")
        import run_burst_model
        win = run_burst_model.swell_bursts()
        tsrc, n = "swell", len(win)
        typing_cols = list(burst_features.FEATURES)
        model = run_burst_model.model().fit(win[typing_cols], win["label"])
    else:
        print("No own data: typing model falls back to the Kaggle keystroke set.")
        win = typing_features.features_from_events(loaders.load_kaggle_keystrokes())
        tsrc, n = "kaggle", len(win)
        typing_cols = [c for c in typing_cols if c in win.columns]
        model = train_typing(win[typing_cols], win["stress_label"])
    joblib.dump(model, out / "typing.joblib")
    meta["models"]["typing"] = {
        "source": tsrc, "n_train": int(n), "features": typing_cols,
        "classes": [str(c) for c in model.classes_],
        "validated": tsrc == "own",
    }
    print(f"typing model: {tsrc}, {n} samples")

    # ---- text ------------------------------------------------------------
    own_text = None
    if paired is not None:
        own_text = paired[paired["task_type"].isin(config.OWN_TEXT_TASKS)
                          & paired["text"].str.strip().astype(bool)]
    if own_text is not None and (source == "synthetic" or len(own_text) >= MIN_OWN_TEXT_ANSWERS):
        xsrc = "synthetic" if source == "synthetic" else "own"
        model, ling_cols = train_text(own_text["text"], own_text["stress_label"],
                                      config.PAIRED_EXCLUDED_TEXT_FEATURES)
        n = len(own_text)
    else:
        dd = loaders.load_dreaddit()
        xsrc, n = "dreaddit", len(dd)
        model, ling_cols = train_text(dd["text"], dd["stress_label"], [])
    joblib.dump(model, out / "text.joblib")
    meta["models"]["text"] = {
        "source": xsrc, "n_train": int(n), "linguistic_features": ling_cols,
        "classes": [str(c) for c in model.classes_],
        "validated": xsrc in ("own", "dreaddit"),
    }
    print(f"text model:   {xsrc}, {n} samples")

    # ---- early fusion ----------------------------------------------------
    early_path = out / "early.joblib"
    if own_text is not None and len(own_text):
        esrc = "synthetic" if source == "synthetic" else "own"
        model, ling_cols = train_early(own_text, typing_cols)
        joblib.dump(model, early_path)
        meta["models"]["early"] = {
            "source": esrc, "n_train": int(len(own_text)),
            "features": typing_cols, "linguistic_features": ling_cols,
            "classes": [str(c) for c in model.classes_],
            "validated": esrc == "own",
        }
        print(f"early fusion: {esrc}, {len(own_text)} samples")
    elif early_path.exists():
        early_path.unlink()

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndemo models written to {out}/  ->  now run: python app/server.py")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic"])
    ap.add_argument("--weight", type=float, default=0.25,
                    help="typing weight for late fusion shown in the app")
    args = ap.parse_args()
    main(args.source, args.weight)
