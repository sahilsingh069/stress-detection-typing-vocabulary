"""
Pipeline A - Typing-behaviour stress classification.

Usage:
    python run_typing.py --source swell        # SWELL-KW feature file
    python run_typing.py --source kaggle       # raw Kaggle keystroke logs
    python run_typing.py --source own          # paired collector sessions
    python run_typing.py --source synthetic    # smoke test, no data needed

Trains Logistic Regression, Random Forest and SVM, prints a comparison table,
saves confusion matrices and the Random Forest feature importances.
"""

from __future__ import annotations

import argparse
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from dataio import loaders
from features import typing_features
from modeling import classifiers, evaluation

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(source: str) -> pd.DataFrame:
    if source == "swell":
        print(f"Loading SWELL-KW from {config.SWELL_PATH} ...")
        return loaders.load_swell()

    if source == "kaggle":
        print(f"Loading Kaggle keystroke logs from "
              f"{config.KEYSTROKE_KAGGLE_DIR} ...")
        events = loaders.load_kaggle_keystrokes()
        print(f"  {len(events):,} keystroke events loaded")
        df = typing_features.features_from_events(events)
        print(f"  aggregated into {len(df)} session windows")
        return df

    if source == "own":
        print(f"Loading collected sessions from {config.DATA_DIR / 'own'} ...")
        return loaders.load_own_sessions()

    print("Loading SYNTHETIC typing data (development only) ...")
    return loaders.load_synthetic_typing()


OWN_NON_BEHAVIOURAL_COLUMNS = [
    "text", "block", "condition", "task_type", "time_limit_s", "timed_out",
    "self_report_intensity", "self_report_pressure", "age_band",
    "typing_familiarity", "started_at", "ended_at", "char_count", "word_count",
]


def main(source: str, save_models: bool):
    df = load_data(source)
    X, y = typing_features.prepare_feature_frame(
        df,
        excluded_columns=OWN_NON_BEHAVIOURAL_COLUMNS if source == "own" else None,
    )

    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print("Class distribution:")
    print(y.value_counts().to_string())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    print(f"\nTrain: {len(X_tr)}  Test: {len(X_te)}")
    print(f"Imbalance strategy: {config.IMBALANCE_STRATEGY}")

    results, fitted = [], {}
    for name, est in classifiers.get_estimators().items():
        pipe = classifiers.build_typing_pipeline(est)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        row = evaluation.evaluate_model(y_te, y_pred, name)
        row.update(evaluation.cross_validate(pipe, X, y))
        results.append(row)
        fitted[name] = pipe

        evaluation.print_report(y_te, y_pred, f"Typing / {name}")
        path = evaluation.save_confusion_matrix(
            y_te, y_pred, f"typing_{name}"
        )
        print(f"saved: {path}")

    table = evaluation.results_table(results)
    print("\n================ TYPING PIPELINE RESULTS ================")
    print(table.to_string(index=False))

    out = config.OUTPUT_DIR / "results_typing.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nsaved: {out}")

    imp = classifiers.feature_importances(
        fitted["RandomForest"], list(X.columns)
    )
    if imp is not None:
        print("\nTop typing features (Random Forest importance):")
        print(imp.to_string(index=False))
        imp.to_csv(config.OUTPUT_DIR / "feature_importance_typing.csv",
                   index=False)

    if save_models:
        for name, pipe in fitted.items():
            joblib.dump(pipe, config.OUTPUT_DIR / f"model_typing_{name}.joblib")
        print(f"\nmodels saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", default="synthetic",
        choices=["swell", "kaggle", "own", "synthetic"],
    )
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    main(args.source, args.save_models)
