"""
Pipeline A - Typing-behaviour stress classification.

Usage:
    python run_typing.py --source swell        # SWELL-KW feature file
    python run_typing.py --source kaggle       # raw Kaggle keystroke logs
    python run_typing.py --source own          # paired collector sessions
    python run_typing.py --source synthetic    # smoke test, no data needed

Trains Logistic Regression, Random Forest and SVM, prints a comparison table,
saves confusion matrices and the Random Forest feature importances.

When the data identifies participants (SWELL-KW, own sessions) the test set
is made of whole held-out participants, and cross-validation is grouped the
same way. Kaggle has only two people, so it keeps the random split; see
run_midterm.py for its cross-user test.
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


def load_data(source: str) -> tuple[pd.DataFrame, str | None]:
    """Returns (frame, name of the participant column or None)."""
    if source == "swell":
        print(f"Loading SWELL-KW from {config.SWELL_PATH} ...")
        return loaders.load_swell(), "participant"

    if source == "kaggle":
        print(f"Loading Kaggle keystroke logs from "
              f"{config.KEYSTROKE_KAGGLE_DIR} ...")
        events = loaders.load_kaggle_keystrokes()
        print(f"  {len(events):,} keystroke events loaded")
        df = typing_features.features_from_events(events)
        print(f"  aggregated into {len(df)} session windows")
        return df, None

    if source == "own":
        print(f"Loading collected sessions from {config.OWN_DIR} ...")
        return loaders.load_own_paired(), "participant_id"

    print("Loading SYNTHETIC typing data (development only) ...")
    return loaders.load_synthetic_typing(), None


def feature_matrix(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.Series]:
    if source == "own":
        # explicit allow-list: everything else in the session file is either
        # the label (self-ratings, condition) or fixed by the timer
        cols = [c for c in config.OWN_TYPING_FEATURES if c in df.columns]
        return df[cols], df["stress_label"].astype(str)
    return typing_features.prepare_feature_frame(df)


def main(source: str, save_models: bool):
    # synthetic smoke tests never overwrite real results
    out_dir = config.OUTPUT_DIR / "synthetic" if source == "synthetic" else config.OUTPUT_DIR
    df, group_col = load_data(source)
    X, y = feature_matrix(df, source)
    groups = df[group_col].astype(str).to_numpy() if group_col else None

    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print("Class distribution:")
    print(y.value_counts().to_string())

    if groups is not None:
        X_tr, X_te, y_tr, y_te = evaluation.grouped_train_test_split(X, y, groups)
        print(f"\nTest set = whole held-out participants "
              f"({len(set(groups))} participants in total)")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y if y.value_counts().min() >= 2 else None,
        )
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
    print(f"Imbalance strategy: {config.IMBALANCE_STRATEGY}")

    results, fitted = [], {}
    for name, est in classifiers.get_estimators().items():
        pipe = classifiers.build_typing_pipeline(est)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        row = evaluation.evaluate_model(y_te, y_pred, name)
        row.update(evaluation.cross_validate(pipe, X, y, groups=groups))
        results.append(row)
        fitted[name] = pipe

        evaluation.print_report(y_te, y_pred, f"Typing / {name}")
        path = evaluation.save_confusion_matrix(
            y_te, y_pred, f"typing_{name}", out_dir=out_dir
        )
        print(f"saved: {path}")

    table = evaluation.results_table(results)
    print("\n================ TYPING PIPELINE RESULTS ================")
    print(table.to_string(index=False))

    out = out_dir / "results_typing.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nsaved: {out}")

    imp = classifiers.feature_importances(
        fitted["RandomForest"], list(X.columns)
    )
    if imp is not None:
        print("\nTop typing features (Random Forest importance):")
        print(imp.to_string(index=False))
        imp.to_csv(out_dir / "feature_importance_typing.csv",
                   index=False)

    if save_models:
        for name, pipe in fitted.items():
            joblib.dump(pipe, out_dir / f"model_typing_{name}.joblib")
        print(f"\nmodels saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", default="synthetic",
        choices=["swell", "kaggle", "own", "synthetic"],
    )
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    main(args.source, args.save_models)
