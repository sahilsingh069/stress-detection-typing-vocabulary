"""
Pipeline B - Text / vocabulary stress classification.

Usage:
    python run_text.py --source dreaddit      # Dreaddit CSVs
    python run_text.py --source own           # paired collector sessions
    python run_text.py --source synthetic     # smoke test, no data needed

Builds hand-crafted linguistic features (sentiment, negative-word ratio,
lexical diversity, ...) plus TF-IDF, then trains and compares the same three
classifiers used in the typing pipeline.
"""

from __future__ import annotations

import argparse
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from dataio import loaders
from features import text_features
from modeling import classifiers, evaluation

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(source: str) -> pd.DataFrame:
    if source == "dreaddit":
        print(f"Loading Dreaddit from {config.DREADDIT_TRAIN_PATH} ...")
        return loaders.load_dreaddit()
    if source == "own":
        print(f"Loading collected sessions from {config.DATA_DIR / 'own'} ...")
        return loaders.load_own_sessions()
    print("Loading SYNTHETIC text data (development only) ...")
    return loaders.load_synthetic_text()


def main(source: str, save_models: bool):
    df = load_data(source)
    print(f"Sentiment backend: {text_features.sentiment_backend()}")

    ling = text_features.build_text_features(df["text"])
    X = pd.concat([df[["text"]].reset_index(drop=True), ling], axis=1)
    y = df["stress_label"].astype(str)
    ling_cols = list(ling.columns)

    print(f"\nDocuments: {len(X)}  linguistic features: {len(ling_cols)}")
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
        pipe = classifiers.build_text_pipeline(est, ling_cols)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        row = evaluation.evaluate_model(y_te, y_pred, name)
        row.update(evaluation.cross_validate(pipe, X, y))
        results.append(row)
        fitted[name] = pipe

        evaluation.print_report(y_te, y_pred, f"Text / {name}")
        path = evaluation.save_confusion_matrix(y_te, y_pred, f"text_{name}")
        print(f"saved: {path}")

    table = evaluation.results_table(results)
    print("\n================ TEXT PIPELINE RESULTS ================")
    print(table.to_string(index=False))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / "results_text.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved: {out}")

    # linguistic-only importances (interpretable subset, excludes TF-IDF)
    ling_only_est = classifiers.get_estimators()["RandomForest"]
    ling_pipe = classifiers.build_typing_pipeline(ling_only_est)
    ling_pipe.fit(X_tr[ling_cols], y_tr)
    imp = classifiers.feature_importances(ling_pipe, ling_cols)
    if imp is not None:
        print("\nTop linguistic features (Random Forest importance):")
        print(imp.to_string(index=False))
        imp.to_csv(config.OUTPUT_DIR / "feature_importance_text.csv",
                   index=False)

    if save_models:
        for name, pipe in fitted.items():
            joblib.dump(pipe, config.OUTPUT_DIR / f"model_text_{name}.joblib")
        print(f"\nmodels saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", default="synthetic", choices=["dreaddit", "own", "synthetic"]
    )
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    main(args.source, args.save_models)
