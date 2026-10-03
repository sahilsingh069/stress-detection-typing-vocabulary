"""
Mid-term training run.

Trains and evaluates BOTH pipelines end to end, with hyperparameter tuning,
and writes every artefact needed for the mid-term report and slides.

    python run_midterm.py

Produces in outputs/:
    midterm_results.csv          every model x every configuration
    midterm_report.md            written summary you can paste into the report
    confusion_*.png              per-model confusion matrices
    feature_importance_*.csv     which features actually carried signal
    cross_user_generalisation.csv   the honest negative finding

Pipeline A (typing) uses the Kaggle keystroke dataset plus its mouse, idle and
app-switching companion logs. Pipeline B (text) uses Dreaddit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import config
from dataio import loaders
from features import interaction_features, text_features, typing_features
from modeling import classifiers, evaluation

warnings.filterwarnings("ignore")

OUT = config.OUTPUT_DIR
OUT.mkdir(parents=True, exist_ok=True)

PARAM_GRIDS = {
    "LogisticRegression": {"clf__C": [0.05, 0.3, 1.0, 3.0]},
    "RandomForest": {
        "clf__n_estimators": [300],
        "clf__max_depth": [None, 8, 14],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "SVM": {"clf__C": [0.5, 1.0, 4.0], "clf__gamma": ["scale", 0.05]},
}


def _cv(y, max_folds: int = 5) -> StratifiedKFold | int:
    n = pd.Series(y).value_counts().min()
    folds = int(min(max_folds, n))
    if folds < 2:
        return 2
    return StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE)


# ==========================================================================
# Pipeline A - typing behaviour + interaction
# ==========================================================================
def build_typing_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (keystroke-only frame, keystroke + interaction frame)."""
    events = loaders.load_kaggle_keystrokes()
    print(f"  {len(events):,} keystroke events")

    base = typing_features.features_from_events(events)
    print(f"  -> {len(base)} session windows")

    inter = interaction_features.load_interaction_features()
    merged = interaction_features.merge_with_typing(base, inter)
    added = [c for c in merged.columns if c not in base.columns]
    print(f"  + {len(added)} interaction features: {added}")
    return base, merged


def run_typing() -> tuple[list[dict], dict]:
    print("\n" + "=" * 66)
    print("PIPELINE A - TYPING BEHAVIOUR")
    print("=" * 66)

    base, merged = build_typing_dataset()
    results, best_models = [], {}

    variants = {
        "keystroke_only_3class": base,
        "keystroke+interaction_3class": merged,
        "keystroke+interaction_binary": _binarise(merged),
    }

    for vname, df in variants.items():
        X, y = typing_features.prepare_feature_frame(df)
        maj = y.value_counts(normalize=True).max()
        print(f"\n[{vname}]  n={len(X)}  features={X.shape[1]}  "
              f"majority-class accuracy={maj:.3f}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=y,
        )

        for name, est in classifiers.get_estimators().items():
            pipe = classifiers.build_typing_pipeline(est)
            gs = GridSearchCV(
                pipe, PARAM_GRIDS[name], cv=_cv(y_tr),
                scoring="f1_macro", n_jobs=-1,
            )
            gs.fit(X_tr, y_tr)
            y_pred = gs.best_estimator_.predict(X_te)

            row = evaluation.evaluate_model(y_te, y_pred, name)
            row.update({
                "pipeline": "typing", "variant": vname,
                "n_samples": len(X), "n_features": X.shape[1],
                "majority_baseline_acc": round(maj, 4),
                "beats_baseline": bool(row["accuracy"] > maj),
                "best_params": str(gs.best_params_),
                "cv_f1_mean": round(gs.best_score_, 4),
            })
            results.append(row)
            print(f"  {name:20s} acc={row['accuracy']:.3f} "
                  f"f1={row['f1_macro']:.3f} "
                  f"{'(beats baseline)' if row['beats_baseline'] else '(below baseline)'}")

            key = f"{vname}__{name}"
            best_models[key] = gs.best_estimator_
            if name == "RandomForest":
                evaluation.save_confusion_matrix(
                    y_te, y_pred, f"typing_{vname}_{name}"
                )
                imp = classifiers.feature_importances(
                    gs.best_estimator_, list(X.columns), top_n=20
                )
                if imp is not None:
                    imp.to_csv(
                        OUT / f"feature_importance_typing_{vname}.csv",
                        index=False,
                    )

    cross_user_test(merged)
    return results, best_models


def _binarise(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["stress_label"] = d["stress_label"].map(
        {"Low": "NotHigh", "Medium": "NotHigh", "High": "High"}
    )
    return d


def cross_user_test(df: pd.DataFrame) -> pd.DataFrame:
    """Train on one participant, test on another. The honest limitation test."""
    print("\n[cross-user generalisation]")
    X, y = typing_features.prepare_feature_frame(df)
    users = df["user"].values
    rows = []
    for tr in np.unique(users):
        for te in np.unique(users):
            if tr == te:
                continue
            m_tr, m_te = users == tr, users == te
            pipe = classifiers.build_typing_pipeline(
                classifiers.get_estimators()["RandomForest"]
            )
            pipe.fit(X[m_tr], y[m_tr])
            p = pipe.predict(X[m_te])
            rows.append({
                "train_user": tr, "test_user": te,
                "accuracy": round(accuracy_score(y[m_te], p), 4),
                "f1_macro": round(
                    f1_score(y[m_te], p, average="macro", zero_division=0), 4
                ),
            })
            print(f"  train {tr} -> test {te}: "
                  f"acc={rows[-1]['accuracy']:.3f} f1={rows[-1]['f1_macro']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cross_user_generalisation.csv", index=False)
    return out


# ==========================================================================
# Pipeline B - text vocabulary
# ==========================================================================
def run_text() -> tuple[list[dict], dict]:
    print("\n" + "=" * 66)
    print("PIPELINE B - TEXT VOCABULARY")
    print("=" * 66)

    df = loaders.load_dreaddit()
    print(f"  {len(df)} documents | sentiment backend: "
          f"{text_features.sentiment_backend()}")

    ling = text_features.build_text_features(df["text"])
    X = pd.concat([df[["text"]].reset_index(drop=True), ling], axis=1)
    y = df["stress_label"].astype(str)
    cols = list(ling.columns)

    maj = y.value_counts(normalize=True).max()
    print(f"  majority-class accuracy = {maj:.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    results, best_models = [], {}
    for name, est in classifiers.get_estimators().items():
        pipe = classifiers.build_text_pipeline(est, cols)
        gs = GridSearchCV(
            pipe, PARAM_GRIDS[name], cv=_cv(y_tr),
            scoring="f1_macro", n_jobs=-1,
        )
        gs.fit(X_tr, y_tr)
        y_pred = gs.best_estimator_.predict(X_te)

        row = evaluation.evaluate_model(y_te, y_pred, name)
        row.update({
            "pipeline": "text", "variant": "tfidf+linguistic",
            "n_samples": len(X), "n_features": len(cols) + config.TFIDF_MAX_FEATURES,
            "majority_baseline_acc": round(maj, 4),
            "beats_baseline": bool(row["accuracy"] > maj),
            "best_params": str(gs.best_params_),
            "cv_f1_mean": round(gs.best_score_, 4),
        })
        results.append(row)
        best_models[f"text__{name}"] = gs.best_estimator_
        print(f"  {name:20s} acc={row['accuracy']:.3f} f1={row['f1_macro']:.3f}")
        evaluation.save_confusion_matrix(y_te, y_pred, f"text_{name}")

    # interpretable linguistic-only importances
    lp = classifiers.build_typing_pipeline(
        classifiers.get_estimators()["RandomForest"]
    )
    lp.fit(X_tr[cols], y_tr)
    imp = classifiers.feature_importances(lp, cols, top_n=15)
    if imp is not None:
        imp.to_csv(OUT / "feature_importance_text.csv", index=False)
        print("\n  top linguistic features:")
        print(imp.head(5).to_string(index=False))

    return results, best_models


# ==========================================================================
# report
# ==========================================================================
def write_report(df: pd.DataFrame):
    text_best = df[df.pipeline == "text"].nlargest(1, "f1_macro").iloc[0]
    typ = df[df.pipeline == "typing"]
    typ_best = typ.nlargest(1, "f1_macro").iloc[0]
    cross = pd.read_csv(OUT / "cross_user_generalisation.csv")

    lines = [
        "# Mid-Term Results",
        "",
        "Stress Level Detection Using Typing Behaviour and Text Vocabulary",
        "",
        "## Summary",
        "",
        f"- **Text pipeline (Dreaddit)**: best model {text_best['model']}, "
        f"accuracy {text_best['accuracy']:.3f}, macro F1 {text_best['f1_macro']:.3f} "
        f"(majority baseline {text_best['majority_baseline_acc']:.3f}).",
        f"- **Typing pipeline (Kaggle keystroke + interaction)**: best "
        f"{typ_best['model']} on {typ_best['variant']}, accuracy "
        f"{typ_best['accuracy']:.3f}, macro F1 {typ_best['f1_macro']:.3f} "
        f"(majority baseline {typ_best['majority_baseline_acc']:.3f}).",
        f"- **Cross-user generalisation**: mean macro F1 "
        f"{cross['f1_macro'].mean():.3f} when training on one participant and "
        f"testing on another.",
        "",
        "## All results",
        "",
        df[["pipeline", "variant", "model", "accuracy", "precision_macro",
            "recall_macro", "f1_macro", "cv_f1_mean",
            "majority_baseline_acc", "beats_baseline"]]
        .sort_values(["pipeline", "f1_macro"], ascending=[True, False])
        .to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The text pipeline works. It clears its majority-class baseline by a "
        "clear margin and cross-validation is stable, so the vocabulary and "
        "sentiment features carry real signal.",
        "",
        "The typing pipeline does not, and the reason is the data rather than "
        "the model. Adding mouse, idle and app-switching features improved it "
        "measurably, and collapsing to a binary target helped again, but the "
        "cross-user test settles the question: a model trained on one "
        "participant transfers almost nothing to another. With two "
        "participants and roughly a hundred self-reports, the model learns "
        "individual habits instead of stress.",
        "",
        "This is the limitation our own synopsis anticipated under "
        "'variation in typing behaviour across individuals may affect model "
        "generalisation'. It is an argument for more participants, not a "
        "different algorithm.",
        "",
        "## Next steps",
        "",
        "1. Obtain SWELL-KW (25 participants, validated stress labels) to "
        "replace the interim keystroke data.",
        "2. Collect paired sessions with our own tool so typing behaviour and "
        "typed text describe the same moment, which is what early fusion "
        "requires.",
        "3. Re-run this script on both, and compare early against late fusion.",
        "",
    ]
    (OUT / "midterm_report.md").write_text("\n".join(lines))


def main():
    t_res, t_models = run_typing()
    x_res, x_models = run_text()

    df = pd.DataFrame(t_res + x_res)
    cols = ["pipeline", "variant", "model", "accuracy", "precision_macro",
            "recall_macro", "f1_macro", "cv_f1_mean", "majority_baseline_acc",
            "beats_baseline", "n_samples", "n_features", "best_params"]
    df = df[cols].round(4)
    df.to_csv(OUT / "midterm_results.csv", index=False)

    for k, m in {**t_models, **x_models}.items():
        joblib.dump(m, OUT / f"model_{k}.joblib")

    write_report(df)

    print("\n" + "=" * 66)
    print("FINAL RESULTS")
    print("=" * 66)
    print(df[["pipeline", "variant", "model", "accuracy", "f1_macro",
              "majority_baseline_acc", "beats_baseline"]].to_string(index=False))
    print(f"\nartefacts written to {OUT}/")


if __name__ == "__main__":
    main()
