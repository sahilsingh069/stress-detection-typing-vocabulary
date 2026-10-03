"""
Model definitions and training routines.

Three classifiers are trained and compared for BOTH modalities, exactly as
described in the synopsis:

    Logistic Regression  - interpretable baseline
    Random Forest        - primary model (handles mixed features, gives
                           feature importances)
    SVM (RBF)            - strong performer on smaller feature sets

Class imbalance is handled via config.IMBALANCE_STRATEGY:
    "class_weight" -> class_weight='balanced' inside each estimator
    "smote"        -> SMOTE oversampling inside an imblearn Pipeline
    "none"         -> no correction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    _HAS_IMBLEARN = False


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def get_estimators(strategy: str | None = None) -> dict:
    """Return the three candidate classifiers, configured for imbalance."""
    strategy = strategy or config.IMBALANCE_STRATEGY
    cw = "balanced" if strategy == "class_weight" else None

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight=cw,
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=cw,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,          # needed for late fusion
            class_weight=cw,
            random_state=config.RANDOM_STATE,
        ),
    }


def _wrap(steps: list, strategy: str) -> SkPipeline:
    """Insert SMOTE into the pipeline when requested and available."""
    if strategy == "smote" and _HAS_IMBLEARN:
        head, tail = steps[:-1], steps[-1]
        return ImbPipeline(
            head + [("smote", SMOTE(random_state=config.RANDOM_STATE)), tail]
        )
    return SkPipeline(steps)


# --------------------------------------------------------------------------
# typing pipeline: scale -> classifier
# --------------------------------------------------------------------------
def build_typing_pipeline(
    estimator, strategy: str | None = None
) -> SkPipeline:
    strategy = strategy or config.IMBALANCE_STRATEGY
    return _wrap(
        [("scaler", StandardScaler()), ("clf", estimator)], strategy
    )


# --------------------------------------------------------------------------
# text pipeline: [TF-IDF on raw text] + [scaled linguistic features] -> clf
# --------------------------------------------------------------------------
def build_text_pipeline(
    estimator, linguistic_cols: list[str], strategy: str | None = None
) -> SkPipeline:
    """
    Expects a DataFrame with a 'text' column plus the linguistic feature
    columns produced by features.text_features.build_text_features().
    """
    strategy = strategy or config.IMBALANCE_STRATEGY

    pre = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.TFIDF_MAX_FEATURES,
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    stop_words="english",
                    sublinear_tf=True,
                ),
                "text",
            ),
            ("ling", StandardScaler(), linguistic_cols),
        ],
        remainder="drop",
    )
    return _wrap([("pre", pre), ("clf", estimator)], strategy)


# --------------------------------------------------------------------------
# feature importance helper (Random Forest / Logistic Regression)
# --------------------------------------------------------------------------
def feature_importances(
    fitted_pipeline, feature_names: list[str], top_n: int = 15
) -> pd.DataFrame | None:
    """Extract interpretable feature importances where the model supports it."""
    clf = fitted_pipeline.named_steps.get("clf")
    if clf is None:
        return None

    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).mean(axis=0)
    else:
        return None

    if len(vals) != len(feature_names):
        feature_names = [f"f{i}" for i in range(len(vals))]

    return (
        pd.DataFrame({"feature": feature_names, "importance": vals})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
