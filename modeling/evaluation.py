"""
Evaluation utilities.

Reports accuracy, macro precision / recall / F1 and the confusion matrix -
macro averages matter here because the stress classes are imbalanced and
plain accuracy hides poor minority-class performance.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

import config


def evaluate_model(y_true, y_pred, name: str = "model") -> dict:
    """Compute the standard metric set for one model."""
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def cross_validate(pipeline, X, y, folds: int | None = None, groups=None) -> dict:
    """
    Stratified k-fold CV on macro-F1 - more trustworthy on small data.
    With `groups` (participant ids) whole participants are held out.
    """
    if groups is not None:
        cv = group_cv(y, groups, folds)
        if cv is None:
            return {"cv_f1_mean": np.nan, "cv_f1_std": np.nan}
        scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv,
                                 scoring="f1_macro")
        return {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}
    folds = folds or config.CV_FOLDS
    counts = pd.Series(y).value_counts()
    folds = int(min(folds, counts.min())) if counts.min() >= 2 else 0
    if folds < 2:
        return {"cv_f1_mean": np.nan, "cv_f1_std": np.nan}

    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    return {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}


def print_report(y_true, y_pred, name: str, labels: list[str] | None = None):
    labels = labels or sorted(set(map(str, y_true)))
    print(f"\n--- {name} ---")
    print(
        classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=3
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


def save_confusion_matrix(
    y_true, y_pred, name: str, out_dir: Path | None = None,
    labels: list[str] | None = None,
) -> Path:
    """Write a confusion-matrix PNG - useful directly in the report/slides."""
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or sorted(set(map(str, y_true)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    safe = re.sub(r"[^a-z0-9_+.-]+", "_", name.lower()).strip("_")
    path = out_dir / f"confusion_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def results_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# participant-grouped evaluation
# --------------------------------------------------------------------------
# With several samples per person, a random split puts the same person in
# both train and test, and the model can score well by recognising people
# instead of stress (the Kaggle cross-user test showed exactly this). Every
# dataset that has participant ids is therefore evaluated with whole
# participants held out.

def majority_baseline(y) -> float:
    return float(pd.Series(y).value_counts(normalize=True).max())


def group_cv(y, groups, max_folds: int | None = None):
    """
    StratifiedGroupKFold over participants, or None when fewer than two
    participants exist (grouped evaluation is impossible).
    """
    max_folds = max_folds or config.GROUP_CV_FOLDS
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        return None
    return StratifiedGroupKFold(
        n_splits=int(min(max_folds, n_groups)),
        shuffle=True, random_state=config.RANDOM_STATE,
    )


def grouped_train_test_split(X, y, groups, test_fraction: float | None = None):
    """
    Hold out roughly `test_fraction` of participants (one fold of a
    StratifiedGroupKFold). Falls back to a plain stratified split when there
    is only one participant.
    """
    from sklearn.model_selection import train_test_split

    test_fraction = test_fraction or config.TEST_SIZE
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return train_test_split(
            X, y, test_size=test_fraction, random_state=config.RANDOM_STATE,
            stratify=y if pd.Series(y).value_counts().min() >= 2 else None,
        )
    folds = int(min(max(2, round(1 / test_fraction)), n_groups))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                              random_state=config.RANDOM_STATE)
    tr, te = next(cv.split(X, y, groups))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def oof_proba(pipeline, X, y, groups, cv) -> tuple[np.ndarray, list[str]]:
    """
    Out-of-fold class probabilities: every sample is predicted by a model
    that never saw its participant. Returns (proba, class order).
    """
    classes = sorted(pd.Series(y).astype(str).unique())
    proba = cross_val_predict(
        pipeline, X, y, groups=groups, cv=cv, method="predict_proba",
    )
    return proba, classes


def leave_one_group_out_f1(pipeline, X, y, groups) -> pd.DataFrame:
    """Train on everyone but one participant, test on that one, for each."""
    from sklearn.base import clone

    groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        te = groups == g
        if pd.Series(y[~te]).nunique() < 2:
            continue
        model = clone(pipeline).fit(X[~te], y[~te])
        pred = model.predict(X[te])
        rows.append({
            "held_out": g, "n": int(te.sum()),
            "accuracy": accuracy_score(y[te], pred),
            "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)
