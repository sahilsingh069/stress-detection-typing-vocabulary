"""
Evaluation utilities.

Reports accuracy, macro precision / recall / F1 and the confusion matrix -
macro averages matter here because the stress classes are imbalanced and
plain accuracy hides poor minority-class performance.
"""

from __future__ import annotations

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
from sklearn.model_selection import StratifiedKFold, cross_val_score

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


def cross_validate(pipeline, X, y, folds: int | None = None) -> dict:
    """Stratified k-fold CV on macro-F1 - more trustworthy on small data."""
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

    path = out_dir / f"confusion_{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def results_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
