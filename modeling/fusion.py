"""
Late (decision-level) fusion.

Why late fusion: SWELL-KW gives us typing behaviour and Dreaddit gives us
text, but they come from DIFFERENT people - there are no paired rows, so a
single concatenated feature vector (early fusion) is not possible yet. Late
fusion trains one model per modality independently and combines only their
predicted probabilities at inference time.

Three combination rules are provided:
    weighted_average  - w * P_typing + (1-w) * P_text
    max_confidence    - take whichever modality is more confident
    stacking          - a small meta-classifier trained on both prob vectors
                        (requires paired data; use once you collect your own)

`align_classes` handles the fact that Dreaddit is binary (Low/High) while the
typing stream is 3-class (Low/Medium/High): probabilities are expanded onto
the shared 3-class space before combining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config


def align_classes(
    proba: np.ndarray,
    model_classes: list[str],
    target_classes: list[str] | None = None,
) -> np.ndarray:
    """
    Expand a probability matrix onto the canonical class list.
    Classes the model never saw get probability 0.
    """
    target_classes = target_classes or config.STRESS_CLASSES
    out = np.zeros((proba.shape[0], len(target_classes)))
    for i, cls in enumerate(model_classes):
        if str(cls) in target_classes:
            out[:, target_classes.index(str(cls))] = proba[:, i]
    # renormalise so rows sum to 1
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sum, out=out, where=row_sum > 0)
    return out


def weighted_average_fusion(
    proba_typing: np.ndarray,
    proba_text: np.ndarray,
    weight_typing: float = 0.5,
    target_classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two aligned probability matrices. Returns (labels, probs)."""
    target_classes = target_classes or config.STRESS_CLASSES
    if not 0.0 <= weight_typing <= 1.0:
        raise ValueError("weight_typing must be between 0 and 1.")
    if proba_typing.shape != proba_text.shape:
        raise ValueError(
            "Probability matrices must be aligned to the same class space "
            "first - call align_classes() on each."
        )
    combined = weight_typing * proba_typing + (1 - weight_typing) * proba_text
    labels = np.array([target_classes[i] for i in combined.argmax(axis=1)])
    return labels, combined


def max_confidence_fusion(
    proba_typing: np.ndarray,
    proba_text: np.ndarray,
    target_classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trust whichever modality is more confident for each sample."""
    target_classes = target_classes or config.STRESS_CLASSES
    conf_t = proba_typing.max(axis=1)
    conf_x = proba_text.max(axis=1)
    pick_typing = (conf_t >= conf_x)[:, None]
    combined = np.where(pick_typing, proba_typing, proba_text)
    labels = np.array([target_classes[i] for i in combined.argmax(axis=1)])
    return labels, combined


class StackingFusion:
    """
    Meta-classifier over concatenated probability vectors.
    Only usable when you have PAIRED samples (same session, both modalities).
    """

    def __init__(self, random_state: int | None = None):
        self.meta = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state or config.RANDOM_STATE,
        )

    def fit(self, proba_typing: np.ndarray, proba_text: np.ndarray, y):
        _validate_fusion_shapes(proba_typing, proba_text)
        self.meta.fit(np.hstack([proba_typing, proba_text]), y)
        return self

    def predict(self, proba_typing: np.ndarray, proba_text: np.ndarray):
        _validate_fusion_shapes(proba_typing, proba_text)
        return self.meta.predict(np.hstack([proba_typing, proba_text]))

    def predict_proba(self, proba_typing: np.ndarray, proba_text: np.ndarray):
        _validate_fusion_shapes(proba_typing, proba_text)
        return self.meta.predict_proba(np.hstack([proba_typing, proba_text]))


def _validate_fusion_shapes(
    proba_typing: np.ndarray,
    proba_text: np.ndarray,
) -> None:
    if proba_typing.ndim != 2 or proba_text.ndim != 2:
        raise ValueError("Fusion probability inputs must be two-dimensional.")
    if proba_typing.shape != proba_text.shape:
        raise ValueError("Fusion probability inputs must have matching shapes.")


def tune_fusion_weight(
    proba_typing: np.ndarray,
    proba_text: np.ndarray,
    y_true,
    target_classes: list[str] | None = None,
    step: float = 0.05,
) -> pd.DataFrame:
    """
    Sweep the typing/text weight and report macro-F1 at each setting.
    Needs paired validation data.
    """
    from sklearn.metrics import f1_score

    rows = []
    for w in np.arange(0.0, 1.0 + step, step):
        labels, _ = weighted_average_fusion(
            proba_typing, proba_text, weight_typing=float(w),
            target_classes=target_classes,
        )
        rows.append(
            {
                "weight_typing": round(float(w), 2),
                "f1_macro": f1_score(
                    y_true, labels, average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
