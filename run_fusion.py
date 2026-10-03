"""
Late-fusion demonstration.

Trains both modality models, aligns their probability outputs onto the shared
3-class space, and combines them at the decision level.

IMPORTANT - read before quoting any number from this script:
SWELL-KW and Dreaddit are not paired (different people, different sessions),
so there is no ground truth for a fused sample. This script therefore
demonstrates that the fusion MECHANISM works and is correctly implemented; it
does not report a validated fused accuracy. Once you collect paired data
(same session -> typing + text + one stress label), pass it here and the same
code produces genuine fused results, plus StackingFusion becomes usable.

Usage:
    python run_fusion.py --typing-source synthetic --text-source synthetic
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from dataio import loaders
from features import text_features, typing_features
from modeling import classifiers, evaluation, fusion

warnings.filterwarnings("ignore", category=UserWarning)


def train_typing(source: str):
    df = (
        loaders.load_swell() if source == "swell"
        else typing_features.features_from_events(
            loaders.load_kaggle_keystrokes()
        ) if source == "kaggle"
        else loaders.load_synthetic_typing()
    )
    X, y = typing_features.prepare_feature_frame(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    pipe = classifiers.build_typing_pipeline(
        classifiers.get_estimators()["RandomForest"]
    )
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


def train_text(source: str):
    df = (
        loaders.load_dreaddit() if source == "dreaddit"
        else loaders.load_synthetic_text()
    )
    ling = text_features.build_text_features(df["text"])
    X = pd.concat([df[["text"]].reset_index(drop=True), ling], axis=1)
    y = df["stress_label"].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    pipe = classifiers.build_text_pipeline(
        classifiers.get_estimators()["RandomForest"], list(ling.columns)
    )
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


def main(typing_source: str, text_source: str, weight: float):
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--weight must be between 0 and 1.")

    print("Training typing model ...")
    typing_pipe, Xt_te, yt_te = train_typing(typing_source)
    print("Training text model ...")
    text_pipe, Xx_te, yx_te = train_text(text_source)

    # individual performance
    print("\n--- individual modality performance ---")
    t_res = evaluation.evaluate_model(
        yt_te, typing_pipe.predict(Xt_te), "Typing (RandomForest)"
    )
    x_res = evaluation.evaluate_model(
        yx_te, text_pipe.predict(Xx_te), "Text (RandomForest)"
    )
    print(pd.DataFrame([t_res, x_res]).round(4).to_string(index=False))

    # align probabilities onto the shared 3-class space
    p_typing = fusion.align_classes(
        typing_pipe.predict_proba(Xt_te),
        [str(c) for c in typing_pipe.classes_],
    )
    p_text = fusion.align_classes(
        text_pipe.predict_proba(Xx_te),
        [str(c) for c in text_pipe.classes_],
    )
    print(f"\nAligned typing probs: {p_typing.shape}")
    print(f"Aligned text probs:   {p_text.shape}")
    print(f"Shared class space:   {config.STRESS_CLASSES}")

    # unpaired -> truncate to a common length purely to exercise the mechanism
    n = min(len(p_typing), len(p_text))
    p_typing, p_text = p_typing[:n], p_text[:n]

    labels_avg, probs_avg = fusion.weighted_average_fusion(
        p_typing, p_text, weight_typing=weight
    )
    labels_max, _ = fusion.max_confidence_fusion(p_typing, p_text)

    print(f"\n--- fusion mechanism check (n={n}, w_typing={weight}) ---")
    print("weighted-average predicted distribution:")
    print(pd.Series(labels_avg).value_counts().to_string())
    print("\nmax-confidence predicted distribution:")
    print(pd.Series(labels_max).value_counts().to_string())
    print(f"\nrows sum to 1: {np.allclose(probs_avg.sum(axis=1), 1.0)}")

    demo = pd.DataFrame(probs_avg[:5], columns=config.STRESS_CLASSES)
    demo["fused_prediction"] = labels_avg[:5]
    print("\nsample fused outputs:")
    print(demo.round(3).to_string(index=False))

    print(
        "\nNOTE: no fused accuracy reported - the two datasets are unpaired, "
        "so no fused ground truth exists. Supply paired sessions to get "
        "validated fusion results and to enable StackingFusion."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--typing-source", default="synthetic",
                    choices=["swell", "kaggle", "synthetic"])
    ap.add_argument("--text-source", default="synthetic",
                    choices=["dreaddit", "synthetic"])
    ap.add_argument("--weight", type=float, default=0.5,
                    help="weight on the typing modality (0-1)")
    args = ap.parse_args()
    main(args.typing_source, args.text_source, args.weight)
