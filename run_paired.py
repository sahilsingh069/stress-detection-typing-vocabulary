"""
Paired experiment - own collected sessions (typing + text + one label).

This is the experiment the project was designed for: the same session has
both a keystroke measurement and the participant's own words, under one
self-reported stress label. It compares

    typing only        keystroke timing features
    text only          TF-IDF + linguistic features
    early fusion       both feature sets in one vector, one classifier
    late fusion (avg)  weighted average of the two single-modality models
    late fusion (max)  whichever single-modality model is more confident
    stacking           meta-classifier over both models' probabilities

for Logistic Regression, Random Forest and SVM.

Evaluation holds out WHOLE PARTICIPANTS (StratifiedGroupKFold): every
prediction comes from a model that never saw that person. A random split
would let the model recognise people instead of stress.

Usage:
    python run_paired.py                      # data/own/ (collector exports)
    python run_paired.py --source synthetic   # smoke test, numbers are fake
    python run_paired.py --weight 0.4         # typing weight for late fusion

Writes to outputs/paired/ (outputs/synthetic/paired/ for --source synthetic).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import config
from dataio import loaders
from features import text_features
from modeling import classifiers, evaluation, fusion

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(source: str) -> pd.DataFrame:
    if source == "synthetic":
        print("Loading SYNTHETIC paired sessions (development only) ...")
        return loaders.load_synthetic_paired_frame()
    print(f"Loading collected sessions from {config.OWN_DIR} ...")
    return loaders.load_own_paired()


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """text + linguistic + typing columns, with the leak-prone ones removed."""
    ling = text_features.build_text_features(df["text"]).drop(
        columns=config.PAIRED_EXCLUDED_TEXT_FEATURES, errors="ignore"
    )
    typing_cols = [c for c in config.OWN_TYPING_FEATURES if c in df.columns]
    X = pd.concat(
        [df[["text"]].reset_index(drop=True), ling,
         df[typing_cols].reset_index(drop=True)],
        axis=1,
    )
    return X, list(ling.columns), typing_cols


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------
def _metrics(y, pred, mode, model, baseline, n) -> dict:
    row = evaluation.evaluate_model(y, pred, model)
    row.update({"mode": mode, "n_samples": n,
                "majority_baseline_acc": round(baseline, 4),
                "beats_baseline": bool(row["accuracy"] > baseline)})
    return row


def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().items()}


def _labels(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.array(classes)[proba.argmax(axis=1)]


def run_experiment(df: pd.DataFrame, weight: float, out_dir: Path) -> pd.DataFrame:
    text_df = df[df["task_type"].isin(config.OWN_TEXT_TASKS)
                 & df["text"].str.strip().astype(bool)].reset_index(drop=True)
    if text_df.empty:
        raise ValueError("No chat / free-writing sessions with text were found.")

    X, ling_cols, typing_cols = build_matrix(text_df)
    y = text_df["stress_label"].astype(str)
    groups = text_df["participant_id"].astype(str).to_numpy()
    cv = evaluation.group_cv(y, groups)
    if cv is None:
        raise ValueError("Grouped evaluation needs at least two participants.")
    baseline = evaluation.majority_baseline(y)
    n = len(y)

    print(f"\nPaired sessions (chat / free text): {n} from "
          f"{len(np.unique(groups))} participants, {cv.get_n_splits()} grouped folds")
    print(f"Label distribution: {_counts(y)}")
    print(f"Majority-class baseline accuracy: {baseline:.3f}")
    print(f"Typing features: {typing_cols}")

    text_cols = ["text"] + ling_cols
    rows, confusion = [], {}
    for name, est in classifiers.get_estimators().items():
        typing_pipe = classifiers.build_typing_pipeline(est)
        text_pipe = classifiers.build_text_pipeline(est, ling_cols)
        early_pipe = classifiers.build_early_fusion_pipeline(est, ling_cols, typing_cols)

        p_typ, classes = evaluation.oof_proba(typing_pipe, X[typing_cols], y, groups, cv)
        p_txt, _ = evaluation.oof_proba(text_pipe, X[text_cols], y, groups, cv)
        p_early, _ = evaluation.oof_proba(early_pipe, X, y, groups, cv)

        avg_lbl, _ = fusion.weighted_average_fusion(
            p_typ, p_txt, weight_typing=weight, target_classes=classes)
        max_lbl, _ = fusion.max_confidence_fusion(p_typ, p_txt, target_classes=classes)
        # stacking: meta-model over out-of-fold probabilities, itself scored
        # out-of-fold on the same participant folds
        meta = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=config.RANDOM_STATE)
        stack_lbl = cross_val_predict(meta, np.hstack([p_typ, p_txt]), y,
                                      groups=groups, cv=cv)

        preds = {
            "typing_only": _labels(p_typ, classes),
            "text_only": _labels(p_txt, classes),
            "early_fusion": _labels(p_early, classes),
            f"late_fusion_avg(w={weight})": avg_lbl,
            "late_fusion_maxconf": max_lbl,
            "stacking": stack_lbl,
        }
        for mode, pred in preds.items():
            rows.append(_metrics(y, pred, mode, name, baseline, n))
            confusion[(mode, name)] = pred
        print(f"  {name:20s} " + "  ".join(
            f"{m.split('(')[0]}={r['f1_macro']:.3f}"
            for m, r in zip(preds, rows[-len(preds):])))

        if name == "LogisticRegression":
            sweep = fusion.tune_fusion_weight(p_typ, p_txt, y, target_classes=classes)
            sweep.to_csv(out_dir / "fusion_weight_sweep.csv", index=False)

    # typing on every session, copy tasks included (more data, no text needed)
    all_X = df[typing_cols]
    all_y = df["stress_label"].astype(str)
    all_g = df["participant_id"].astype(str).to_numpy()
    all_cv = evaluation.group_cv(all_y, all_g)
    for name, est in classifiers.get_estimators().items():
        pipe = classifiers.build_typing_pipeline(est)
        p, classes = evaluation.oof_proba(pipe, all_X, all_y, all_g, all_cv)
        rows.append(_metrics(all_y, _labels(p, classes), "typing_only_all_tasks",
                             name, evaluation.majority_baseline(all_y), len(all_y)))

    table = pd.DataFrame(rows)
    num = table.select_dtypes(include=[np.number]).columns
    table[num] = table[num].round(4)
    table = table[["mode", "model", "accuracy", "precision_macro", "recall_macro",
                   "f1_macro", "majority_baseline_acc", "beats_baseline",
                   "n_samples"]].sort_values(["f1_macro"], ascending=False)

    # confusion matrices: best estimator for each mode on the paired set
    paired = table[table["mode"] != "typing_only_all_tasks"]
    for mode, grp in paired.groupby("mode"):
        best = grp.iloc[0]["model"]
        evaluation.save_confusion_matrix(
            y, confusion[(mode, best)], f"paired_{mode}_{best}",
            out_dir=out_dir, labels=[c for c in config.STRESS_CLASSES if c in set(y)],
        )

    # interpretable feature ranking from an early-fusion-style RF on the
    # hand-crafted columns only (TF-IDF columns are not human-readable)
    rf = classifiers.build_typing_pipeline(classifiers.get_estimators()["RandomForest"])
    rf.fit(X[ling_cols + typing_cols], y)
    imp = classifiers.feature_importances(rf, ling_cols + typing_cols, top_n=20)
    if imp is not None:
        imp["modality"] = np.where(imp["feature"].isin(typing_cols), "typing", "text")
        imp.to_csv(out_dir / "feature_importance_paired.csv", index=False)

    return table


def transfer_from_dreaddit(df: pd.DataFrame) -> dict | None:
    """
    Cross-domain check: a text model trained only on Dreaddit (Reddit posts)
    applied to our participants' chat answers. Dreaddit is binary, so this
    is scored as High vs. not-High.
    """
    if not config.DREADDIT_TRAIN_PATH.exists():
        return None
    text_df = df[df["task_type"].isin(config.OWN_TEXT_TASKS)
                 & df["text"].str.strip().astype(bool)]
    dd = loaders.load_dreaddit()
    ling_dd = text_features.build_text_features(dd["text"])
    pipe = classifiers.build_text_pipeline(
        classifiers.get_estimators()["LogisticRegression"], list(ling_dd.columns))
    pipe.fit(pd.concat([dd[["text"]], ling_dd], axis=1), dd["stress_label"])

    ling_own = text_features.build_text_features(text_df["text"])
    pred = pipe.predict(pd.concat(
        [text_df[["text"]].reset_index(drop=True), ling_own], axis=1))
    truth = np.where(text_df["stress_label"].to_numpy() == "High", "High", "NotHigh")
    pred = np.where(pred == "High", "High", "NotHigh")
    row = evaluation.evaluate_model(truth, pred, "LogisticRegression (Dreaddit-trained)")
    row["majority_baseline_acc"] = evaluation.majority_baseline(truth)
    row["n_samples"] = len(truth)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def manipulation_check(df: pd.DataFrame) -> str:
    """Did the time-pressure condition actually feel more stressful?"""
    if "condition" not in df.columns:
        return "_No condition column in the data._"
    blocks = df.drop_duplicates(subset=["participant_id", "condition", "task_type"])
    ct = pd.crosstab(blocks["condition"], blocks["stress_label"])
    ct = ct[[c for c in config.STRESS_CLASSES if c in ct.columns]]
    lines = ["Self-reported stress per block, by condition:", "",
             ct.to_markdown(), ""]
    for col in ["self_report_pressure", "self_report_intensity"]:
        if col in blocks.columns:
            m = blocks.groupby("condition")[col].mean().round(2).to_dict()
            lines.append(f"- mean {col.replace('self_report_', '')} rating (1-5): {m}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(df, table, transfer, weight, out_dir, source):
    paired = table[table["mode"] != "typing_only_all_tasks"]
    best_by_mode = paired.groupby("mode")["f1_macro"].max().sort_values(ascending=False)
    single = max(best_by_mode.get("typing_only", 0), best_by_mode.get("text_only", 0))
    fused_modes = [m for m in best_by_mode.index if m not in ("typing_only", "text_only")]
    best_fused = best_by_mode[fused_modes].max() if fused_modes else np.nan
    best_fused_mode = best_by_mode[fused_modes].idxmax() if fused_modes else "-"
    n_part = df["participant_id"].nunique()

    lines = [
        "# Paired Experiment Results",
        "",
        "Typing behaviour and text vocabulary measured in the same sessions.",
        "",
    ]
    if source == "synthetic":
        lines += ["> **SYNTHETIC DATA. These numbers come from generated sessions "
                  "with a planted signal. Never quote them.**", ""]
    if n_part < 10:
        lines += [f"> Only {n_part} participants: treat every number below as "
                  "provisional. Target 20-30.", ""]
    lines += [
        "## Data",
        "",
        f"- Participants: {n_part}",
        f"- Sessions with a typing measurement: {len(df)} "
        f"({(df['task_type'] == 'copy').sum()} copy tasks, "
        f"{df['task_type'].isin(config.OWN_TEXT_TASKS).sum()} chat / free-text answers)",
        f"- Label distribution (chat / free text): "
        f"{_counts(df.loc[df['task_type'].isin(config.OWN_TEXT_TASKS), 'stress_label'])}",
        "",
        "## Manipulation check",
        "",
        manipulation_check(df),
        "",
        "## Results (whole participants held out)",
        "",
        table.to_markdown(index=False),
        "",
        "## Summary",
        "",
        f"- Best single modality macro F1: {single:.3f}",
        f"- Best fusion macro F1: {best_fused:.3f} ({best_fused_mode})",
        f"- Fusion {'improves on' if best_fused > single else 'does not improve on'} "
        f"the best single modality by {best_fused - single:+.3f} macro F1.",
        f"- Late-fusion typing weight used: {weight} "
        "(`fusion_weight_sweep.csv` shows all weights; choosing the best one "
        "after seeing it is optimistic, so the fixed weight is the reported one).",
        "",
    ]
    if transfer:
        lines += [
            "## Cross-domain check: Dreaddit text model on our answers",
            "",
            "Scored as High vs. not-High because Dreaddit is binary.",
            "",
            pd.DataFrame([transfer]).to_markdown(index=False),
            "",
        ]
    lines += [
        "## What the models could and could not see",
        "",
        f"- Typing features: {', '.join(config.OWN_TYPING_FEATURES)}.",
        "- Excluded on purpose: block number, condition, time limit, "
        "duration, raw key and word counts (all fixed or capped by the "
        "timer), and every self-rating (those are the label).",
        "- Text stream uses chat / free-writing answers only; copy tasks "
        "reproduce a fixed passage.",
        "",
    ]
    (out_dir / "report_paired.md").write_text("\n".join(lines))


def main(source: str = "own", weight: float = 0.5) -> pd.DataFrame:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--weight must be between 0 and 1.")
    out_dir = config.OUTPUT_DIR / ("synthetic/paired" if source == "synthetic" else "paired")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(source)
    table = run_experiment(df, weight, out_dir)
    transfer = transfer_from_dreaddit(df) if source == "own" else None
    table.to_csv(out_dir / "results_paired.csv", index=False)
    write_report(df, table, transfer, weight, out_dir, source)

    print("\n================ PAIRED RESULTS (grouped by participant) ================")
    print(table.to_string(index=False))
    if transfer:
        print(f"\nDreaddit text model on own answers (High vs not-High): "
              f"acc={transfer['accuracy']:.3f} f1={transfer['f1_macro']:.3f}")
    print(f"\nartefacts written to {out_dir}/")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="own", choices=["own", "synthetic"])
    ap.add_argument("--weight", type=float, default=0.5,
                    help="typing weight for weighted late fusion (0-1)")
    args = ap.parse_args()
    main(args.source, args.weight)
