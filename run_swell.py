"""
SWELL-KW typing-behaviour experiment.

SWELL-KW (Koldijk et al., 2014): 25 office workers, per-minute
computer-interaction features (keystrokes, error keys, mouse, app switches)
under three conditions - neutral (N), interruptions (I), time pressure (T).
Mapped to Low / Medium / High; a binary view (no stress vs. stress) is
reported too.

Evaluation:
  grouped     whole participants held out (StratifiedGroupKFold) - the
              headline number, because a deployed system meets new people
  calibrated  same, but each participant's features are first z-scored
              against that participant's own mean and spread. This uses the
              test person's unlabelled typing (no labels), i.e. it models a
              system that observes a new user for a while before judging
              them. Reported separately, never mixed with 'grouped'.
  random      ordinary stratified 5-fold that mixes people across train and
              test - reported only to show how much person-leakage inflates
              scores.

Usage:
    python run_swell.py                     # config.SWELL_PATH
    python run_swell.py --path my_file.csv
    python run_swell.py --source synthetic  # smoke test, numbers are fake
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from dataio import loaders
from features import typing_features
from modeling import classifiers, evaluation

warnings.filterwarnings("ignore")


def normalise_per_participant(X: pd.DataFrame, groups) -> pd.DataFrame:
    """z-score every feature within each participant (labels never used)."""
    out = X.copy().astype(float)
    for g in np.unique(groups):
        m = np.asarray(groups) == g
        block = out.loc[m]
        sd = block.std(ddof=0).replace(0, 1.0)
        out.loc[m] = (block - block.mean()) / sd
    return out.fillna(0.0)


def _row(y, pred, variant, evaluation_kind, model, n_feat) -> dict:
    r = evaluation.evaluate_model(y, pred, model)
    base = evaluation.majority_baseline(y)
    r.update({"variant": variant, "evaluation": evaluation_kind,
              "majority_baseline_acc": base,
              "beats_baseline": bool(r["accuracy"] > base),
              "n_samples": len(y), "n_features": n_feat})
    return r


KEYBOARD_FEATURES = ["SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys",
                     "SnErrorKeys", "SnShortcutKeys", "SnSpaces"]


def run(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups_all = df["participant"].astype(str).to_numpy()
    X_all, y3 = typing_features.prepare_feature_frame(df)
    n_part = len(np.unique(groups_all))
    print(f"\nSWELL-KW: {len(X_all)} minutes x {X_all.shape[1]} features, "
          f"{n_part} participants")
    print(f"features: {list(X_all.columns)}")
    print(f"3-class distribution: {y3.value_counts().to_dict()}")

    # Calibration statistics come from ALL of a participant's working
    # minutes, before any subset is chosen, so they never depend on labels.
    Xcal_all = normalise_per_participant(X_all, groups_all)

    # Targets:
    #   3class  N / I / T -> Low / Medium / High
    #   binary  N vs. (I or T)
    #   n_vs_t  N vs. T only: same kind of work (N is always block 1, though). The
    #           interruption condition is excluded because its e-mails force
    #           app switches and extra keys by design (see report).
    keep_nt = y3.isin(["Low", "High"]).to_numpy()
    targets = [
        ("3class", np.ones(len(y3), bool), y3),
        ("binary", np.ones(len(y3), bool), y3.map(config.SWELL_BINARY_MAP)),
        ("n_vs_t", keep_nt, y3.map({"Low": "Neutral", "High": "TimePressure"})),
    ]
    kb_cols = [c for c in KEYBOARD_FEATURES if c in X_all.columns]
    feature_sets = [("all interaction", list(X_all.columns)), ("keyboard only", kb_cols)]

    rows = []
    for target_name, mask, y_full in targets:
        y = y_full[mask].reset_index(drop=True)
        groups = groups_all[mask]
        cv = evaluation.group_cv(y, groups)
        if cv is None:
            raise ValueError("SWELL file has a single participant; cannot evaluate.")
        labels = [c for c in ["Low", "Medium", "High", "NoStress", "Stress",
                              "Neutral", "TimePressure"] if c in set(y)]
        for fs_name, cols in feature_sets:
            if not cols:
                continue
            X = X_all.loc[mask, cols].reset_index(drop=True)
            Xcal = Xcal_all.loc[mask, cols].reset_index(drop=True)
            for kind, feats in [("grouped", X), ("calibrated", Xcal)]:
                for name, est in classifiers.get_estimators().items():
                    pipe = classifiers.build_typing_pipeline(est)
                    pred = cross_val_predict(pipe, feats, y, groups=groups, cv=cv)
                    r = _row(y, pred, target_name, kind, name, len(cols))
                    r["features"] = fs_name
                    rows.append(r)
                    print(f"  [{target_name:6s} {fs_name:15s} {kind:10s}] {name:20s} "
                          f"acc={r['accuracy']:.3f} f1={r['f1_macro']:.3f}")
                    if name == "RandomForest":
                        evaluation.save_confusion_matrix(
                            y, pred, f"swell_{target_name}_{fs_name}_{kind}_{name}",
                            out_dir=out_dir, labels=labels)

            if fs_name == "all interaction":
                # person-leakage comparison: the same RF, people mixed across folds
                rf = classifiers.build_typing_pipeline(
                    classifiers.get_estimators()["RandomForest"])
                skf = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
                pred = cross_val_predict(rf, X, y, cv=skf)
                r = _row(y, pred, target_name, "random (people mixed)",
                         "RandomForest", len(cols))
                r["features"] = fs_name
                rows.append(r)

    X, groups = X_all, groups_all
    Xcal = Xcal_all
    table = pd.DataFrame(rows)
    num = table.select_dtypes(include=[np.number]).columns
    table[num] = table[num].round(4)
    table = table[["variant", "features", "evaluation", "model", "accuracy", "precision_macro",
                   "recall_macro", "f1_macro", "majority_baseline_acc",
                   "beats_baseline", "n_samples", "n_features"]]

    # per-participant spread for the headline configuration
    rf = classifiers.build_typing_pipeline(classifiers.get_estimators()["RandomForest"])
    logo = evaluation.leave_one_group_out_f1(rf, X, y3, groups).round(4)
    logo_cal = evaluation.leave_one_group_out_f1(rf, Xcal, y3, groups).round(4)
    logo = logo.merge(logo_cal[["held_out", "f1_macro"]], on="held_out",
                      suffixes=("", "_calibrated"))

    rf.fit(X, y3)
    imp = classifiers.feature_importances(rf, list(X.columns), top_n=20)
    if imp is not None:
        imp.to_csv(out_dir / "feature_importance_swell.csv", index=False)
    return table, logo


def write_report(table, logo, out_dir, source):
    def best(variant, features, kind):
        t = table[(table.variant == variant) & (table.features == features)
                  & (table.evaluation == kind)]
        return t.loc[t.f1_macro.idxmax()] if len(t) else None

    names = {"3class": "3-class (N / I / T)", "binary": "Binary (N vs. I or T)",
             "n_vs_t": "Neutral vs. time pressure"}
    lines = ["# SWELL-KW Results", ""]
    if source == "synthetic":
        lines += ["> **SYNTHETIC DATA shaped like SWELL-KW. Never quote these numbers.**", ""]
    lines += ["## Headline (macro F1, best of LR / RF / SVM)", "",
              "| Task | Features | New person F1 | chance F1 | New person acc | majority acc | Calibrated F1 | People mixed F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for variant in ["3class", "binary", "n_vs_t"]:
        for features in ["all interaction", "keyboard only"]:
            g = best(variant, features, "grouped")
            if g is None:
                continue
            c = best(variant, features, "calibrated")
            r = best(variant, features, "random (people mixed)")
            k = 3 if variant == "3class" else 2
            lines.append(
                f"| {names[variant]} | {features} | **{g.f1_macro:.3f}** ({g.model}) | "
                f"{1 / k:.3f} | {g.accuracy:.3f} | {g.majority_baseline_acc:.3f} | "
                f"{c.f1_macro:.3f} | {'—' if r is None else f'{r.f1_macro:.3f}'} |")
    lines += [
        "",
        "*chance F1* is the macro F1 of guessing at random in proportion to the "
        "classes (1 / number of classes); *majority acc* is the accuracy of always "
        "guessing the most common class. Macro F1 is compared with chance F1, "
        "accuracy with majority accuracy.",
        "",
        "*New person*: whole participants held out — the headline. *Calibrated*: "
        "each person's features z-scored against their own working minutes "
        "(no labels used). *People mixed*: ordinary random folds, shown only to "
        "measure how much a random split would overstate the result.",
        "",
        "## Read this before quoting the 3-class or binary result",
        "",
        "In the interruption condition participants receive e-mails they must "
        "open and answer, so app switches and extra keystrokes rise *because of "
        "the task design*. A model can recognise the interruption condition from "
        "that alone, which is not the same as recognising stress. The "
        "**neutral vs. time pressure** task avoids this because the work is the "
        "same kind, but neutral was always the first block, so it also compares "
        "early with late in the session. The keyboard-only rows show what typing "
        "carries without mouse and app-switch features. Figures here are best of "
        "three models; `run_evaluation.py` tests one model fixed in advance.",
        "",
        "## All results",
        "",
        table.to_markdown(index=False),
        "",
        "## Leave-one-participant-out (Random Forest, 3-class, all interaction features)",
        "",
        f"Mean macro F1 {logo.f1_macro.mean():.3f} "
        f"(min {logo.f1_macro.min():.3f}, max {logo.f1_macro.max():.3f}); "
        f"calibrated mean {logo.f1_macro_calibrated.mean():.3f}.",
        "",
        logo.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Inputs are computer-interaction features only. SWELL-KW's "
        "questionnaire columns (Stress, Valence, NASA-TLX, ...) are the "
        "participant's own ratings and are never used as features.",
        "- Labels come from the experimental condition (N / I / T), not from "
        "self-report. Rest periods (R) are dropped.",
        "- Source: Koldijk et al. (2014), DANS doi:10.17026/dans-x55-69zp, "
        "CC BY-NC-SA 4.0.",
        "",
    ]
    (out_dir / "report_swell.md").write_text("\n".join(lines))


def main(source: str = "swell", path: str | None = None) -> pd.DataFrame | None:
    out_dir = config.OUTPUT_DIR / ("synthetic/swell" if source == "synthetic" else "swell")
    if source == "synthetic":
        print("Loading SYNTHETIC SWELL-shaped data (development only) ...")
        df = loaders.load_swell(frame=loaders.load_synthetic_swell_raw())
    else:
        p = Path(path) if path else config.SWELL_PATH
        if not p.exists():
            print(f"SWELL-KW not found at {p}.\n"
                  "Request access at http://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html, "
                  "then save the computer-interaction feature file there "
                  "(or pass --path).")
            return None
        print(f"Loading SWELL-KW from {p} ...")
        df = loaders.load_swell(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    table, logo = run(df, out_dir)
    table.to_csv(out_dir / "results_swell.csv", index=False)
    logo.to_csv(out_dir / "leave_one_participant_out.csv", index=False)
    write_report(table, logo, out_dir, source)
    print("\n================ SWELL-KW RESULTS ================")
    print(table.to_string(index=False))
    print(f"\nartefacts written to {out_dir}/")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="swell", choices=["swell", "synthetic"])
    ap.add_argument("--path", default=None, help="SWELL-KW feature file")
    args = ap.parse_args()
    sys.exit(0 if main(args.source, args.path) is not None else 1)
