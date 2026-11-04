"""
Text ensemble on Dreaddit's official split. Equal-weight average of DistilBERT, RoBERTa
and the interpretable classical model, declared as the headline before scoring, so no
weights or members were chosen on the test set. Needs both run_transformer.py runs:

    python run_transformer.py --split official
    python run_transformer.py --split official --model roberta-base
    python run_text_ensemble.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import train_demo
from dataio import loaders
from features import text_features
from modeling import extended_eval as ev

TF = config.OUTPUT_DIR / "transformer"
MEMBERS = {"DistilBERT": "test_predictions_official.csv", "RoBERTa": "test_predictions_official_roberta-base.csv"}
CLASSES = ["High", "Low"]


def classical_probabilities() -> tuple[np.ndarray, pd.DataFrame]:
    df = loaders.load_dreaddit()
    n_train = len(pd.read_csv(config.DREADDIT_TRAIN_PATH))
    tr, te = df.iloc[:n_train], df.iloc[n_train:].reset_index(drop=True)
    model, ling = train_demo.train_text(tr["text"], tr["stress_label"], [])
    X = pd.concat([te[["text"]], text_features.build_text_features(te["text"])[ling]], axis=1)
    return model.predict_proba(X)[:, list(model.classes_).index("High")], te


def main() -> pd.DataFrame:
    classical, te = classical_probabilities()
    members = {"Classical (interpretable)": classical}
    for name, file in MEMBERS.items():
        p = pd.read_csv(TF / file)
        assert (p["y"].to_numpy() == te["stress_label"].astype(str).to_numpy()).all(), "test rows differ"
        members[name] = p["proba_high"].to_numpy()
    members["Ensemble (declared headline)"] = np.mean(list(members.values()), axis=0)

    y = te["stress_label"].astype(str).to_numpy()
    groups = te["post_id"].to_numpy()
    both = lambda high: np.column_stack([high, 1 - high])
    rows = []
    for name, high in members.items():
        m = ev.full_metrics(y, both(high), CLASSES)
        ci = ev.bootstrap_ci(y, both(high), CLASSES, groups, n_boot=2000)
        rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["f1_macro"], "f1_lo": ci["f1_lo"],
                     "f1_hi": ci["f1_hi"], "roc_auc": m["roc_auc"], "calibration_error": m["calibration_error"]})
    table = pd.DataFrame(rows).round(4)
    label = lambda high: np.where(high >= 0.5, "High", "Low")
    head = label(members["Ensemble (declared headline)"])
    tests = pd.DataFrame([{"ensemble minus": n, **ev.bootstrap_f1_difference(y, head, label(h), groups, 2000),
                           "mcnemar_p": ev.mcnemar(y, head, label(h)).get("p_value")}
                          for n, h in members.items() if not n.startswith("Ensemble")]).round(4)
    table.to_csv(TF / "ensemble_official.csv", index=False)
    tests.to_csv(TF / "ensemble_tests_official.csv", index=False)
    (TF / "report_ensemble_official.md").write_text(
        "# Text models, official Dreaddit split\n\n" + table.to_markdown(index=False)
        + "\n\n" + tests.to_markdown(index=False) + "\n")
    print(table.to_string(index=False), "\n\n", tests.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
