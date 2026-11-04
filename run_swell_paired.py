"""
Paired experiment on real public data: SWELL-KW raw keystroke logs.

The same comparison as run_paired.py (typing only, text only, early fusion,
late fusion, stacking) on 5-minute windows of Word typing from SWELL-KW, with
whole participants held out. Both modalities come from one keystroke stream, so
every window has timing, words and one condition label. This is what makes
fusion testable before, or without, our own data collection.

    python -m dataio.swell_raw      # once: download the logs (~400 MB)
    python run_swell_paired.py

Tasks:
    n_vs_t         neutral vs. time pressure
    n_vs_stressor  neutral vs. either stressor
    i_vs_t         interruption vs. time pressure (order counterbalanced)
    3class         neutral / interruption / time pressure
    order_control  block 2 vs. block 3 regardless of condition. Not a stress
                   task: it measures how much a model can tell from position in
                   the session alone. Neutral is ALWAYS block 1, so any
                   neutral-vs-stressor result may partly be this effect.

Writes to outputs/paired_swell/.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
import run_paired
from dataio import swell_raw

warnings.filterwarnings("ignore")

OUT = config.OUTPUT_DIR / "paired_swell"


def targets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c = df["condition"]
    return {
        "n_vs_t": df[c.isin(["N", "T"])],
        "n_vs_stressor": df.assign(stress_label=np.where(c == "N", "NoStress", "Stress")),
        "i_vs_t": df[c.isin(["I", "T"])],
        "3class": df,
        "order_control": df[df["block"].isin([2, 3])].assign(
            stress_label=lambda d: "Block" + d["block"].astype(str)),
    }


def manipulation_check() -> tuple[str, dict]:
    """Did the conditions actually feel more stressful? The participants' own ratings."""
    q = swell_raw.load_questionnaire()
    means = q.groupby("Condition")[["Stress", "TemporalDemand"]].mean().round(2)
    pvals = {}
    for rating in ["Stress", "TemporalDemand"]:
        wide = q.pivot_table(index="PP", columns="Condition", values=rating)
        for a, b in [("N", "T"), ("N", "I"), ("I", "T")]:
            pair = wide[[a, b]].dropna()
            pvals[f"{rating} {a} vs {b}"] = float(wilcoxon(pair[a], pair[b]).pvalue)
    topics = pd.crosstab(pd.concat([q["Topic1"], q["Topic2"]]).values,
                         pd.concat([q["Condition"], q["Condition"]]).values)
    text = "\n".join([
        means.to_markdown(), "",
        "Paired Wilcoxon on the self-ratings: "
        + ", ".join(f"{k} p={v:.3f}" for k, v in pvals.items()) + ".", "",
        "Topic by condition (a topic tied to a condition would make word features "
        "detect the topic):", "", topics.to_markdown(),
    ])
    return text, {"means": means, "pvals": pvals}


def main(weight: float = 0.5) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df = swell_raw.load_paired()
    tables = []
    for name, sub in targets(df).items():
        print(f"\n########## {name} ##########")
        out_dir = OUT / name
        out_dir.mkdir(exist_ok=True)
        t = run_paired.run_experiment(sub.reset_index(drop=True), weight, out_dir)
        t = t[t["mode"] != "typing_only_all_tasks"].assign(task=name)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / "results_paired_swell.csv", index=False)

    check, _ = manipulation_check()
    best = table.groupby(["task", "mode"])["f1_macro"].max().unstack(0).round(3)
    lines = [
        "# Paired Experiment on SWELL-KW Raw Logs", "",
        f"{df['participant_id'].nunique()} English-writing participants, {len(df)} "
        "5-minute windows of Word typing, whole participants held out.", "",
        "Windows per condition: "
        + ", ".join(f"{k} {v}" for k, v in df["condition"].value_counts().items()) + ".", "",
        "## Manipulation check", "", check, "",
        "## Best macro F1 per mode and task (best of LR / RF / SVM)", "",
        best.to_markdown(), "",
        "Choosing the best of three on the same folds is slightly optimistic: "
        "`run_evaluation.py` tests a fixed model for significance.", "",
        "## How to read this", "",
        "- Neutral is always block 1. `order_control` shows how much position in "
        "the session alone can be detected, so neutral-vs-stressor scores are an "
        "upper limit on the stress signal.",
        "- `i_vs_t` is counterbalanced but compares two stressors, not stress vs. none.",
        "- No dwell time: uLog records key entry only.",
        "- Labels are conditions. The manipulation check shows whether they match "
        "how participants felt.", "",
        "## All results", "", table.to_markdown(index=False), "",
    ]
    (OUT / "report_paired_swell.md").write_text("\n".join(lines))
    print("\n" + best.to_string())
    print(f"\nartefacts written to {OUT}/")
    return table


if __name__ == "__main__":
    main()
