"""
Extended evaluation of every model on every dataset.

    python run_evaluation.py            # full run, up to about an hour
    python run_evaluation.py --quick    # fewer repeats, to check it works
    python run_evaluation.py --source synthetic   # numbers are fake

Adds to the accuracy / precision / recall / F1 / confusion matrices reported
elsewhere: balanced accuracy, MCC, Cohen's kappa, ROC-AUC, PR-AUC, log loss,
Brier score, calibration, per-class results, 95% bootstrap confidence intervals,
permutation significance tests (Holm corrected), McNemar tests, repeated
cross-validation, learning curves by number of people, per-person spread and
real-time prediction latency.

To avoid choosing the winner on the test folds, significance and stability use
one model fixed in advance: Logistic Regression.

Writes to outputs/evaluation/ (outputs/synthetic/evaluation/ for synthetic data).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
import run_paired
from dataio import loaders, swell_raw
from features import interaction_features, text_features, typing_features
from modeling import classifiers, extended_eval as ev, fusion

warnings.filterwarnings("ignore")

MODELS = ["LogisticRegression", "RandomForest", "SVM"]
LR = "LogisticRegression"


@dataclass
class Variant:
    name: str
    X: pd.DataFrame
    build: object            # estimator -> pipeline


@dataclass
class Task:
    dataset: str
    task: str
    y: pd.Series
    variants: list
    groups: np.ndarray | None = None
    blocks: np.ndarray | None = None
    held_out: str = "random folds"
    significance: bool = True
    fusion: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)


def _typing_variant(name, X):
    return Variant(name, X.reset_index(drop=True), classifiers.build_typing_pipeline)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def dreaddit_tasks() -> tuple[list[Task], pd.DataFrame]:
    df = loaders.load_dreaddit()
    ling = text_features.build_text_features(df["text"])
    X = pd.concat([df[["text"]].reset_index(drop=True), ling], axis=1)
    cols = list(ling.columns)
    y = df["stress_label"].astype(str).reset_index(drop=True)
    variant = Variant("tfidf+linguistic", X, lambda e: classifiers.build_text_pipeline(e, cols))
    posts = df["post_id"].to_numpy()
    tasks = [
        Task("Dreaddit", "random folds (posts can be split)", y, [variant], significance=False,
             note="Segments of one post can sit in both train and test."),
        Task("Dreaddit", "whole posts held out", y, [variant], groups=posts,
             held_out="posts held out"),
    ]
    return tasks, X.assign(_y=y)


def kaggle_task() -> Task:
    events = loaders.load_kaggle_keystrokes()
    base = typing_features.features_from_events(events)
    merged = interaction_features.merge_with_typing(
        base, interaction_features.load_interaction_features())
    X, y = typing_features.prepare_feature_frame(merged)
    return Task("Kaggle keystroke", "3-class", y.reset_index(drop=True),
                [_typing_variant("keystroke+interaction", X)], significance=False,
                note="Two people and neighbouring windows in train and test, so these "
                     "scores are optimistic. No significance test: two people cannot "
                     "support one. See the cross-user test.")


def swell_minute_tasks() -> list[Task]:
    df = loaders.load_swell()
    raw = pd.read_csv(config.SWELL_PATH)
    raw = raw[raw["Condition"].astype(str).str.lower().map(config.SWELL_CONDITION_MAP).notna()]
    if len(raw) != len(df):
        raise ValueError("SWELL rows and block column do not align.")
    block = raw["Blok"].astype(int).to_numpy()
    people = df["participant"].astype(str).to_numpy()
    X, y3 = typing_features.prepare_feature_frame(df)
    X, y3 = X.reset_index(drop=True), y3.reset_index(drop=True)
    kb = [c for c in ["SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys", "SnErrorKeys",
                      "SnShortcutKeys", "SnSpaces"] if c in X.columns]

    def make(task, mask, y, keyboard=False, note=""):
        vs = [_typing_variant("all interaction", X[mask])]
        if keyboard:
            vs.append(_typing_variant("keyboard only", X.loc[mask, kb]))
        return Task("SWELL-KW minutes", task, y[mask].reset_index(drop=True), vs,
                    groups=people[mask], blocks=block[mask], held_out="people held out", note=note)

    every = np.ones(len(y3), bool)
    nt, it = y3.isin(["Low", "High"]).to_numpy(), y3.isin(["Medium", "High"]).to_numpy()
    b23 = np.isin(block, [2, 3])
    return [
        make("neutral vs time pressure", nt, y3, keyboard=True),
        make("interruption vs time pressure", it, y3),
        make("3-class", every, y3),
        make("order control: block 2 vs 3", b23,
             pd.Series(["Block" + str(b) for b in block]),
             note="Not stress: how much position in the session alone can be detected."),
    ]


def _paired_task(dataset, task, df, note="") -> Task:
    df = df.reset_index(drop=True)
    X, ling, typing = run_paired.build_matrix(df)
    text_cols = ["text"] + ling
    blocks = df["session_id"].astype(str).str.replace(r"_(W|T)\d+$", "", regex=True).to_numpy()
    return Task(dataset, task, df["stress_label"].astype(str),
                [Variant("typing only", X[typing], classifiers.build_typing_pipeline),
                 Variant("text only", X[text_cols],
                         lambda e, c=ling: classifiers.build_text_pipeline(e, c)),
                 Variant("early fusion", X,
                         lambda e, c=ling, t=typing: classifiers.build_early_fusion_pipeline(e, c, t))],
                groups=df["participant_id"].astype(str).to_numpy(), blocks=blocks,
                held_out="people held out", fusion=True, note=note)


def swell_paired_tasks() -> list[Task]:
    import run_swell_paired

    t = run_swell_paired.targets(swell_raw.load_paired())
    names = {"n_vs_t": "neutral vs time pressure", "n_vs_stressor": "neutral vs either stressor",
             "i_vs_t": "interruption vs time pressure", "3class": "3-class",
             "order_control": "order control: block 2 vs 3"}
    return [_paired_task("SWELL-KW paired", names[k], v) for k, v in t.items()]


def own_tasks(source: str) -> list[Task]:
    df = (loaders.load_synthetic_paired_frame() if source == "synthetic"
          else loaders.load_own_paired())
    df = df[df["task_type"].isin(config.OWN_TEXT_TASKS) & df["text"].str.strip().astype(bool)]
    return [_paired_task("Own collected" if source != "synthetic" else "SYNTHETIC", "3-class", df)]


# --------------------------------------------------------------------------
# running one task
# --------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def evaluate_task(t: Task, cfg: dict, out: Path, store: dict) -> None:
    classes = sorted(t.y.unique())
    y_idx = t.y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    tag = _slug(f"{t.dataset}_{t.task}")
    print(f"\n[{t.dataset}] {t.task}  n={len(t.y)}  classes={classes}", flush=True)

    probas = {}
    for v in t.variants:
        curves = {}
        for name in MODELS:
            est = classifiers.get_estimators()[name]
            proba = ev.oof_proba(v.build(est), v.X, t.y, t.groups)
            probas[(v.name, name)] = proba
            row = {"dataset": t.dataset, "task": t.task, "evaluation": t.held_out,
                   "variant": v.name, "model": name}
            row.update(ev.full_metrics(t.y, proba, classes))
            row.update(ev.bootstrap_ci(t.y, proba, classes, t.groups, cfg["boot"]))
            store["metrics"].append(row)
            pc = ev.per_class_table(t.y, proba, classes)
            for k, val in (("dataset", t.dataset), ("task", t.task), ("variant", v.name), ("model", name)):
                pc.insert(0, k, val)
            store["per_class"].append(pc)
            if len(classes) == 2:
                curves[name] = (y_idx, proba[:, 1])
            print(f"  {v.name:22s} {name:20s} f1={row['f1_macro']:.3f} "
                  f"[{row['f1_lo']:.3f},{row['f1_hi']:.3f}] auc={row['roc_auc']:.3f}", flush=True)

        vtag = f"{tag}_{_slug(v.name)}"
        title = f"{t.dataset}: {t.task}"
        if curves:
            ev.plot_roc(curves, f"{title}\n{v.name}", out / f"roc_{vtag}.png")
        lr = probas[(v.name, LR)]
        ev.plot_calibration(y_idx, lr, f"Calibration, {title}\n{v.name}", out / f"calibration_{vtag}.png")
        ev.plot_confusion_normalised(y_idx, lr, classes, f"{t.task}\n{v.name}, LR",
                                     out / f"confusion_{vtag}.png")

        pipe = v.build(classifiers.get_estimators()[LR])
        stab = ev.repeated_cv(pipe, v.X, t.y, t.groups, seeds=range(cfg["seeds"]))
        stab.update({"dataset": t.dataset, "task": t.task, "variant": v.name})
        store["stability"].append(stab)
        if t.significance:
            # A p-value cannot go below 1 / (shuffles + 1), so the shuffle count must
            # leave room for the Holm correction. Dreaddit is slow and needs little.
            n_perm = cfg["perm_large"] if t.dataset == "Dreaddit" else cfg["perm"]
            res = ev.permutation_test(pipe, v.X, t.y, t.groups, t.blocks, n_perm)
            res.update({"dataset": t.dataset, "task": t.task, "variant": v.name})
            store["significance"].append(res)
            print(f"  {v.name:22s} permutation p={res['p_value']:.4f}", flush=True)
        if t.groups is not None and "people" in t.held_out:
            pp = ev.per_person_accuracy(t.y, lr, classes, t.groups)
            store["people"].append({"dataset": t.dataset, "task": t.task, "variant": v.name,
                                    "people": len(pp), "min": pp["accuracy"].min(),
                                    "median": pp["accuracy"].median(), "max": pp["accuracy"].max(),
                                    "share_above_majority":
                                        (pp["accuracy"] > pd.Series(y_idx).value_counts(normalize=True).max()).mean()})

    if t.fusion:
        p_typ, p_txt = probas[("typing only", LR)], probas[("text only", LR)]
        late_lbl, late_p = fusion.weighted_average_fusion(p_typ, p_txt, 0.5, target_classes=classes)
        lab = lambda p: np.array(classes)[p.argmax(axis=1)]
        row = {"dataset": t.dataset, "task": t.task, "evaluation": t.held_out,
               "variant": "late fusion (0.5)", "model": LR}
        row.update(ev.full_metrics(t.y, late_p, classes))
        row.update(ev.bootstrap_ci(t.y, late_p, classes, t.groups, cfg["boot"]))
        store["metrics"].append(row)
        early = lab(probas[("early fusion", LR)])
        for fused, pred in [("early fusion", early), ("late fusion", late_lbl)]:
            for single, p in [("typing only", p_typ), ("text only", p_txt)]:
                d = ev.bootstrap_f1_difference(t.y, pred, lab(p), t.groups, cfg["boot"] * 2)
                d.update({"dataset": t.dataset, "task": t.task, "comparison": f"{fused} minus {single}"})
                store["fusion"].append(d)


def official_dreaddit(store: dict, cfg: dict, out: Path, X: pd.DataFrame) -> None:
    """The dataset authors' own split, where no post is in both sides."""
    n_train = len(pd.read_csv(config.DREADDIT_TRAIN_PATH))
    y = X["_y"].to_numpy()
    F = X.drop(columns=["_y"])
    cols = [c for c in F.columns if c != "text"]
    preds, classes = {}, sorted(set(y))
    for name in MODELS:
        pipe = classifiers.build_text_pipeline(classifiers.get_estimators()[name], cols)
        pipe.fit(F.iloc[:n_train], y[:n_train])
        proba = pipe.predict_proba(F.iloc[n_train:])
        preds[name] = np.array(classes)[proba.argmax(axis=1)]
        row = {"dataset": "Dreaddit", "task": "official train/test split", "evaluation": "official split",
               "variant": "tfidf+linguistic", "model": name}
        row.update(ev.full_metrics(y[n_train:], proba, classes))
        row.update(ev.bootstrap_ci(y[n_train:], proba, classes, None, cfg["boot"]))
        store["metrics"].append(row)
        print(f"  official split {name:20s} f1={row['f1_macro']:.3f} acc={row['accuracy']:.3f}", flush=True)
    for a, b in [(LR, "RandomForest"), (LR, "SVM"), ("RandomForest", "SVM")]:
        m = ev.mcnemar(y[n_train:], preds[a], preds[b])
        m.update({"pair": f"{a} vs {b}"})
        store["mcnemar"].append(m)


def learning_curves(tasks: list[Task], store: dict, out: Path, cfg: dict) -> None:
    wanted = [("SWELL-KW minutes", "neutral vs time pressure", "all interaction"),
              ("SWELL-KW paired", "neutral vs time pressure", "typing only"),
              ("SWELL-KW paired", "neutral vs time pressure", "early fusion"),
              ("Own collected", "3-class", "early fusion"),
              ("SYNTHETIC", "3-class", "early fusion")]
    for t in tasks:
        for v in t.variants:
            if (t.dataset, t.task, v.name) not in wanted:
                continue
            pipe = v.build(classifiers.get_estimators()[LR])
            table = ev.learning_curve_people(pipe, v.X, t.y, t.groups, [3, 5, 10, 15, 20],
                                             repeats=cfg["curve"])
            table.insert(0, "curve", f"{t.dataset}: {v.name}")
            store["learning"].append(table)
            ev.plot_learning_curve(table, 1 / t.y.nunique(),
                                   f"Learning curve, {t.dataset}\n{t.task}, {v.name}",
                                   out / f"learning_{_slug(t.dataset + '_' + v.name)}.png")


def measure_latency(store: dict) -> None:
    """Time real requests to the demo server, as the extension makes them."""
    sys.path.insert(0, str(config.PROJECT_ROOT / "app"))
    import server

    if not server.MODELS.ready:
        return
    client = server.app.test_client()
    text = ("I have several reports due this week and I am trying to keep up with them while "
            "answering messages from my team and planning the next few days of work ") * 2
    events, t = [], 0.0
    for ch in text:
        t += 150
        code = "Space" if ch == " " else f"Key{ch.upper()}"
        events += [{"ev": "down", "key": ch, "code": code, "t": t},
                   {"ev": "up", "key": ch, "code": code, "t": t + 90}]
    times = []
    for _ in range(100):
        start = time.perf_counter()
        client.post("/api/predict", json={"text": text, "events": events})
        times.append((time.perf_counter() - start) * 1000)
    store["latency"] = {"keys_per_request": len(text), "median_ms": float(np.median(times)),
                        "p95_ms": float(np.percentile(times, 95))}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(store: dict, out: Path, source: str) -> None:
    m = pd.DataFrame(store["metrics"]).round(3)
    m.to_csv(out / "metrics_all.csv", index=False)
    pd.concat(store["per_class"]).round(3).to_csv(out / "per_class.csv", index=False)
    stab = pd.DataFrame(store["stability"]).round(3)
    stab.to_csv(out / "stability_repeated_cv.csv", index=False)

    sig = pd.DataFrame(store["significance"])
    if not sig.empty:
        # Holm within each dataset. Order controls are not stress hypotheses, so they
        # are reported uncorrected: a correction would only make them easier to pass.
        sig["control"] = sig["task"].str.startswith("order control")
        sig["p_holm"] = sig["p_value"]
        for _, idx in sig[~sig["control"]].groupby("dataset").groups.items():
            sig.loc[idx, "p_holm"] = ev.holm_adjust(sig.loc[idx, "p_value"])
        sig["significant"] = sig["p_holm"] < 0.05
        sig = sig[["dataset", "task", "variant", "observed_f1", "null_mean_f1", "null_95th",
                   "p_value", "p_holm", "significant", "control", "n_perm"]].round(4)
        sig.to_csv(out / "significance_permutation.csv", index=False)

    lines = ["# Extended Evaluation", ""]
    if source == "synthetic":
        lines += ["> **SYNTHETIC DATA. Never quote these numbers.**", ""]
    lines += [
        "## Headline: macro F1 with 95% confidence interval", "",
        "Logistic Regression, fixed in advance so no winner is picked on the test folds.", "",
    ]
    h = m[m["model"] == LR].copy()
    h["macro F1 [95% CI]"] = h.apply(lambda r: f"{r.f1_macro:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]", axis=1)
    h["ROC-AUC [95% CI]"] = h.apply(lambda r: f"{r.roc_auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]", axis=1)
    h = h.rename(columns={"chance_f1": "chance F1", "accuracy": "acc", "majority_acc": "majority acc"})
    lines += [h[["dataset", "task", "variant", "macro F1 [95% CI]", "chance F1", "ROC-AUC [95% CI]",
                 "acc", "majority acc", "n"]].to_markdown(index=False), ""]

    if not sig.empty:
        lines += ["## Significance: better than shuffled labels?", "",
                  f"Each model is re-run on up to {int(sig['n_perm'].max())} shuffled labellings "
                  "(see `n_perm`). On SWELL and own data, labels move between a person's blocks "
                  "as whole units. `p_holm` corrects for the number of stress tests within each "
                  "dataset; `significant` uses it at 0.05. Rows with `control` true are order "
                  "checks, which should NOT be significant.", "",
                  sig.drop(columns="n_perm").to_markdown(index=False), ""]
    if store["fusion"]:
        f = pd.DataFrame(store["fusion"]).round(3)
        f["excludes zero"] = (f["lo"] > 0) | (f["hi"] < 0)
        f.to_csv(out / "fusion_gain.csv", index=False)
        lines += ["## Does fusion help?", "",
                  "Macro F1 difference on the same samples (Logistic Regression), whole people "
                  "resampled. If the interval includes zero, the data does not show a difference.", "",
                  f[["dataset", "task", "comparison", "difference", "lo", "hi", "excludes zero"]]
                  .to_markdown(index=False), ""]
    if store["mcnemar"]:
        lines += ["## McNemar tests (Dreaddit official test set)", "",
                  "Do two models differ on the samples where they disagree?", "",
                  pd.DataFrame(store["mcnemar"]).round(4)[
                      ["pair", "only_a_correct", "only_b_correct", "p_value"]].to_markdown(index=False), ""]
    lines += ["## Stability: repeated cross-validation", "",
              f"Logistic Regression, {int(stab['repeats'].iloc[0])} different fold assignments.", "",
              stab[["dataset", "task", "variant", "f1_mean", "f1_sd", "f1_min", "f1_max"]]
              .to_markdown(index=False), ""]
    if store["people"]:
        pp = pd.DataFrame(store["people"]).round(3)
        pp.to_csv(out / "per_person.csv", index=False)
        lines += ["## Spread across people (accuracy per held-out person, LR)", "",
                  pp.to_markdown(index=False), ""]
    if store["learning"]:
        lc = pd.concat(store["learning"]).round(3)
        lc.to_csv(out / "learning_curves.csv", index=False)
        lines += ["## Learning curves: would more people help?", "",
                  "Macro F1 on 5 unseen people as training people increase (mean, SD over repeats).", "",
                  lc.to_markdown(index=False), ""]
    if store["latency"]:
        l = store["latency"]
        lines += ["## Real-time latency", "",
                  f"Server time for one prediction on a {l['keys_per_request']}-key burst: median "
                  f"{l['median_ms']:.0f} ms, 95th percentile {l['p95_ms']:.0f} ms (this computer).", ""]
    lines += ["## All models, all metrics", "",
              "Per-class results are in `per_class.csv`. Charts: `roc_*`, `calibration_*`, "
              "`confusion_*`, `learning_*`.", "",
              m.drop(columns=["acc_lo", "acc_hi"]).to_markdown(index=False), "",
              "## What the metrics mean", "",
              "- **Balanced accuracy**: mean recall per class; ignores class size.",
              "- **MCC / Cohen's kappa**: agreement beyond chance; 0 is chance, 1 is perfect.",
              "- **ROC-AUC**: chance a random positive is ranked above a random negative; 0.5 is chance.",
              "- **PR-AUC**: precision against recall across thresholds, averaged per class.",
              "- **Log loss / Brier**: quality of the probabilities; lower is better.",
              "- **Calibration error**: gap between stated confidence and actual accuracy; lower is better.",
              "- **Bootstrap CI**: range the score plausibly takes on a different sample, resampling "
              "whole people where they are identifiable.",
              "- **Permutation test**: how often shuffled labels score as well as the real ones.",
              ""]
    (out / "report_evaluation.md").write_text("\n".join(lines))


def main(source: str = "auto", quick: bool = False) -> None:
    cfg = (dict(boot=100, perm=20, perm_large=20, seeds=2, curve=5) if quick
           else dict(boot=500, perm=2000, perm_large=200, seeds=5, curve=30))
    out = config.OUTPUT_DIR / ("synthetic/evaluation" if source == "synthetic" else "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    store = dict(metrics=[], per_class=[], stability=[], significance=[], people=[], fusion=[],
                 mcnemar=[], learning=[], latency=None)

    tasks: list[Task] = []
    dreaddit_X = None
    if source == "synthetic":
        tasks += own_tasks("synthetic")
    else:
        d, dreaddit_X = dreaddit_tasks()
        tasks += d
        tasks.append(kaggle_task())
        if config.SWELL_PATH.exists():
            tasks += swell_minute_tasks()
        if swell_raw.ULOG_DIR.exists() and any(swell_raw.ULOG_DIR.glob("*.xml")):
            tasks += swell_paired_tasks()
        if loaders.own_data_available():
            tasks += own_tasks("own")

    for t in tasks:
        evaluate_task(t, cfg, out, store)
    if dreaddit_X is not None:
        official_dreaddit(store, cfg, out, dreaddit_X)
    learning_curves(tasks, store, out, cfg)
    if source != "synthetic":
        measure_latency(store)
    write_report(store, out, source)
    print(f"\nartefacts written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic"])
    ap.add_argument("--quick", action="store_true", help="fewer repeats, to check it works")
    args = ap.parse_args()
    main(args.source, args.quick)
