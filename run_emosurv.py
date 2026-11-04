"""
EmoSurv experiment: typing, text and fusion on people the models have not seen.

Fixed before any model was run:

  Primary data  free text, one sample per person and emotion. Fixed text is reported
                only as confounded, because each emotion has its own passage.
  Primary task  anger vs. calm, the closest proxy for stress.
  Other tasks   negative vs. positive (A, S vs. H, C); active vs. passive (A, H vs.
                S, C); four emotions. Neutral vs. any emotion is reported as
                confounded, because neutral always came first and followed no video.
  Primary model Logistic Regression on features minus the person's neutral block.
  Other options the same on raw features, and gradient boosting on both. Significance
                compares the best option with the best option on shuffled labels, so
                trying four options cannot make the result look stronger.
  Evaluation    5-fold, whole people held out, intervals resampling whole people.
  Text          (a) the Dreaddit model applied unchanged, with nothing fitted here;
                (b) TF-IDF fitted on training people, then early and late fusion.

    python run_emosurv.py [--perm 1000]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataio import emosurv, loaders
from features import text_features
from modeling import classifiers
from modeling import extended_eval as ev

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "emosurv"
SEED = config.RANDOM_STATE
TASKS = {
    "anger vs calm": {"A": "negative", "C": "positive"},
    "negative vs positive": {"A": "negative", "S": "negative", "H": "positive", "C": "positive"},
    "active vs passive": {"A": "active", "H": "active", "S": "passive", "C": "passive"},
    "four emotions": {"A": "anger", "H": "happiness", "C": "calmness", "S": "sadness"},
}
MIN_TEXT_WORDS = 5
REPEATS = 20   # fold assignments: with so few blocks, one assignment can mislead


def logistic():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def boosting():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.05,
                                                          class_weight="balanced", random_state=SEED))])


def oof(pipe, X, y, groups, seed=SEED):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")


def macro_f1(y, proba, classes) -> float:
    return f1_score(y, np.asarray(classes)[proba.argmax(axis=1)], average="macro", zero_division=0)


def typing_task(blocks: pd.DataFrame, task: str, mapping: dict, n_perm: int) -> dict:
    """Primary option plus the best-of-options test, for one task on one text type."""
    cols = emosurv.feature_columns(blocks)
    raw = blocks[blocks["emotion"].isin(mapping)].reset_index(drop=True)
    base = emosurv.from_baseline(blocks)
    base = base[base["emotion"].isin(mapping)].reset_index(drop=True)
    y_raw, y_base = raw["emotion"].map(mapping).to_numpy(), base["emotion"].map(mapping).to_numpy()
    options = {"baseline, logistic": (logistic, base[cols], y_base, base["participant_id"].to_numpy()),
               "raw, logistic": (logistic, raw[cols], y_raw, raw["participant_id"].to_numpy()),
               "baseline, boosting": (boosting, base[cols], y_base, base["participant_id"].to_numpy()),
               "raw, boosting": (boosting, raw[cols], y_raw, raw["participant_id"].to_numpy())}
    classes = sorted(set(y_base))

    def run(shuffle_seed=None):
        scores = {}
        for name, (make, X, y, g) in options.items():
            # Labels move across blocks: most people have one emotion block, so moving
            # them only within a person would change nothing.
            lab = y if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(y)
            scores[name] = macro_f1(lab, oof(make(), X, lab, g), classes)
        return scores

    observed = run()
    null = np.array(Parallel(n_jobs=-1)(
        delayed(lambda s: max(run(s).values()))(s) for s in np.random.SeedSequence(SEED).spawn(n_perm)))
    best = max(observed.values())

    make, X, y, g = options["baseline, logistic"]
    proba = oof(make(), X, y, g)
    m = ev.full_metrics(y, proba, classes)
    ci = ev.bootstrap_ci(y, proba, classes, g, n_boot=1000)
    repeats = [macro_f1(y, oof(make(), X, y, g, s), classes) for s in range(REPEATS)]
    return {"task": task, "f1_repeat_mean": np.mean(repeats), "f1_repeat_sd": np.std(repeats), "n_blocks": len(y), "n_people": len(set(g)), "chance_f1": 1 / len(classes),
            "primary_f1": m["f1_macro"], "f1_lo": ci["f1_lo"], "f1_hi": ci["f1_hi"],
            "roc_auc": m["roc_auc"], "auc_lo": ci["auc_lo"], "auc_hi": ci["auc_hi"],
            "accuracy": m["accuracy"], "majority_acc": m["majority_acc"],
            **{f"f1 {k}": v for k, v in observed.items()},
            "best_f1": best, "null_95th": np.percentile(null, 95),
            "p_best_of_options": (1 + (null >= best).sum()) / (n_perm + 1)}


def dreaddit_transfer(blocks: pd.DataFrame) -> pd.DataFrame:
    """The Dreaddit stress model, unchanged, scored on people's own words."""
    import train_demo
    dd = loaders.load_dreaddit()
    model, ling = train_demo.train_text(dd["text"], dd["stress_label"], [])
    b = blocks[blocks["text"].str.split().str.len() >= MIN_TEXT_WORDS].reset_index(drop=True)
    X = pd.concat([b[["text"]], text_features.build_text_features(b["text"])[ling]], axis=1)
    high = model.predict_proba(X)[:, list(model.classes_).index("High")]
    rows = []
    for task, mapping in list(TASKS.items())[:2] + [("neutral vs any emotion", None)]:
        mask = b["emotion"].isin(mapping) if mapping else np.ones(len(b), bool)
        y = (b.loc[mask, "emotion"].map(mapping) == "negative").to_numpy() if mapping else (b.loc[mask, "emotion"] != "N").to_numpy()
        g, score = b.loc[mask, "participant_id"].to_numpy(), high[mask]
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(1000):
            people = rng.choice(np.unique(g), len(np.unique(g)))
            idx = np.concatenate([np.flatnonzero(g == p) for p in people])
            if len(set(y[idx])) == 2:
                boots.append(roc_auc_score(y[idx], score[idx]))
        rows.append({"task": task, "n": int(mask.sum()), "auc": roc_auc_score(y, score),
                     "auc_lo": np.percentile(boots, 2.5), "auc_hi": np.percentile(boots, 97.5),
                     "mean_stress_prob_positive_class": score[y].mean(),
                     "mean_stress_prob_other_class": score[~y].mean()})
    return pd.DataFrame(rows)


def fusion_task(blocks: pd.DataFrame, task: str, mapping: dict, n_perm: int) -> tuple[dict, dict]:
    """Typing minus baseline, text, and both, on the same samples."""
    cols = emosurv.feature_columns(blocks)
    text_ok = blocks[blocks["text"].str.split().str.len() >= MIN_TEXT_WORDS]
    base = emosurv.from_baseline(blocks)
    base = base[base["emotion"].isin(mapping) & base.set_index(["participant_id", "emotion"]).index.isin(
        text_ok.set_index(["participant_id", "emotion"]).index)].reset_index(drop=True)
    y, g = base["emotion"].map(mapping).to_numpy(), base["participant_id"].to_numpy()
    classes = sorted(set(y))
    tfidf = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    text = Pipeline([("v", tfidf()), ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    early = Pipeline([("x", ColumnTransformer([
        ("t", tfidf(), "text"),
        ("k", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols)])),
        ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    p_typ = oof(logistic(), base[cols], y, g)
    p_txt = oof(text, base["text"], y, g)
    p_early = oof(early, base[["text"] + cols], y, g)
    p_late = (p_typ + p_txt) / 2
    names = {"typing": p_typ, "text": p_txt, "early fusion": p_early, "late fusion": p_late}
    row = {"task": task, "n_blocks": len(y), "chance_f1": 1 / len(classes),
           **{k: macro_f1(y, p, classes) for k, p in names.items()}}
    stable = {k: [] for k in names}
    for s in range(REPEATS):
        a, b, c = oof(logistic(), base[cols], y, g, s), oof(text, base["text"], y, g, s), \
            oof(early, base[["text"] + cols], y, g, s)
        for k, p in zip(names, (a, b, c, (a + b) / 2)):
            stable[k].append(macro_f1(y, p, classes))
    row.update({f"{k} repeat mean": np.mean(v) for k, v in stable.items()})
    row.update({f"{k} repeat sd": np.std(v) for k, v in stable.items()})
    shuffled = lambda s: (lambda lab: macro_f1(lab, oof(text, base["text"], lab, g), classes))(
        np.random.default_rng(s).permutation(y))
    null = np.array(Parallel(n_jobs=-1)(delayed(shuffled)(s) for s in np.random.SeedSequence(SEED).spawn(n_perm)))
    row["p_text"] = (1 + (null >= row["text"]).sum()) / (n_perm + 1)
    pred = lambda p: np.asarray(classes)[p.argmax(axis=1)]
    d = ev.bootstrap_f1_difference(y, pred(p_late),
                                   pred(p_typ if row["typing"] >= row["text"] else p_txt), g)
    gain = {"task": task, "late minus best single": d["difference"], "lo": d["lo"], "hi": d["hi"]}
    return row, gain


def main(n_perm: int = 1000) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    free, fixed = emosurv.load_blocks("free"), emosurv.load_blocks("fixed")
    print(f"free: {len(free)} blocks, {free.participant_id.nunique()} people, {free.emotion.value_counts().to_dict()}")
    print(f"fixed: {len(fixed)} blocks, {fixed.participant_id.nunique()} people")

    typing_free = pd.DataFrame([typing_task(free, t, m, n_perm) for t, m in TASKS.items()])
    typing_free["p_holm"] = ev.holm_adjust(typing_free["p_best_of_options"])
    typing_fixed = pd.DataFrame([typing_task(fixed, t, m, min(n_perm, 200)) for t, m in TASKS.items()])

    # neutral vs. emotion cannot use the baseline: neutral rows would all be zero
    cols = emosurv.feature_columns(free)
    y = np.where(free["emotion"] == "N", "neutral", "emotion")
    p = oof(logistic(), free[cols], y, free["participant_id"].to_numpy())
    neutral = ev.full_metrics(y, p, sorted(set(y)))

    transfer = dreaddit_transfer(free)
    pairs = [fusion_task(free, t, m, n_perm) for t, m in TASKS.items()]
    fusion, gains = pd.DataFrame([a for a, _ in pairs]), pd.DataFrame([b for _, b in pairs])
    fusion["p_text_holm"] = ev.holm_adjust(fusion["p_text"])

    for name, frame in [("typing_free", typing_free), ("typing_fixed_confounded", typing_fixed),
                        ("dreaddit_transfer", transfer), ("fusion", fusion), ("fusion_gain", gains)]:
        frame.round(4).to_csv(OUT / f"{name}.csv", index=False)

    show = ["task", "n_blocks", "n_people", "chance_f1", "primary_f1", "f1_lo", "f1_hi",
            "f1_repeat_mean", "f1_repeat_sd", "roc_auc",
            "accuracy", "majority_acc", "best_f1", "p_best_of_options"]
    lines = ["# EmoSurv: typing, text and fusion on unseen people", "",
             "Labels are emotions, not stress. Whole people held out. Design fixed in `run_emosurv.py`.", "",
             "## Typing, free text (primary)", "", typing_free[show + ["p_holm"]].round(3).to_markdown(index=False), "",
             "## Typing, fixed text (confounded: each emotion has its own passage)", "",
             typing_fixed[show].round(3).to_markdown(index=False), "",
             f"## Neutral vs. any emotion (confounded with order)\n\nmacro F1 {neutral['f1_macro']:.3f}, "
             f"ROC-AUC {neutral['roc_auc']:.3f}, n = {neutral['n']}.", "",
             "## Dreaddit stress model applied unchanged", "", transfer.round(3).to_markdown(index=False), "",
             "## Fusion, same samples", "", fusion.round(3).to_markdown(index=False), "",
             gains.round(3).to_markdown(index=False), ""]
    (OUT / "report_emosurv.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return typing_free


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=1000)
    main(ap.parse_args().perm)
