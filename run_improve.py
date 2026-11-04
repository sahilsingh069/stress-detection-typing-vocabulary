"""
Can the models honestly be made stronger?

Every option is listed here before running, and all results are reported. Trying
several options and keeping the best inflates scores, so significance compares
the BEST option on real labels with the BEST option on shuffled labels.

Typing, SWELL-KW raw logs, whole people held out:
    base 8        the current features, Logistic Regression
    rich          gap distribution, bursts, pauses, corrections (rich_features)
    rich+person   the same, z-scored within each person (no labels used)
    each with Logistic Regression and gradient boosting
Each is scored per 5-minute window and per block (one person's whole condition,
averaging its windows), since the dashboard reports over longer periods.

Text, Dreaddit official split: options are chosen by cross-validation on the
official training set only, then scored once on the official test set.

    python run_improve.py [--perm 500]
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataio import loaders, swell_raw
from features import typing_features
from modeling import extended_eval as ev

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "improve"
SEED = config.RANDOM_STATE


# --------------------------------------------------------------------------
# typing
# --------------------------------------------------------------------------
def typing_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = swell_raw.load_paired()
    keys = swell_raw.windowed_word_keys()
    keys = keys[keys["session_id"].isin(paired["session_id"])]
    rich = pd.DataFrame([
        {"session_id": sid, **typing_features.rich_features(g["timestamp"], g["key"])}
        for sid, g in keys.groupby("session_id")
    ]).set_index("session_id").loc[paired["session_id"]].reset_index(drop=True)
    base_cols = [c for c in config.OWN_TYPING_FEATURES if c in paired.columns]
    return paired, paired[base_cols].reset_index(drop=True), rich


def by_person(X: pd.DataFrame, people) -> pd.DataFrame:
    filled = X.fillna(X.median())
    grouped = filled.groupby(np.asarray(people))
    return ((filled - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)).fillna(0)


def logistic():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def boosting():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05,
                                                          class_weight="balanced", random_state=SEED))])


def oof(pipe, X, y, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")


def block_score(y, proba, blocks):
    """macro F1 after averaging each block's window probabilities."""
    d = pd.DataFrame(proba)
    d["b"], d["y"] = blocks, y
    g = d.groupby("b")
    p = g[list(range(proba.shape[1]))].mean().to_numpy()
    truth = g["y"].first().to_numpy()
    return f1_score(truth, p.argmax(axis=1), average="macro", zero_division=0), truth, p


def run_options(options, y, groups, blocks):
    """(window scores, block scores) per option name for one labelling."""
    win, blk = {}, {}
    for name, (make, X) in options.items():
        p = oof(make(), X, y, groups)
        win[name] = f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)
        blk[name] = block_score(y, p, blocks)[0]
    return win, blk


def typing_experiment(n_perm: int) -> pd.DataFrame:
    paired, base, rich = typing_frames()
    people = paired["participant_id"].to_numpy()
    blocks = (paired["participant_id"] + "_" + paired["block"].astype(str)).to_numpy()
    c = paired["condition"].to_numpy()
    tasks = {"neutral vs time pressure": (np.isin(c, ["N", "T"]), np.where(c == "N", 0, 1)),
             "neutral vs either stressor": (np.ones(len(c), bool), np.where(c == "N", 0, 1)),
             "3-class": (np.ones(len(c), bool), pd.Series(c).map({"N": 0, "I": 1, "T": 2}).to_numpy()),
             # Position, not stress: blocks 2 and 3 are counterbalanced. Should find nothing.
             "order control: block 2 vs 3": (paired["block"].isin([2, 3]).to_numpy(),
                                             np.where(paired["block"] == 2, 0, 1))}
    rows = []
    for task, (mask, labels) in tasks.items():
        idx = np.flatnonzero(mask)
        y, g, b = labels[idx], people[idx], blocks[idx]
        B, R = base.iloc[idx].reset_index(drop=True), rich.iloc[idx].reset_index(drop=True)
        # person statistics come from all of that person's windows, not the task subset
        RP = by_person(rich, people).iloc[idx].reset_index(drop=True)
        BP = by_person(base, people).iloc[idx].reset_index(drop=True)
        options = {"base 8, logistic": (logistic, B), "base 8+person, logistic": (logistic, BP),
                   "rich, logistic": (logistic, R), "rich+person, logistic": (logistic, RP),
                   "rich, boosting": (boosting, R), "rich+person, boosting": (boosting, RP)}
        win, blk = run_options(options, y, g, b)
        def one(s):
            rng = np.random.default_rng(s)
            perm = ev.permute_labels(y, g, b, rng).astype(int)
            w, k = run_options(options, perm, g, b)
            return max(w.values()), max(k.values())
        null = np.array(Parallel(n_jobs=-1)(delayed(one)(s) for s in np.random.SeedSequence(SEED).spawn(n_perm)))
        best_w, best_b = max(win.values()), max(blk.values())
        p_w = (1 + (null[:, 0] >= best_w).sum()) / (n_perm + 1)
        p_b = (1 + (null[:, 1] >= best_b).sum()) / (n_perm + 1)
        print(f"\n{task}: n={len(y)} blocks={len(np.unique(b))} best window {best_w:.3f} p={p_w:.4f} | "
              f"best block {best_b:.3f} p={p_b:.4f} | null 95th window {np.percentile(null[:,0],95):.3f} "
              f"block {np.percentile(null[:,1],95):.3f}", flush=True)
        for name in options:
            print(f"  {name:26s} window {win[name]:.3f}  block {blk[name]:.3f}", flush=True)
            rows.append({"task": task, "option": name, "window_f1": win[name], "block_f1": blk[name],
                         "chance": 1 / len(np.unique(y)), "p_best_window": p_w, "p_best_block": p_b,
                         "n_windows": len(y), "n_blocks": len(np.unique(b))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def text_pipe(kind: str, C: float):
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=30000)
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2,
                                   max_features=60000)
    vec = {"word": word(), "char": char(),
           "word+char": FeatureUnion([("w", word()), ("c", char())])}[kind]
    return Pipeline([("v", vec), ("m", LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))])


def text_experiment() -> tuple[pd.DataFrame, dict]:
    df = loaders.load_dreaddit()
    n_train = len(pd.read_csv(config.DREADDIT_TRAIN_PATH))
    tr, te = df.iloc[:n_train], df.iloc[n_train:]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for kind in ["word", "char", "word+char"]:
        for C in [1.0, 3.0, 10.0, 30.0]:
            pred = cross_val_predict(text_pipe(kind, C), tr["text"], tr["stress_label"],
                                     groups=tr["post_id"], cv=cv)
            rows.append({"kind": kind, "C": C, "train_cv_f1": f1_score(tr["stress_label"], pred, average="macro")})
            print(f"  {kind:10s} C={C:<5} train CV macro F1 {rows[-1]['train_cv_f1']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    best = table.loc[table["train_cv_f1"].idxmax()]
    model = text_pipe(best["kind"], float(best["C"])).fit(tr["text"], tr["stress_label"])
    proba = model.predict_proba(te["text"])
    pred = model.predict(te["text"])
    pos = list(model.classes_).index("High")
    result = {"chosen": f"{best['kind']}, C={best['C']}", "test_f1": f1_score(te["stress_label"], pred, average="macro"),
              "test_acc": accuracy_score(te["stress_label"], pred),
              "test_auc": roc_auc_score(te["stress_label"] == "High", proba[:, pos])}
    print("chosen on training data only:", result, flush=True)
    pd.DataFrame({"y": te["stress_label"].to_numpy(), "pred": pred, "post_id": te["post_id"].to_numpy()}).to_csv(
        OUT / "text_tfidf_test_predictions.csv", index=False)
    return table, result


def lexicon_experiment() -> dict:
    """
    TF-IDF plus the LIWC, DAL and syntax columns Dreaddit ships. LIWC is licensed, so
    this could not run on what a user types live: it is a benchmark, not the product.
    """
    frames = [pd.read_csv(p) for p in (config.DREADDIT_TRAIN_PATH, config.DREADDIT_TEST_PATH)]
    n_train = len(frames[0])
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c.startswith(("lex_", "syntax_")) or c == "sentiment"]
    model = Pipeline([("x", ColumnTransformer([
        ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2,
                              max_features=60000), "text"),
        ("l", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols)])),
        ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    tr, te = raw.iloc[:n_train], raw.iloc[n_train:]
    model.fit(tr[["text"] + cols], tr["label"])
    proba = model.predict_proba(te[["text"] + cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n_lexicon_features": len(cols), "test_f1": f1_score(te["label"], pred, average="macro"),
            "test_acc": accuracy_score(te["label"], pred), "test_auc": roc_auc_score(te["label"], proba)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== TEXT ===", flush=True)
    t, r = text_experiment()
    t.round(4).to_csv(OUT / "text_options.csv", index=False)
    pd.DataFrame([r]).round(4).to_csv(OUT / "text_result.csv", index=False)
    lex = lexicon_experiment()
    print("with Dreaddit's LIWC columns (benchmark only):", lex, flush=True)
    pd.DataFrame([lex]).round(4).to_csv(OUT / "text_lexicon_benchmark.csv", index=False)
    print("=== TYPING ===", flush=True)
    typing_experiment(a.perm).round(4).to_csv(OUT / "typing_options.csv", index=False)
    print(f"written to {OUT}/")
