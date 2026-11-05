"""
Can typing and text together beat either alone? Nested, whole people held out.

Fixed before running. Three competitors, each choosing among its own options on training
people only, and scored on the same units:

  typing   gap and rhythm features, raw or standardised within the person
  text     TF-IDF; sentence embeddings; pretrained emotion, sentiment and Dreaddit stress scores
  combined everything above, plus weighted averages of one typing and one text option
           (typing weight 0.25, 0.5 or 0.75), so it can also fall back to a single stream

Run run_text_embed.py first. Scored per 5-minute window and per block on SWELL-KW (English
writers, because the text models are English) and per block on EmoSurv free text.

    python run_fusion_search.py [--perm 100]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from dataio import swell_raw
from modeling import extended_eval as ev
from modeling.nested import Option, Part, nested_proba
from run_typing_search import (EMOTION_TASKS, by_person, emosurv_options, logistic, swell_windows)

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "fusion_search"
FEATURES = config.OUTPUT_DIR / "text_features"
SEED = config.RANDOM_STATE
REPEATS = 10
WEIGHTS = (0.25, 0.5, 0.75)


def tfidf():
    return Pipeline([("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)),
                     ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def text_parts(table: pd.DataFrame) -> dict:
    """The text options for one table with `text` and pretrained feature columns."""
    emb = [c for c in table.columns if c.startswith("emb_")]
    scores = [c for c in table.columns if c.startswith(("emotion_", "sentiment_", "dreaddit_"))]
    return {"tfidf": lambda w=1.0: Part(table, "text", tfidf(), w),
            "embeddings": lambda w=1.0: Part(table, emb, logistic(), w),
            "scores": lambda w=1.0: Part(table, scores, logistic(), w),
            "embeddings+scores": lambda w=1.0: Part(table, emb + scores, logistic(), w)}


def competitors(typing_parts: dict, text: dict) -> dict:
    typing = [Option(n, [p()]) for n, p in typing_parts.items()]
    words = [Option(n, [p()]) for n, p in text.items()]
    mixed = [Option(f"{a} + {b} w{w}", [pa(w), pb(1 - w)]) for a, pa in typing_parts.items()
             for b, pb in text.items() for w in WEIGHTS]
    return {"typing": typing, "text": words, "combined": typing + words + mixed}


def score(dataset, task, sets, units, labels, shuffle, n_perm) -> tuple[list, list]:
    y = np.asarray(labels)
    classes = sorted(set(y))
    people = units["person"].to_numpy()
    averaged, rows = {}, []
    for name, options in sets.items():
        runs = Parallel(n_jobs=-1)(delayed(nested_proba)(options, units, y, s) for s in range(REPEATS))
        f1s = [ev.full_metrics(y, p, classes)["f1_macro"] for p, _ in runs]
        averaged[name] = np.mean([p for p, _ in runs], axis=0)
        m = ev.full_metrics(y, averaged[name], classes)
        ci = ev.bootstrap_ci(y, averaged[name], classes, people, n_boot=1000)
        p_value = None
        if n_perm:
            def one(s):
                lab = shuffle(y, np.random.default_rng(s))
                return ev.full_metrics(lab, nested_proba(options, units, lab, 0)[0], sorted(set(lab)))["f1_macro"]
            null = np.array(Parallel(n_jobs=-1)(delayed(one)(s) for s in np.random.SeedSequence(SEED).spawn(n_perm)))
            p_value = (1 + (null >= f1s[0]).sum()) / (n_perm + 1)
        picks = pd.Series([x for _, ps in runs for x in ps]).value_counts(normalize=True).head(3).round(2).to_dict()
        rows.append({"dataset": dataset, "task": task, "stream": name, "n_units": len(y), "n_people": len(set(people)),
                     "chance_f1": 1 / len(classes), "f1_mean_of_repeats": np.mean(f1s), "f1_sd": np.std(f1s),
                     "f1_averaged_probs": m["f1_macro"], "f1_lo": ci["f1_lo"], "f1_hi": ci["f1_hi"], "roc_auc": m["roc_auc"],
                     "accuracy": m["accuracy"], "majority_acc": m["majority_acc"], "p_nested": p_value, "picked": str(picks)})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pred = lambda p: np.asarray(classes)[p.argmax(axis=1)]
    best_single = max(("typing", "text"), key=lambda k: ev.full_metrics(y, averaged[k], classes)["f1_macro"])
    d = ev.bootstrap_f1_difference(y, pred(averaged["combined"]), pred(averaged[best_single]), people)
    gain = {"dataset": dataset, "task": task, "best_single": best_single,
            "combined_minus_best_single": d["difference"], "lo": d["lo"], "hi": d["hi"]}
    print(gain, flush=True)
    return rows, [gain]


def swell_sets() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    paired = swell_raw.load_paired()
    keys = swell_raw.windowed_word_keys().sort_values(["participant_id", "block", "timestamp"])
    win = swell_windows(keys, 300, 100)
    col = lambda *p: [c for c in win.columns if c.startswith(p)]
    base, full = col("base_"), col("rich_", "rhythm_", "pair_")
    scaled = by_person(win, base + full)
    text = paired[["session_id", "text"]].merge(pd.read_csv(FEATURES / "swell.csv.gz"), left_on="session_id", right_on="key")
    join = lambda t: t.merge(text, on="session_id").reset_index(drop=True)
    return {"raw": join(win), "person": join(scaled), "cols": (base, full)}, join(win), paired


def main(n_perm: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, gains = [], []
    tables, win, _ = swell_sets()
    base, full = tables["cols"]
    within = lambda u: (lambda y, rng: pd.Series(y, index=u.index).groupby(u["person"]).transform(
        lambda s: rng.permutation(s.to_numpy())).to_numpy())
    labelings = {"neutral vs time pressure": (["N", "T"], lambda c: c),
                 "neutral vs either stressor": (["N", "I", "T"], lambda c: np.where(c == "N", "N", "S")),
                 "interruption vs time pressure": (["I", "T"], lambda c: c), "3-class": (["N", "I", "T"], lambda c: c)}
    for level, unit_of in [("block", lambda t: t["unit"]), ("window", lambda t: t["session_id"])]:
        raw, person = tables["raw"].assign(unit=unit_of(tables["raw"])), tables["person"].assign(unit=unit_of(tables["person"]))
        typing = {"base raw": lambda w=1.0: Part(raw, base, logistic(), w),
                  "base person": lambda w=1.0: Part(person, base, logistic(), w),
                  "full person": lambda w=1.0: Part(person, full, logistic(), w)}
        sets = competitors(typing, text_parts(raw))
        info = raw.groupby("unit").agg(person=("person", "first"), condition=("condition", "first"), block=("unit", "first"))
        for task, (conds, label) in labelings.items():
            u = info[info["condition"].isin(conds)]
            # windows of one block share a label, so shuffled labels must move as blocks
            shuffle = within(u) if level == "block" else None
            a, b = score(f"SWELL-KW per {level}", task, sets, u[["person"]], label(u["condition"].to_numpy()),
                         shuffle, n_perm if level == "block" else 0)
            rows += a
            gains += b
            pd.DataFrame(rows).round(4).to_csv(OUT / "results.csv", index=False)

    options, em_units = emosurv_options()
    extra = pd.read_csv(FEATURES / "emosurv.csv.gz")
    blocks = options[0].parts[0].table[["unit", "person", "text"]].merge(extra, left_on="unit", right_on="key")
    keep = set(blocks["unit"])
    def narrowed(name):
        part = next(o for o in options if o.name == name).parts[0]
        table = part.table[part.table["unit"].isin(keep)]
        return lambda w=1.0: Part(table, part.columns, logistic(), w)
    typing = {"raw timing": narrowed("raw timing"), "baseline timing": narrowed("baseline timing"), "raw all": narrowed("raw all")}
    sets = competitors(typing, text_parts(blocks))
    for task, mapping in EMOTION_TASKS.items():
        u = em_units[em_units.index.isin(keep) & em_units["emotion"].isin(mapping)]
        a, b = score("EmoSurv free text", task, sets, u[["person"]], u["emotion"].map(mapping),
                     lambda y, rng: rng.permutation(y), n_perm)
        rows += a
        gains += b
        pd.DataFrame(rows).round(4).to_csv(OUT / "results.csv", index=False)

    table, gain = pd.DataFrame(rows), pd.DataFrame(gains)
    gain.round(4).to_csv(OUT / "gains.csv", index=False)
    (OUT / "report_fusion_search.md").write_text(
        "# Fusion search, nested, whole people held out\n\n" + table.round(3).to_markdown(index=False)
        + "\n\n## Combined minus the better single stream\n\n" + gain.round(3).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=100)
    main(ap.parse_args().perm)
