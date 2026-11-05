"""
Can typing be made stronger on people the model has never seen?

Options (fixed before running) are chosen by inner folds on training people only, so the
outer score is not inflated by trying many. Scored per block: one person's whole condition,
averaging its windows. Significance reruns the whole nested procedure on shuffled labels.

  SWELL-KW   Word keystrokes, all 25 people. 2 or 5 minute windows; base, gap distribution,
             rhythm and letter-pair features; raw or standardised within the person;
             Logistic Regression or shallow boosting.
  EmoSurv    Free text. Raw or minus the person's neutral block; same feature families.
  Separately SWELL-KW per-minute features with mouse and app use. Reported apart, because
             application use differs by task as well as by stress.

    python run_typing_search.py [--perm 200]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataio import emosurv, swell_raw
from features import keystroke_dynamics as kd
from features import typing_features
from modeling import extended_eval as ev
from modeling.nested import Option, Part, nested_proba

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "typing_search"
SEED = config.RANDOM_STATE
REPEATS = 10
EMOTION_TASKS = {
    "anger vs calm": {"A": "negative", "C": "positive"},
    "negative vs positive": {"A": "negative", "S": "negative", "H": "positive", "C": "positive"},
    "active vs passive": {"A": "active", "H": "active", "S": "passive", "C": "passive"},
    "four emotions": {"A": "anger", "H": "happiness", "C": "calmness", "S": "sadness"},
}


def logistic():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced"))])


def boosting():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.05,
                                                          class_weight="balanced", random_state=SEED))])


def by_person(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each feature as z-scores within the person, from their own typing. No labels used."""
    out = table.copy()
    values = table[columns].astype(float)
    values = values.fillna(values.median())
    grouped = values.groupby(table["person"].to_numpy())
    out[columns] = ((values - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)).fillna(0)
    return out


# --------------------------------------------------------------------------
# SWELL-KW
# --------------------------------------------------------------------------
def swell_windows(keys: pd.DataFrame, seconds: int, min_presses: int) -> pd.DataFrame:
    start = keys.groupby(["participant_id", "block"])["timestamp"].transform("min")
    win = ((keys["timestamp"] - start).dt.total_seconds() // seconds).astype(int)
    k = keys.assign(session_id=keys["participant_id"] + "_B" + keys["block"].astype(str) + "_W" + win.astype(str))

    def gaps(frame):
        return frame["timestamp"].diff().dt.total_seconds().fillna(0).to_numpy()[1:] * 1000

    usable = lambda frame: frame["key"].astype(str).to_numpy()
    tables = {p: kd.pair_table([(usable(b), gaps(b)) for _, b in g.groupby("session_id")])
              for p, g in k.groupby("participant_id")}
    shared = kd.pair_table([(usable(b), gaps(b)) for _, b in k.groupby("session_id")])

    base = typing_features.features_per_session(k[["session_id", "timestamp", "key", "event"]], min_presses=min_presses)
    rows = []
    for sid, b in k.groupby("session_id"):
        if len(b) < min_presses:
            continue
        person = b["participant_id"].iloc[0]
        g = gaps(b)
        rows.append({"session_id": sid, "person": person, "block": int(b["block"].iloc[0]),
                     "condition": b["condition"].iloc[0], "unit": f"{person}_B{int(b['block'].iloc[0])}",
                     **{f"rich_{a}": v for a, v in typing_features.rich_features(b["timestamp"], b["key"]).items()},
                     **kd.rhythm_features(g),
                     **kd.pair_features(usable(b), g, tables[person], "pair_own"),
                     **kd.pair_features(usable(b), g, shared, "pair_all")})
    keep = [c for c in config.OWN_TYPING_FEATURES if c in base.columns and base[c].notna().any()]
    return pd.DataFrame(rows).merge(base[["session_id"] + keep].rename(columns={c: f"base_{c}" for c in keep}), on="session_id")


def groups_of(table: pd.DataFrame) -> dict:
    col = lambda *p: [c for c in table.columns if c.startswith(p)]
    return {"base": col("base_"), "rich": col("rich_"), "rich+rhythm": col("rich_", "rhythm_"),
            "rich+rhythm+pairs": col("rich_", "rhythm_", "pair_"), "base+rhythm+pairs": col("base_", "rhythm_", "pair_")}


def swell_options(keys: pd.DataFrame) -> list[Option]:
    options = []
    for seconds, floor in [(120, 40), (300, 100)]:
        raw = swell_windows(keys, seconds, floor)
        print(f"windows {seconds}s: {len(raw)} rows, {raw.person.nunique()} people", flush=True)
        groups = groups_of(raw)
        scaled = by_person(raw, sorted({c for v in groups.values() for c in v}))
        for name, cols in groups.items():
            options.append(Option(f"{seconds}s {name} raw", [Part(raw, cols, logistic())]))
            options.append(Option(f"{seconds}s {name} person", [Part(scaled, cols, logistic())]))
        options.append(Option(f"{seconds}s all person boost", [Part(scaled, groups["rich+rhythm+pairs"], boosting())]))
    return options


def swell_minute_options() -> list[Option]:
    d = pd.read_csv(config.SWELL_PATH)
    d = d[d["Condition"].isin(["N", "I", "T"])]
    cols = [c for c in d.columns if c.startswith("Sn")]
    table = pd.DataFrame({"person": d["PP"].astype(str), "unit": d["PP"].astype(str) + "_B" + d["Blok"].astype(str),
                          **{c: pd.to_numeric(d[c], errors="coerce") for c in cols}})
    scaled = by_person(table, cols)
    return [Option("minutes raw", [Part(table, cols, logistic())]), Option("minutes person", [Part(scaled, cols, logistic())]),
            Option("minutes person boost", [Part(scaled, cols, boosting())])]


def swell_units(keys: pd.DataFrame) -> pd.DataFrame:
    u = keys.groupby(["participant_id", "block"])["condition"].first().reset_index()
    u["unit"] = u["participant_id"] + "_B" + u["block"].astype(str)
    return u.rename(columns={"participant_id": "person"}).set_index("unit")


# --------------------------------------------------------------------------
# EmoSurv
# --------------------------------------------------------------------------
def emosurv_options() -> tuple[list[Option], pd.DataFrame]:
    keys = emosurv.load_keys("free")
    seq = {}
    for k, g in keys.groupby(["participant_id", "emotion"]):
        # Clock first, then drop auto-repeats: a press's gap runs to the next press, not the next logged row.
        clock = g["gap_ms"].fillna(0).cumsum().shift(fill_value=0).to_numpy()
        pressed = g["is_press"].to_numpy()
        seq[k] = (g["key"].to_numpy()[pressed], np.diff(clock[pressed]))
    shared = kd.pair_table(seq.values())
    blocks = emosurv.load_blocks("free")
    extra = [{**kd.rhythm_features(seq[(p, e)][1]), **kd.pair_features(*seq[(p, e)], shared, "pair_all")}
             for p, e in zip(blocks["participant_id"], blocks["emotion"])]
    blocks = pd.concat([blocks, pd.DataFrame(extra)], axis=1)
    all_cols = emosurv.feature_columns(blocks)
    base = emosurv.from_baseline(blocks)
    frames = {}
    for name, f in {"raw": blocks[blocks["emotion"] != "N"].reset_index(drop=True), "baseline": base}.items():
        frames[name] = f.assign(person=f["participant_id"], unit=f["participant_id"] + "_" + f["emotion"])
    timing = [c for c in all_cols if not c.startswith(("rhythm_", "pair_"))]
    groups = {"timing": timing, "timing+rhythm": timing + [c for c in all_cols if c.startswith("rhythm_")],
              "all": all_cols}
    options = []
    for src, f in frames.items():
        for name, cols in groups.items():
            options.append(Option(f"{src} {name}", [Part(f, cols, logistic())]))
        options.append(Option(f"{src} all boost", [Part(f, all_cols, boosting())]))
    units = frames["raw"].set_index("unit")[["person", "emotion"]]
    return options, units


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def run_task(dataset, task, options, units, labels, shuffle, n_perm) -> dict:
    y = np.asarray(labels)
    classes = sorted(set(y))
    runs = Parallel(n_jobs=-1)(delayed(nested_proba)(options, units, y, s) for s in range(REPEATS))
    f1s = [ev.full_metrics(y, p, classes)["f1_macro"] for p, _ in runs]
    proba, _ = runs[0]
    mean_proba = np.mean([p for p, _ in runs], axis=0)
    m = ev.full_metrics(y, mean_proba, classes)
    ci = ev.bootstrap_ci(y, mean_proba, classes, units["person"].to_numpy(), n_boot=1000)
    picks = pd.Series([x for _, ps in runs for x in ps]).value_counts(normalize=True).head(3).round(2).to_dict()
    p_value = None
    observed = ev.full_metrics(y, proba, classes)["f1_macro"]
    print(f"{dataset} | {task}: F1 {np.mean(f1s):.3f} (averaged {m['f1_macro']:.3f}, "
          f"CI {ci['f1_lo']:.3f}-{ci['f1_hi']:.3f}); permutation test next", flush=True)
    if n_perm:
        def one(s):
            lab = shuffle(y, np.random.default_rng(s))
            return ev.full_metrics(lab, nested_proba(options, units, lab, 0)[0], sorted(set(lab)))["f1_macro"]
        null = np.array(Parallel(n_jobs=-1)(delayed(one)(s) for s in np.random.SeedSequence(SEED).spawn(n_perm)))
        p_value = (1 + (null >= observed).sum()) / (n_perm + 1)
    row = {"dataset": dataset, "task": task, "n_units": len(y), "n_people": units["person"].nunique(),
           "chance_f1": 1 / len(classes), "f1_mean_of_repeats": np.mean(f1s), "f1_sd": np.std(f1s),
           "f1_averaged_probs": m["f1_macro"], "f1_lo": ci["f1_lo"], "f1_hi": ci["f1_hi"], "roc_auc": m["roc_auc"],
           "accuracy": m["accuracy"], "majority_acc": m["majority_acc"], "first_run_f1": observed, "p_nested": p_value,
           "picked": str(picks)}
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def main(n_perm: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    if n_perm:
        rows = [r for r in rows if pd.notna(r["p_nested"])]

    def run(dataset, task, *args):
        """Tasks already saved are skipped, so a restart resumes instead of starting over."""
        if not any(r["dataset"] == dataset and r["task"] == task for r in rows):
            rows.append(run_task(dataset, task, *args))
            pd.DataFrame(rows).round(4).to_csv(path, index=False)

    keys = swell_raw.windowed_word_keys().sort_values(["participant_id", "block", "timestamp"])
    units = swell_units(keys)
    swell_opts = swell_options(keys)
    minute_opts = swell_minute_options()
    tasks = {"neutral vs time pressure": (["N", "T"], lambda u: u["condition"]),
             "neutral vs either stressor": (["N", "I", "T"], lambda u: np.where(u["condition"] == "N", "N", "S")),
             "interruption vs time pressure": (["I", "T"], lambda u: u["condition"]),
             "3-class": (["N", "I", "T"], lambda u: u["condition"]),
             "order control: block 2 vs 3": (None, lambda u: u["block"].astype(str))}

    def within_person(u):
        def shuffle(y, rng):
            out = pd.Series(y, index=u.index)
            for _, idx in u.groupby("person").groups.items():
                out.loc[idx] = rng.permutation(out.loc[idx].to_numpy())
            return out.to_numpy()
        return shuffle

    for task, (conds, label) in tasks.items():
        u = units[units["condition"].isin(conds)] if conds else units[units["block"].isin([2, 3])]
        run("SWELL-KW keystrokes", task, swell_opts, u[["person"]], label(u), within_person(u), n_perm)
    for task in ["neutral vs time pressure", "3-class"]:
        conds, label = tasks[task]
        d = pd.read_csv(config.SWELL_PATH)
        d = d[d["Condition"].isin(["N", "I", "T"])].copy()
        d["unit"] = d["PP"].astype(str) + "_B" + d["Blok"].astype(str)
        u = d.groupby("unit").agg(person=("PP", "first"), condition=("Condition", "first"))
        u["person"] = u["person"].astype(str)
        u = u[u["condition"].isin(conds)]
        run("SWELL-KW with mouse and apps", task, minute_opts, u[["person"]], label(u), within_person(u), n_perm)

    em_opts, em_units = emosurv_options()
    for task, mapping in EMOTION_TASKS.items():
        u = em_units[em_units["emotion"].isin(mapping)]
        run("EmoSurv free text", task, em_opts, u[["person"]], u["emotion"].map(mapping),
            lambda y, rng: rng.permutation(y), n_perm)
    table = pd.DataFrame(rows)
    stress = (table["dataset"] == "SWELL-KW keystrokes") & ~table["task"].str.startswith("order")
    table["p_holm"] = np.nan
    for mask in (stress, table["dataset"] == "SWELL-KW with mouse and apps", table["dataset"] == "EmoSurv free text"):
        if n_perm and mask.any():
            table.loc[mask, "p_holm"] = ev.holm_adjust(table.loc[mask, "p_nested"])
    table.round(4).to_csv(OUT / "results.csv", index=False)
    (OUT / "report_typing_search.md").write_text("# Typing search, nested, whole people held out\n\n" + table.round(3).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    main(ap.parse_args().perm)
