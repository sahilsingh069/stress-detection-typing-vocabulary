"""
Text model search on Dreaddit, with the official test set kept out of every choice.

Fixed before running:
  Dev set     15% of the official training posts, whole posts, seed 42.
  Candidates  CANDIDATES below, 3 epochs each at lr 2e-5. Best epoch by dev macro F1.
  Ensemble    Add models greedily, best dev score first, while the equal-weight
              average improves dev macro F1. The classical model may join.
  Final       Each chosen model is retrained on the whole training set for its best
              epoch count, with 3 seeds, and averaged. The test set is scored once.

Needs requirements-transformer.txt. Finished candidates are reused, so it can resume.

    python run_text_search.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import config
from dataio import loaders

OUT = config.OUTPUT_DIR / "text_search"
CANDIDATES = ["distilbert-base-uncased", "roberta-base", "google/electra-base-discriminator",
              "vinai/bertweet-base", "microsoft/deberta-v3-small", "mental/mental-roberta-base"]
EPOCHS, LR, BATCH, MAX_LEN = 3, 2e-5, 16, 256
FINAL_SEEDS = [42, 43, 44]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(tok, texts, labels=None, length=MAX_LEN) -> TensorDataset:
    enc = tok(list(texts), truncation=True, padding="max_length", max_length=length, return_tensors="pt")
    y = torch.tensor(labels if labels is not None else [0] * len(texts), dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def predict(model, data) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for ids, mask, _ in DataLoader(data, batch_size=BATCH * 2):
            out.append(torch.softmax(model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train(name, tr_text, tr_y, evals: dict, epochs: int, seed: int, on_epoch=None):
    """Trains and returns {eval name: probabilities after the last epoch}."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    # .float(): DeBERTa's checkpoint is stored in float16, which trains into NaN.
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).float().to(DEVICE)
    # BERTweet accepts 128 tokens, not 256; the 2 reserves its position offset.
    length = min(MAX_LEN, model.config.max_position_embeddings - 2)
    data = encode(tok, tr_text, tr_y, length)
    sets = {k: encode(tok, v, length=length) for k, v in evals.items()}
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    probs = {}
    for epoch in range(1, epochs + 1):
        model.train()
        for ids, mask, lab in loader:
            loss = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE), labels=lab.to(DEVICE)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        probs = {k: predict(model, v) for k, v in sets.items()}
        if on_epoch:
            on_epoch(epoch, probs)
    return probs, model, tok


def f1(y, p) -> float:
    return f1_score(y, (p >= 0.5).astype(int), average="macro")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = loaders.load_dreaddit()
    n_train = len(pd.read_csv(config.DREADDIT_TRAIN_PATH))
    train_df, test_df = df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
    y_all = (train_df["stress_label"].astype(str) == "High").astype(int).to_numpy()
    tr, dev = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(train_df, y_all, train_df["post_id"]))
    print(f"train {len(tr)} dev {len(dev)} test kept aside {len(test_df)} on {DEVICE}", flush=True)
    y_dev = y_all[dev]

    results_path = OUT / "dev_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    dev_probs = {}
    for name in CANDIDATES:
        slug = name.replace("/", "-")
        if name in results and (OUT / f"dev_{slug}.npy").exists():
            dev_probs[name] = np.load(OUT / f"dev_{slug}.npy")
            continue
        start = time.time()
        best = {"f1": -1}

        def keep(epoch, probs):
            if np.isnan(probs["dev"]).any():
                raise ValueError("model produced NaN")
            score = f1(y_dev, probs["dev"])
            print(f"  {name} epoch {epoch} dev macro F1 {score:.4f}", flush=True)
            if score > best["f1"]:
                best.update(f1=score, epoch=epoch)
                np.save(OUT / f"dev_{slug}.npy", probs["dev"])

        try:
            train(name, train_df["text"].iloc[tr], y_all[tr], {"dev": train_df["text"].iloc[dev]}, EPOCHS, 42, keep)
        except Exception as err:  # e.g. a model that cannot be downloaded
            print(f"  skipped {name}: {err}", flush=True)
            results[name] = {"f1": None, "error": str(err)[:200]}
            results_path.write_text(json.dumps(results, indent=1))
            continue
        results[name] = {**best, "minutes": round((time.time() - start) / 60, 1)}
        results_path.write_text(json.dumps(results, indent=1))
        dev_probs[name] = np.load(OUT / f"dev_{slug}.npy")
    print(json.dumps(results, indent=1), flush=True)

    # classical model, fitted without the dev set, so it can compete on dev
    import train_demo
    from features import text_features

    def classical(fit, score):
        model, ling = train_demo.train_text(fit["text"], fit["stress_label"], [])
        X = pd.concat([score[["text"]].reset_index(drop=True),
                       text_features.build_text_features(score["text"].reset_index(drop=True))[ling]], axis=1)
        return model.predict_proba(X)[:, list(model.classes_).index("High")]

    dev_probs["classical"] = classical(train_df.iloc[tr], train_df.iloc[dev])
    order = sorted(dev_probs, key=lambda k: f1(y_dev, dev_probs[k]), reverse=True)
    chosen, score = [order[0]], f1(y_dev, dev_probs[order[0]])
    for name in order[1:]:
        trial = f1(y_dev, np.mean([dev_probs[k] for k in chosen + [name]], axis=0))
        if trial > score:
            chosen.append(name)
            score = trial
    print(f"chosen on dev: {chosen}, dev macro F1 {score:.4f}", flush=True)
    (OUT / "chosen.json").write_text(json.dumps({"chosen": chosen, "dev_f1": score,
                                                 "dev_each": {k: f1(y_dev, v) for k, v in dev_probs.items()}}, indent=1))

    # final: retrain on all training posts, score the test set once
    members = {}
    for name in chosen:
        if name == "classical":
            members[name] = classical(train_df, test_df)
            continue
        runs = []
        for seed in FINAL_SEEDS:
            path = OUT / f"test_{name.replace('/', '-')}_{seed}.npy"
            if not path.exists():
                print(f"final {name} seed {seed}", flush=True)
                probs, model, tok = train(name, train_df["text"], y_all, {"test": test_df["text"]},
                                          results[name]["epoch"], seed)
                np.save(path, probs["test"])
                if seed == FINAL_SEEDS[0]:
                    model.save_pretrained(OUT / "models" / name.replace("/", "-"))
                    tok.save_pretrained(OUT / "models" / name.replace("/", "-"))
                del model
                torch.mps.empty_cache() if DEVICE.type == "mps" else None
            runs.append(np.load(path))
        members[name] = np.mean(runs, axis=0)
    pd.DataFrame({"y": test_df["stress_label"].astype(str), "post_id": test_df["post_id"],
                  "proba_high": np.mean(list(members.values()), axis=0),
                  **{f"member_{k}": v for k, v in members.items()}}).to_csv(OUT / "test_predictions.csv", index=False)
    print("final predictions written", flush=True)
    report()


def report() -> None:
    """Scores the saved test predictions. The headline is the ensemble chosen on dev."""
    from modeling import extended_eval as ev
    from run_text_ensemble import CLASSES, TF, classical_probabilities

    d = pd.read_csv(OUT / "test_predictions.csv")
    y, groups = d["y"].to_numpy(), d["post_id"].to_numpy()
    classical, _ = classical_probabilities()
    earlier = [pd.read_csv(TF / f)["proba_high"].to_numpy()
               for f in ("test_predictions_official.csv", "test_predictions_official_roberta-base.csv")]
    models = {"Ensemble chosen on dev (headline)": d["proba_high"].to_numpy(),
              **{c.removeprefix("member_"): d[c].to_numpy() for c in d if c.startswith("member_")},
              "Earlier ensemble": np.mean([classical, *earlier], axis=0), "Classical": classical}
    both = lambda high: np.column_stack([high, 1 - high])
    label = lambda high: np.where(high >= 0.5, "High", "Low")
    rows, tests = [], []
    head = label(models["Ensemble chosen on dev (headline)"])
    for name, high in models.items():
        m = ev.full_metrics(y, both(high), CLASSES)
        ci = ev.bootstrap_ci(y, both(high), CLASSES, groups, n_boot=2000)
        rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["f1_macro"], "f1_lo": ci["f1_lo"],
                     "f1_hi": ci["f1_hi"], "roc_auc": m["roc_auc"], "calibration_error": m["calibration_error"]})
        if not name.startswith("Ensemble chosen"):
            tests.append({"headline minus": name, **ev.bootstrap_f1_difference(y, head, label(high), groups, 2000)})
    table, gains = pd.DataFrame(rows).round(4), pd.DataFrame(tests).round(4)
    table.to_csv(OUT / "results.csv", index=False)
    gains.to_csv(OUT / "gains.csv", index=False)
    (OUT / "report_text_search.md").write_text("# Text search, Dreaddit official test set\n\n" + table.to_markdown(index=False)
                                               + "\n\n" + gains.to_markdown(index=False) + "\n")
    print(table.to_string(index=False), "\n", gains.to_string(index=False), flush=True)


if __name__ == "__main__":
    sys.exit(report() if "--report" in sys.argv else main())
