"""
Optional transformer baseline for the text pipeline (DistilBERT on Dreaddit).

Not part of the core project: it needs PyTorch and Hugging Face transformers
(pip install -r requirements-transformer.txt) and downloads ~260 MB of
pretrained weights on first use. Everything else runs without it.

It answers one question for the report: how much accuracy does the
interpretable classical pipeline (VADER + lexical features + TF-IDF) give up
against a fine-tuned transformer? The train/test split is IDENTICAL to
run_midterm.py (same rows, same seed), so the numbers compare directly.

Usage:
    python run_transformer.py                 # full run (GPU / Apple MPS if available)
    python run_transformer.py --limit 200     # quick smoke test on a subset
    python run_transformer.py --epochs 3 --model bert-base-uncased

Writes outputs/transformer/results_transformer.csv and report_transformer.md.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from dataio import loaders
from modeling import evaluation

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError:  # pragma: no cover
    torch = None


def pick_device(requested: str) -> "torch.device":
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(tok, texts, labels, max_len) -> TensorDataset:
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"],
                         torch.tensor(labels, dtype=torch.long))


def main(args) -> pd.DataFrame | None:
    if torch is None:
        print("PyTorch / transformers are not installed. This comparison is optional:\n"
              "    pip install -r requirements-transformer.txt")
        return None

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    out_dir = config.OUTPUT_DIR / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = loaders.load_dreaddit()
    y = df["stress_label"].astype(str)
    if args.split == "official":
        # the dataset authors' split: no post appears on both sides
        n_train = len(pd.read_csv(config.DREADDIT_TRAIN_PATH))
        tr_idx, te_idx = np.arange(n_train), np.arange(n_train, len(df))
    else:
        # same call, same rows, same seed as run_midterm.run_text()
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)), test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=y,
        )
    if args.limit:
        rng = np.random.default_rng(config.RANDOM_STATE)
        tr_idx = rng.choice(tr_idx, size=min(args.limit, len(tr_idx)), replace=False)
        te_idx = rng.choice(te_idx, size=min(max(args.limit // 4, 50), len(te_idx)), replace=False)

    classes = sorted(y.unique())
    to_id = {c: i for i, c in enumerate(classes)}
    device = pick_device(args.device)
    print(f"Model {args.model} on {device}; train {len(tr_idx)}, test {len(te_idx)}"
          + (" (SUBSET - smoke test only)" if args.limit else ""))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes)).to(device)

    train_ds = encode(tok, df["text"].iloc[tr_idx], y.iloc[tr_idx].map(to_id).to_numpy(), args.max_len)
    test_ds = encode(tok, df["text"].iloc[te_idx], y.iloc[te_idx].map(to_id).to_numpy(), args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(config.RANDOM_STATE))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for i, (ids, mask, lab) in enumerate(train_dl, 1):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=lab.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += out.loss.item()
            if i % 20 == 0:
                print(f"  epoch {epoch + 1} step {i}/{len(train_dl)} loss {total / i:.4f}")
        print(f"epoch {epoch + 1}: mean loss {total / max(len(train_dl), 1):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for ids, mask, _ in test_dl:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    y_te = y.iloc[te_idx].to_numpy()
    y_pred = np.array(classes)[probs.argmax(axis=1)]

    row = evaluation.evaluate_model(y_te, y_pred, args.model)
    row.update({"pipeline": "text", "variant": "fine-tuned transformer",
                "epochs": args.epochs, "subset": bool(args.limit),
                "majority_baseline_acc": evaluation.majority_baseline(y_te),
                "train_minutes": round((time.time() - start) / 60, 1)})
    row["split"] = args.split
    table = pd.DataFrame([row]).round(4)
    suffix = "" if args.split == "random" else f"_{args.split}"
    if args.model != "distilbert-base-uncased":
        suffix += "_" + args.model.replace("/", "-")
    table.to_csv(out_dir / f"results_transformer{suffix}.csv", index=False)
    pd.DataFrame({"y": y_te, "pred": y_pred, "proba_high": probs[:, classes.index("High")],
                  "post_id": df["post_id"].iloc[te_idx].to_numpy()}).to_csv(
        out_dir / f"test_predictions{suffix}.csv", index=False)
    evaluation.save_confusion_matrix(y_te, y_pred, "text_transformer",
                                     out_dir=out_dir, labels=classes)

    lines = ["# Transformer Baseline (Dreaddit)", ""]
    if args.limit:
        lines += ["> **Subset smoke test - do not report these numbers.**", ""]
    lines += [table.to_markdown(index=False), ""]
    classical = config.OUTPUT_DIR / "midterm_results.csv"
    if classical.exists():
        c = pd.read_csv(classical)
        c = c[c.pipeline == "text"][["model", "accuracy", "f1_macro"]]
        lines += ["## Classical pipeline on the same split", "", c.to_markdown(index=False), "",
                  f"Difference (transformer minus best classical macro F1): "
                  f"{row['f1_macro'] - c.f1_macro.max():+.3f}", ""]
    lines += ["The classical pipeline remains the reported model because it is "
              "interpretable: it is what surfaced first-person pronoun ratio as "
              "the strongest single predictor (Decision Log D9).", ""]
    (out_dir / f"report_transformer{suffix}.md").write_text("\n".join(lines))

    print("\n" + table.to_string(index=False))
    print(f"\nartefacts written to {out_dir}/")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--split", default="random", choices=["random", "official"])
    ap.add_argument("--limit", type=int, default=0, help="train on a subset (smoke test)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    sys.exit(0 if main(ap.parse_args()) is not None else 1)
