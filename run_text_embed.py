"""
Pretrained language features for the paired datasets, fitted on neither of them:
sentence embeddings (all-MiniLM-L6-v2), emotion probabilities (j-hartmann) and sentiment
probabilities (cardiffnlp), plus the stress probability from a model trained on Dreaddit
when run_text_search.py has saved one. Needs requirements-transformer.txt.

    python run_text_embed.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import config
from dataio import emosurv, swell_raw

OUT = config.OUTPUT_DIR / "text_features"
SEARCH = config.OUTPUT_DIR / "text_search"
CLASSIFIERS = {"emotion": "j-hartmann/emotion-english-distilroberta-base",
               "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest"}


def batches(texts, size=16):
    for i in range(0, len(texts), size):
        yield texts[i:i + size]


def embed(texts) -> np.ndarray:
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for b in batches(texts):
            enc = tok(b, truncation=True, padding=True, max_length=256, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
    return np.concatenate(out)


def classify(name, texts) -> tuple[np.ndarray, list[str]]:
    tok, model = AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for b in batches(texts):
            enc = tok(b, truncation=True, padding=True, max_length=256, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, 1).numpy())
    return np.concatenate(out), [model.config.id2label[i] for i in range(model.config.num_labels)]


def build(name: str, frame: pd.DataFrame) -> None:
    texts = frame["text"].astype(str).tolist()
    out = pd.DataFrame({"key": frame["key"]})
    for i, column in enumerate(embed(texts).T):
        out[f"emb_{i}"] = column
    for kind, model in CLASSIFIERS.items():
        probs, labels = classify(model, texts)
        for j, label in enumerate(labels):
            out[f"{kind}_{label}"] = probs[:, j]
    chosen = SEARCH / "chosen.json"
    if chosen.exists():
        best = [m for m in json.loads(chosen.read_text())["chosen"] if m != "classical"]
        for m in best:
            path = SEARCH / "models" / m.replace("/", "-")
            if path.exists():
                out[f"dreaddit_{m.replace('/', '-')}"] = classify(str(path), texts)[0][:, 1]
    out.to_csv(OUT / f"{name}.csv.gz", index=False)
    print(name, out.shape, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    em = emosurv.load_blocks("free")
    em = em[em["text"].str.split().str.len() >= 5]
    build("emosurv", em.assign(key=em["participant_id"] + "_" + em["emotion"])[["key", "text"]])
    sw = swell_raw.load_paired()
    build("swell", sw.assign(key=sw["session_id"])[["key", "text"]])


if __name__ == "__main__":
    main()
