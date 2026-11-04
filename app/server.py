"""
Demo web app: type in the browser, see a live stress estimate.

    python train_demo.py      # once, to build outputs/demo/
    python app/server.py      # then open http://127.0.0.1:8765

The page sends the typed text plus raw key-down / key-up timings to
/api/predict. The server computes features with the SAME Python code used
for training (features.typing_features / features.text_features), so the
demo and the reported experiments cannot drift apart.

Nothing is stored: every request is processed in memory and discarded. The
server listens on 127.0.0.1 only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from features import text_features, typing_features  # noqa: E402
from modeling import fusion  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MIN_WORDS = 3
MAX_EVENTS = 50_000
MAX_TEXT = 20_000

app = Flask(__name__, static_folder=None)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Models:
    def __init__(self, directory: Path):
        self.dir = directory
        self.meta: dict = {}
        self.typing = self.text = self.early = None
        self.reload()

    def reload(self):
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            self.meta = {}
            return
        self.meta = json.loads(meta_path.read_text())
        load = lambda name: joblib.load(self.dir / f"{name}.joblib") \
            if (self.dir / f"{name}.joblib").exists() else None
        self.typing = load("typing")
        self.text = load("text")
        self.early = load("early") if "early" in self.meta.get("models", {}) else None

    @property
    def ready(self) -> bool:
        return self.typing is not None and self.text is not None


MODELS = Models(config.DEMO_DIR)


# --------------------------------------------------------------------------
# feature extraction for one live input
# --------------------------------------------------------------------------
def typing_row(events: list[dict]) -> pd.DataFrame | None:
    """Browser key events -> one row of typing features (None if too few keys)."""
    if not events:
        return None
    df = pd.DataFrame(events[:MAX_EVENTS])
    if not {"ev", "t"}.issubset(df.columns):
        return None
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    df["event"] = df["ev"].map({"down": "press", "up": "release"})
    df = df.dropna(subset=["event"])
    df["key"] = df["key"].astype(str) if "key" in df.columns else ""
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", origin="unix")
    df["session_id"] = "live"
    feats = typing_features.features_per_session(
        df, session_col="session_id", min_presses=config.MIN_PRESSES_PER_SESSION)
    return None if feats.empty else feats


def text_row(text: str, ling_cols: list[str]) -> pd.DataFrame:
    ling = text_features.build_text_features(pd.Series([text]))
    row = pd.concat([pd.DataFrame({"text": [text]}), ling], axis=1)
    keep = ["text"] + [c for c in ling_cols if c in row.columns]
    return row[keep]


def _result(proba_row: np.ndarray, source: str, validated: bool, note: str = "") -> dict:
    probs = {c: round(float(p), 4) for c, p in zip(config.STRESS_CLASSES, proba_row)}
    label = max(probs, key=probs.get)
    return {"probs": probs, "label": label, "confidence": probs[label],
            "source": source, "validated": validated, "note": note}


def _aligned(model, X) -> np.ndarray:
    return fusion.align_classes(model.predict_proba(X), [str(c) for c in model.classes_])


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/status")
def status():
    MODELS.reload()
    return jsonify({"ready": MODELS.ready, "meta": MODELS.meta,
                    "min_presses": config.MIN_PRESSES_PER_SESSION,
                    "min_words": MIN_WORDS})


@app.post("/api/predict")
def predict():
    if not MODELS.ready:
        return jsonify({"error": "Models not trained. Run: python train_demo.py"}), 503
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", ""))[:MAX_TEXT]
    events = body.get("events") or []
    if not isinstance(events, list):
        return jsonify({"error": "events must be a list"}), 400
    try:
        weight = float(body.get("weight", MODELS.meta.get("fusion_weight_typing", 0.5)))
    except (TypeError, ValueError):
        weight = 0.5
    weight = min(max(weight, 0.0), 1.0)

    meta = MODELS.meta["models"]
    out: dict = {"warnings": [], "features": {}}

    # typing ---------------------------------------------------------------
    trow = typing_row(events)
    p_typ = None
    if trow is None:
        out["typing"] = None
        out["warnings"].append(
            f"Type at least {config.MIN_PRESSES_PER_SESSION} keys for a typing estimate.")
    else:
        cols = meta["typing"]["features"]
        p_typ = _aligned(MODELS.typing, trow[cols])[0]
        note = "" if meta["typing"]["validated"] else (
            "Trained on the 2-person Kaggle set, which did not generalise across people. "
            "Collect data and re-run train_demo.py." if meta["typing"]["source"] == "kaggle"
            else "Trained on SWELL-KW 5-minute windows from 23 people. On unseen people it is only "
            "weakly better than chance, and short bursts may differ." if meta["typing"]["source"] == "swell"
            else "Trained on synthetic data.")
        out["typing"] = _result(p_typ, meta["typing"]["source"], meta["typing"]["validated"], note)
        out["features"]["typing"] = {
            k: (None if pd.isna(v) else round(float(v), 3))
            for k, v in trow[cols + ["n_keystrokes"]].iloc[0].items()}

    # text -----------------------------------------------------------------
    p_txt = None
    n_words = len(text.split())
    if n_words < MIN_WORDS:
        out["text"] = None
        out["warnings"].append(f"Write at least {MIN_WORDS} words for a text estimate.")
    else:
        xrow = text_row(text, meta["text"]["linguistic_features"])
        p_txt = _aligned(MODELS.text, xrow)[0]
        note = ("Dreaddit labels are binary, so this model never predicts Medium."
                if meta["text"]["source"] == "dreaddit" else
                "Trained on synthetic data." if meta["text"]["source"] == "synthetic" else "")
        out["text"] = _result(p_txt, meta["text"]["source"], meta["text"]["validated"], note)
        f = xrow.drop(columns=["text"]).iloc[0]
        out["features"]["text"] = {k: round(float(f[k]), 4) for k in
                                   ["sent_compound", "sent_neg", "first_person_ratio",
                                    "negative_word_ratio", "stress_word_ratio",
                                    "lexical_diversity"] if k in f}

    # late fusion ----------------------------------------------------------
    if p_typ is not None and p_txt is not None:
        _, fused = fusion.weighted_average_fusion(p_typ[None], p_txt[None], weight_typing=weight)
        out["fused"] = _result(fused[0], f"late fusion, typing weight {weight:.2f}",
                               meta["typing"]["validated"] and meta["text"]["validated"])
    else:
        only = p_typ if p_typ is not None else p_txt
        out["fused"] = None if only is None else _result(
            only, "single modality (the other needs more input)", False)

    # early fusion ---------------------------------------------------------
    out["early"] = None
    if MODELS.early is not None and trow is not None and n_words >= MIN_WORDS:
        em = meta["early"]
        ling = text_row(text, em["linguistic_features"])
        X = pd.concat([ling.reset_index(drop=True),
                       trow[em["features"]].reset_index(drop=True)], axis=1)
        out["early"] = _result(_aligned(MODELS.early, X)[0], em["source"], em["validated"])

    return jsonify(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765,
                    help="8765 by default: macOS uses 5000 for AirPlay")
    args = ap.parse_args()
    if not MODELS.ready:
        print("No trained demo models in outputs/demo/. Run first:\n"
              "    python train_demo.py\n"
              "(or: python train_demo.py --source synthetic)")
    print(f"Open http://127.0.0.1:{args.port} in your browser.")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except OSError as err:
        sys.exit(f"Could not use port {args.port} ({err}). Another program is using it; "
                 f"try: python app/server.py --port 8766\n"
                 "(the browser extension expects 8765, so free that port for it)")
