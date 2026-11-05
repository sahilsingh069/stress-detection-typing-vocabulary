"""
A typing model for what the extension actually receives: short bursts, not 5-minute windows.

    python run_burst_model.py

Trains on SWELL-KW Word keystrokes split at pauses, like live typing. Scored per burst with
whole people held out, against the 5-minute model. EmoSurv's bursts, from other people on
other keyboards, check whether the model leans to one level on new data.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataio import emosurv, swell_raw
from features.burst_features import FEATURES, burst_features, split_bursts

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR / "burst_model"
LABEL = {"N": "Low", "I": "Medium", "T": "High"}


def swell_bursts() -> pd.DataFrame:
    rows = []
    keys = swell_raw.windowed_word_keys().sort_values(["participant_id", "block", "timestamp"])
    for (person, block), g in keys.groupby(["participant_id", "block"]):
        t = g["timestamp"].astype("int64").to_numpy() / 1e6
        if pd.api.types.is_datetime64_any_dtype(g["timestamp"]):
            t = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy() * 1000
        back = (g["key"] == "backspace").to_numpy()
        for a, b in split_bursts(t):
            rows.append({"person": person, "block": block, "label": LABEL[g["condition"].iloc[0]], "keys": b - a,
                         **burst_features(t[a:b], back[a:b])})
    return pd.DataFrame(rows)


def emosurv_bursts() -> pd.DataFrame:
    rows = []
    keys = emosurv.load_keys("free")
    for (person, state), g in keys.groupby(["participant_id", "emotion"]):
        clock = g["gap_ms"].fillna(0).cumsum().shift(fill_value=0).to_numpy()
        press = g["is_press"].to_numpy()
        t, back = clock[press], (g["key"] == "backspace").to_numpy()[press]
        for a, b in split_bursts(t):
            rows.append({"person": person, "state": state, "keys": b - a, **burst_features(t[a:b], back[a:b])})
    return pd.DataFrame(rows)


def model():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced"))])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = swell_bursts()
    print(d.groupby("label").size().to_dict(), "bursts", len(d), "people", d.person.nunique(), "median keys", d["keys"].median())
    rows, classes = [], ["High", "Low", "Medium"]
    for task, keep in [("3-class", ["Low", "Medium", "High"]), ("neutral vs time pressure", ["Low", "High"])]:
        s = d[d["label"].isin(keep)].reset_index(drop=True)
        scores, accs = [], []
        for seed in range(5):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            pred = cross_val_predict(model(), s[FEATURES], s["label"], groups=s["person"], cv=cv)
            scores.append(f1_score(s["label"], pred, average="macro"))
            accs.append(accuracy_score(s["label"], pred))
        # a block's verdict from all its bursts, the way the dashboard summarises
        proba = cross_val_predict(model(), s[FEATURES], s["label"], groups=s["person"],
                                  cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), method="predict_proba")
        order = sorted(keep)
        block = pd.DataFrame(proba, columns=order).assign(u=s["person"] + s["block"].astype(str), y=s["label"]).groupby("u")
        verdict = block[order].mean().idxmax(axis=1)
        rows.append({"task": task, "n_bursts": len(s), "burst_macro_f1": np.mean(scores), "sd": np.std(scores),
                     "burst_accuracy": np.mean(accs), "majority": s["label"].value_counts(normalize=True).max(),
                     "block_macro_f1": f1_score(block["y"].first(), verdict, average="macro"), "chance": 1 / len(keep),
                     "share_pred_on_swell": pd.Series(pred).value_counts(normalize=True).round(3).to_dict()})
    final = model().fit(d[FEATURES], d["label"])
    em = emosurv_bursts()
    p = pd.DataFrame(final.predict_proba(em[FEATURES]), columns=final.classes_)
    lean = p.idxmax(axis=1).value_counts(normalize=True).round(3).to_dict()
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "results.csv", index=False)
    print(table.to_string(index=False)); print("on EmoSurv bursts", len(em), lean, "mean probs", p.mean().round(3).to_dict(), "max conf p90", round(p.max(axis=1).quantile(.9), 3))
    print("feature medians swell vs emosurv\n", pd.DataFrame({"swell": d[FEATURES].median(), "emosurv": em[FEATURES].median()}).round(3))
    coefs = dict(zip(FEATURES, final.named_steps["m"].coef_.round(3).T.tolist()))
    (OUT / "summary.json").write_text(json.dumps({"lean_on_emosurv": lean, "classes": list(final.classes_), "coef": coefs}, indent=1))


if __name__ == "__main__":
    main()
