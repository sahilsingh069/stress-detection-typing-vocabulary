"""
Nested cross-validation with whole people held out.

Inner folds choose an option using training people only, so trying many options does not
inflate the score on the held-out people. An option's rows can differ from another's (for
example other window lengths): every option predicts rows, which are averaged into `units`
(a person's block), and units are what get scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class Part:
    """One model on one table. `table` rows carry `person` and `unit`; X columns are `columns`."""
    table: pd.DataFrame
    columns: object
    model: object
    weight: float = 1.0


@dataclass
class Option:
    name: str
    parts: list[Part] = field(default_factory=list)


def _part_proba(part: Part, y_by_unit: dict, train_people, test_units, classes) -> pd.DataFrame:
    t = part.table
    fit = t[t["person"].isin(train_people) & t["unit"].isin(y_by_unit)]
    use = t[t["unit"].isin(test_units)]
    model = clone(part.model).fit(fit[part.columns], fit["unit"].map(y_by_unit))
    proba = model.predict_proba(use[part.columns])
    frame = pd.DataFrame(0.0, index=use.index, columns=classes)
    frame[list(model.classes_)] = proba
    frame["unit"] = use["unit"]
    return frame.groupby("unit")[classes].mean().reindex(test_units)


def option_proba(option: Option, y_by_unit, train_people, test_units, classes) -> np.ndarray:
    total, weights = 0.0, 0.0
    for part in option.parts:
        got = _part_proba(part, y_by_unit, train_people, test_units, classes).fillna(1 / len(classes))
        total = total + part.weight * got.to_numpy()
        weights += part.weight
    return total / weights


def _f1(y, proba, classes) -> float:
    return f1_score(y, np.asarray(classes)[proba.argmax(axis=1)], average="macro", zero_division=0)


def nested_proba(options: list[Option], units: pd.DataFrame, labels, seed: int, outer: int = 5, inner: int = 4):
    """units: index unit, column person. Returns (probabilities per unit, options picked)."""
    y = np.asarray(labels)
    names = units.index.to_numpy()
    people = units["person"].to_numpy()
    classes = sorted(set(y))
    out = np.zeros((len(y), len(classes)))
    picks = []
    for tr, te in StratifiedGroupKFold(outer, shuffle=True, random_state=seed).split(names, y, people):
        scores = np.zeros(len(options))
        if len(options) > 1:
            pool = np.zeros((len(options), len(tr), len(classes)))
            cv = StratifiedGroupKFold(inner, shuffle=True, random_state=seed + 1000)
            for a, b in cv.split(names[tr], y[tr], people[tr]):
                known = dict(zip(names[tr][a], y[tr][a]))
                for i, opt in enumerate(options):
                    pool[i, b] = option_proba(opt, known, set(people[tr][a]), names[tr][b], classes)
            scores = np.array([_f1(y[tr], pool[i], classes) for i in range(len(options))])
        best = options[int(scores.argmax())]
        picks.append(best.name)
        out[te] = option_proba(best, dict(zip(names[tr], y[tr])), set(people[tr]), names[te], classes)
    return out, picks
