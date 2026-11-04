"""
Extended evaluation: threshold-free metrics, calibration, confidence intervals,
significance tests, stability and learning curves.

Every function takes out-of-fold predictions or a pipeline plus participant
groups, so whole people stay held out wherever they are identifiable.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

import config

SERIES = {"LogisticRegression": "#2a78d6", "RandomForest": "#eb6834", "SVM": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def make_cv(y, groups=None, seed: int | None = None, folds: int | None = None):
    """Whole groups held out when groups are given, else plain stratified folds."""
    seed = config.RANDOM_STATE if seed is None else seed
    folds = folds or config.GROUP_CV_FOLDS
    if groups is None:
        return StratifiedKFold(folds, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(min(folds, len(np.unique(groups))), shuffle=True,
                                random_state=seed)


def oof_proba(pipe, X, y, groups=None, seed=None) -> np.ndarray:
    """Out-of-fold probabilities, columns in sorted class order."""
    return cross_val_predict(pipe, X, y, groups=groups, cv=make_cv(y, groups, seed),
                             method="predict_proba")


def expected_calibration_error(y_idx, proba, bins: int = 10) -> float:
    """Average gap between stated confidence and actual accuracy."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)


def _auc(y_idx, proba) -> float:
    if proba.shape[1] == 2:
        return roc_auc_score(y_idx, proba[:, 1])
    return roc_auc_score(y_idx, proba, multi_class="ovr", average="macro",
                         labels=np.arange(proba.shape[1]))


def full_metrics(y, proba, classes) -> dict:
    classes = list(classes)
    y_idx = np.array([classes.index(str(v)) for v in y])
    pred = proba.argmax(axis=1)
    onehot = np.eye(len(classes))[y_idx]
    ap = np.mean([average_precision_score(onehot[:, i], proba[:, i])
                  for i in range(len(classes))])
    p = np.clip(proba, 1e-12, 1)
    return {
        "accuracy": accuracy_score(y_idx, pred),
        "balanced_accuracy": balanced_accuracy_score(y_idx, pred),
        "precision_macro": precision_score(y_idx, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_idx, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_idx, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_idx, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_idx, pred),
        "cohen_kappa": cohen_kappa_score(y_idx, pred),
        "roc_auc": _auc(y_idx, proba),
        "pr_auc": ap,
        "log_loss": log_loss(y_idx, p / p.sum(axis=1, keepdims=True),
                             labels=np.arange(len(classes))),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "calibration_error": expected_calibration_error(y_idx, proba),
        "chance_f1": 1 / len(classes),
        "majority_acc": float(pd.Series(y_idx).value_counts(normalize=True).max()),
        "n": len(y_idx),
    }


def per_class_table(y, proba, classes) -> pd.DataFrame:
    classes = list(classes)
    y_idx = np.array([classes.index(str(v)) for v in y])
    p, r, f, s = precision_recall_fscore_support(
        y_idx, proba.argmax(axis=1), labels=np.arange(len(classes)), zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})


# --------------------------------------------------------------------------
# uncertainty and significance
# --------------------------------------------------------------------------
def _resamples(n: int, groups, n_boot: int, seed: int):
    """Row indices for each bootstrap. Whole groups are resampled when given:
    rows from one person are not independent, so resampling rows alone would
    make the interval too narrow."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_group), len(by_group))
        yield np.concatenate([by_group[i] for i in pick])


def bootstrap_ci(y, proba, classes, groups=None, n_boot: int = 500, seed: int = 42) -> dict:
    """95% percentile intervals for accuracy, macro F1 and ROC-AUC."""
    classes = list(classes)
    y_idx = np.array([classes.index(str(v)) for v in y])
    groups = None if groups is None else np.asarray(groups)
    acc, f1, auc = [], [], []
    for idx in _resamples(len(y_idx), groups, n_boot, seed):
        yi, pr = y_idx[idx], proba[idx]
        if len(np.unique(yi)) < len(classes):
            continue
        pred = pr.argmax(axis=1)
        acc.append((pred == yi).mean())
        f1.append(f1_score(yi, pred, average="macro", zero_division=0))
        auc.append(_auc(yi, pr))
    q = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    (al, ah), (fl, fh), (rl, rh) = q(acc), q(f1), q(auc)
    return {"acc_lo": al, "acc_hi": ah, "f1_lo": fl, "f1_hi": fh, "auc_lo": rl, "auc_hi": rh}


def bootstrap_f1_difference(y, pred_a, pred_b, groups=None, n_boot: int = 1000, seed: int = 42) -> dict:
    """macro F1 of A minus B on the same samples, with a paired 95% interval."""
    y = np.asarray(y).astype(str)
    pred_a, pred_b = np.asarray(pred_a).astype(str), np.asarray(pred_b).astype(str)
    groups = None if groups is None else np.asarray(groups)
    score = lambda t, p: f1_score(t, p, average="macro", zero_division=0)
    diffs = [score(y[i], pred_a[i]) - score(y[i], pred_b[i])
             for i in _resamples(len(y), groups, n_boot, seed)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": score(y, pred_a) - score(y, pred_b), "lo": lo, "hi": hi}


def permute_labels(y, groups=None, blocks=None, rng=None) -> np.ndarray:
    """
    Labels under "no real relation". With blocks, labels move between a person's
    blocks as whole units: every window of a block shares its label, so shuffling
    windows singly would break that structure and make the test too easy to pass.
    """
    rng = rng or np.random.default_rng()
    y = np.asarray(y).astype(object)
    if blocks is None:
        return rng.permutation(y)
    out = y.copy()
    groups, blocks = np.asarray(groups), np.asarray(blocks)
    for g in np.unique(groups):
        in_g = groups == g
        names = np.unique(blocks[in_g])
        labels = [y[in_g & (blocks == b)][0] for b in names]
        for b, new in zip(names, rng.permutation(labels)):
            out[in_g & (blocks == b)] = new
    return out


def permutation_test(pipe, X, y, groups=None, blocks=None, n_perm: int = 200, seed: int = 42) -> dict:
    """Is the macro F1 better than the same model on shuffled labels?"""
    y = pd.Series(np.asarray(y).astype(str))

    def score(labels):
        pred = cross_val_predict(clone(pipe), X, labels, groups=groups,
                                 cv=make_cv(labels, groups, seed))
        return f1_score(labels, pred, average="macro", zero_division=0)

    observed = score(y)
    seeds = np.random.SeedSequence(seed).spawn(n_perm)
    null = Parallel(n_jobs=-1)(
        delayed(score)(pd.Series(permute_labels(y, groups, blocks, np.random.default_rng(s))))
        for s in seeds)
    null = np.array(null)
    return {"observed_f1": observed, "null_mean_f1": null.mean(),
            "null_95th": np.percentile(null, 95),
            "p_value": (1 + (null >= observed).sum()) / (n_perm + 1), "n_perm": n_perm}


def holm_adjust(pvals) -> np.ndarray:
    """Holm correction: running many tests makes a lucky p < 0.05 more likely."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar: do two models differ on the samples where they disagree?"""
    y, a, b = (np.asarray(v).astype(str) for v in (y, pred_a, pred_b))
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    total = only_a + only_b
    p = binomtest(only_a, total, 0.5).pvalue if total else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def repeated_cv(pipe, X, y, groups=None, seeds=range(5)) -> dict:
    """Does the score depend on which people happened to share a fold?"""
    y = pd.Series(np.asarray(y).astype(str))
    scores = []
    for seed in seeds:
        pred = cross_val_predict(clone(pipe), X, y, groups=groups, cv=make_cv(y, groups, seed))
        scores.append(f1_score(y, pred, average="macro", zero_division=0))
    return {"f1_mean": float(np.mean(scores)), "f1_sd": float(np.std(scores)),
            "f1_min": float(np.min(scores)), "f1_max": float(np.max(scores)),
            "repeats": len(scores)}


def learning_curve_people(pipe, X, y, groups, sizes, n_test: int = 5, repeats: int = 30,
                          seed: int = 42) -> pd.DataFrame:
    """Macro F1 on unseen people as the number of training people grows."""
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups)
    people = np.unique(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for k in sizes:
        if k + n_test > len(people):
            continue
        scores = []
        for _ in range(repeats):
            chosen = rng.permutation(people)
            test, train = chosen[:n_test], chosen[n_test:n_test + k]
            tr, te = np.isin(groups, train), np.isin(groups, test)
            if len(np.unique(y[tr])) < len(np.unique(y)):
                continue
            model = clone(pipe).fit(X[tr], y[tr])
            scores.append(f1_score(y[te], model.predict(X[te]), average="macro", zero_division=0))
        rows.append({"train_people": k, "f1_mean": np.mean(scores), "f1_sd": np.std(scores),
                     "repeats": len(scores)})
    return pd.DataFrame(rows)


def per_person_accuracy(y, proba, classes, groups) -> pd.DataFrame:
    classes = list(classes)
    y_idx = np.array([classes.index(str(v)) for v in y])
    hit = proba.argmax(axis=1) == y_idx
    return (pd.DataFrame({"person": np.asarray(groups), "hit": hit}).groupby("person")["hit"]
            .agg(accuracy="mean", n="size").reset_index())


# --------------------------------------------------------------------------
# charts (static PNG for the report)
# --------------------------------------------------------------------------
def _axes(size=(5.2, 4.2)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel, path):
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 1:
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(curves: dict, title: str, path) -> None:
    """curves: model name -> (y_binary, positive-class probability)."""
    fig, ax = _axes()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), label="Chance")
    for name, (yb, score) in curves.items():
        fpr, tpr, _ = roc_curve(yb, score)
        ax.plot(fpr, tpr, color=SERIES.get(name, "#6250d6"), linewidth=2,
                label=f"{name} (AUC {roc_auc_score(yb, score):.2f})")
    _finish(fig, ax, title, "False positive rate", "True positive rate", path)


def plot_calibration(y_idx, proba, title: str, path, bins: int = 10) -> None:
    """Stated confidence of the top class against how often it was right."""
    conf, hit = proba.max(axis=1), proba.argmax(axis=1) == y_idx
    edges = np.linspace(conf.min(), 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(hit[m].mean()); ns.append(m.sum())
    fig, ax = _axes()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), label="Perfect calibration")
    ax.plot(xs, ys, color=SERIES["LogisticRegression"], linewidth=2, marker="o", markersize=5,
            label="Logistic Regression")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _finish(fig, ax, title, "Stated confidence", "Actual accuracy", path)


def plot_confusion_normalised(y_idx, proba, classes, title: str, path) -> None:
    """Row-normalised, so each row reads as the recall of that true class."""
    cm = confusion_matrix(y_idx, proba.argmax(axis=1), labels=np.arange(len(classes)))
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f7fb", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(4.6, 4), facecolor=SURFACE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, color=MUTED)
    ax.set_yticks(range(len(classes)), classes, color=MUTED)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if share[i, j] > 0.6 else INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curve(table: pd.DataFrame, chance: float, title: str, path) -> None:
    fig, ax = _axes()
    x, m, s = table["train_people"], table["f1_mean"], table["f1_sd"]
    ax.fill_between(x, m - s, m + s, color=SERIES["LogisticRegression"], alpha=0.15, linewidth=0)
    ax.plot(x, m, color=SERIES["LogisticRegression"], linewidth=2, marker="o", markersize=5,
            label="Logistic Regression (band: 1 SD)")
    ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), label="Chance")
    ax.set_xticks(list(x))
    ax.set_ylim(min(chance, (m - s).min()) - 0.06, (m + s).max() + 0.01)   # room for the legend
    _finish(fig, ax, title, "People in the training set", "Macro F1 on unseen people", path)
