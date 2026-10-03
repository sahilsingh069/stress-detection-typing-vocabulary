"""
Text / vocabulary feature extraction.

Produces two complementary representations (Objective 2 in the synopsis):

  A. Hand-crafted linguistic features - interpretable, panel-friendly:
       sentiment polarity (VADER compound / pos / neg / neu)
       negative-word ratio, stress-word ratio
       lexical diversity (type-token ratio), avg word length,
       first-person pronoun ratio, negation ratio,
       exclamation / question / all-caps ratios

  B. TF-IDF vectors - lets the model pick up specific stress vocabulary.

`build_text_features()` returns the hand-crafted block; the TF-IDF block is
added inside the sklearn Pipeline in modeling.classifiers so it is fitted on
training folds only (no leakage).

VADER is used if the nltk lexicon is available; otherwise a small built-in
lexicon fallback keeps everything runnable offline. The module never attempts
to download resources at import time, so offline runs do not pause or emit
network errors.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# sentiment backend: VADER if available, else a light built-in fallback
# --------------------------------------------------------------------------
_VADER = None
try:  # pragma: no cover
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        _VADER = SentimentIntensityAnalyzer()
    except LookupError:
        _VADER = None
except Exception:
    _VADER = None


NEGATIVE_WORDS = {
    "anxious", "afraid", "angry", "awful", "bad", "broken", "burnout",
    "cannot", "cant", "crying", "depressed", "desperate", "difficult",
    "disappointed", "exhausted", "fail", "failing", "failure", "fear",
    "frustrated", "guilty", "hate", "helpless", "hopeless", "horrible",
    "hurt", "lonely", "lost", "miserable", "nervous", "overwhelmed",
    "pain", "panic", "pressure", "problem", "sad", "scared", "sick",
    "stress", "stressed", "struggle", "struggling", "suffering", "terrible",
    "tired", "trapped", "upset", "worried", "worry", "worse", "worst",
}

STRESS_WORDS = {
    "deadline", "overwhelmed", "pressure", "burnout", "anxiety", "anxious",
    "panic", "exhausted", "insomnia", "workload", "overworked", "tense",
    "stressed", "stress", "breakdown", "sleepless", "racing", "worried",
}

POSITIVE_WORDS = {
    "calm", "relaxed", "happy", "good", "great", "fine", "peaceful",
    "content", "enjoy", "enjoyed", "smooth", "smoothly", "comfortable",
    "confident", "grateful", "pleased", "wonderful", "nice", "easy",
}

NEGATION_WORDS = {"no", "not", "never", "none", "cannot", "cant", "wont",
                  "dont", "didnt", "isnt", "wasnt", "couldnt", "shouldnt"}

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _fallback_sentiment(tokens: list[str]) -> dict:
    if not tokens:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    total = len(tokens)
    compound = (pos - neg) / max(total ** 0.5, 1)
    return {
        "compound": float(np.clip(compound, -1, 1)),
        "pos": pos / total,
        "neg": neg / total,
        "neu": max(0.0, 1 - (pos + neg) / total),
    }


def extract_text_features(text: str) -> dict:
    """Compute all hand-crafted linguistic features for one document."""
    text = str(text)
    tokens = _tokenize(text)
    n = len(tokens)

    if _VADER is not None:
        s = _VADER.polarity_scores(text)
    else:
        s = _fallback_sentiment(tokens)

    if n == 0:
        return {
            "sent_compound": 0.0, "sent_pos": 0.0, "sent_neg": 0.0,
            "sent_neu": 1.0, "negative_word_ratio": 0.0,
            "stress_word_ratio": 0.0, "positive_word_ratio": 0.0,
            "negation_ratio": 0.0, "first_person_ratio": 0.0,
            "lexical_diversity": 0.0, "avg_word_len": 0.0,
            "word_count": 0, "exclamation_ratio": 0.0,
            "question_ratio": 0.0, "caps_ratio": 0.0,
        }

    return {
        "sent_compound": s["compound"],
        "sent_pos": s["pos"],
        "sent_neg": s["neg"],
        "sent_neu": s["neu"],
        "negative_word_ratio": sum(t in NEGATIVE_WORDS for t in tokens) / n,
        "stress_word_ratio": sum(t in STRESS_WORDS for t in tokens) / n,
        "positive_word_ratio": sum(t in POSITIVE_WORDS for t in tokens) / n,
        "negation_ratio": sum(t in NEGATION_WORDS for t in tokens) / n,
        "first_person_ratio": sum(t in FIRST_PERSON for t in tokens) / n,
        "lexical_diversity": len(set(tokens)) / n,
        "avg_word_len": float(np.mean([len(t) for t in tokens])),
        "word_count": n,
        "exclamation_ratio": text.count("!") / max(len(text), 1),
        "question_ratio": text.count("?") / max(len(text), 1),
        "caps_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
    }


def build_text_features(texts: pd.Series) -> pd.DataFrame:
    """Vectorised wrapper: Series[str] -> DataFrame of linguistic features."""
    return pd.DataFrame([extract_text_features(t) for t in texts])


def sentiment_backend() -> str:
    return "nltk-vader" if _VADER is not None else "builtin-lexicon"
