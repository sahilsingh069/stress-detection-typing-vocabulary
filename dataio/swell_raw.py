"""
SWELL-KW raw computer-interaction logs (uLog XML), as a PAIRED dataset.

Each log holds one participant's block with every keystroke: its timestamp, the
character or key, and the application it went to. That gives both modalities
from the same people at the same moment, under an experimental condition:
key timing for the typing stream, and the words (rebuilt from the same keys)
for the text stream.

Limits compared with our own collector: uLog logs key entry only, so there is
no dwell time; timestamps tick in about 15.6 ms steps; labels are conditions;
and the neutral block always came first, so N-vs-stressor also compares early
with late in the session (block 2 vs. block 3 is counterbalanced).

Choices fixed before looking at any model result:
  - Word only: PowerPoint is nearly absent from the neutral block and e-mail is
    specific to the interruption block, so other apps would reveal the condition
  - 5-minute windows (config.SESSION_WINDOW_SECONDS, same as the Kaggle data)
  - at least 100 key presses and 20 words per window
  - English writers only: the sentiment and word lists are English

    python -m dataio.swell_raw      # download the logs (~300 MB, open licence)

Source: Koldijk et al. (2014), DANS doi:10.17026/dans-x55-69zp, CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import json
import re
import subprocess
from html import unescape
from pathlib import Path

import pandas as pd

import config

API = "https://ssh.datastations.nl/api"
DOI = "doi:10.17026/dans-x55-69zp"
ULOG_FOLDER = "0 - Raw data/A - anonimized Computer interaction - raw data uLog"
ULOG_DIR = config.DATA_DIR / "swell_raw" / "ulog"
KEYS_CACHE = config.DATA_DIR / "swell_raw" / "ulog_keystrokes.csv.gz"
QUESTIONNAIRE = config.DATA_DIR / "swell_raw" / "questionnaire_blocks.tab"
QUESTIONNAIRE_FILE = "Questionnaire Blok Results_final - performance added.tab"

MIN_WINDOW_PRESSES = 100
MIN_WINDOW_WORDS = 20
_DUTCH = set("de het een van en is dat op te zijn voor met niet ook aan".split())
_ENGLISH = set("the of and is to in that for with it are on as".split())

_EVENT = re.compile(r"<Event>(.*?)</Event>", re.S)
_FIELD = lambda tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.S)
_TYPE, _ACTION = _FIELD("EventType"), _FIELD("EventAction")
_TIME, _VALUE = _FIELD("TimeStamp"), _FIELD("KeyboardValue")
_APP = _FIELD("ControlApplication")
_NAME = re.compile(r"a_pp(\d+)_c(\d)_", re.I)


def download(dest: Path = ULOG_DIR) -> int:
    """Fetch every uLog XML that is not already on disk. Returns files fetched."""
    dest.mkdir(parents=True, exist_ok=True)
    # curl rather than urllib: python.org builds on macOS often lack CA certificates
    listing = subprocess.run(
        ["curl", "-sSfL", f"{API}/datasets/:persistentId/?persistentId={DOI}"],
        check=True, capture_output=True,
    ).stdout
    files = json.loads(listing)["data"]["latestVersion"]["files"]
    fetched = 0
    for f in files:
        df = f["dataFile"]
        if df["filename"] == QUESTIONNAIRE_FILE:
            subprocess.run(["curl", "-sSfL", "-o", str(QUESTIONNAIRE),
                            f"{API}/access/datafile/{df['id']}"], check=True)
            continue
        if f.get("directoryLabel") != ULOG_FOLDER or not df["filename"].endswith(".xml"):
            continue
        target = dest / df["filename"]
        if target.exists() and target.stat().st_size == df.get("filesize"):
            continue
        subprocess.run(["curl", "-sSfL", "-o", str(target),
                        f"{API}/access/datafile/{df['id']}"], check=True)
        fetched += 1
        print(f"  {df['filename']}")
    return fetched


def parse_ulog(path: Path) -> pd.DataFrame:
    """All keyboard events of one log: timestamp, action, value, application."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for ev in _EVENT.findall(raw):
        kind = _TYPE.search(ev)
        if not kind or kind.group(1) != "Keyboard":
            continue
        t, action, value = _TIME.search(ev), _ACTION.search(ev), _VALUE.search(ev)
        if not (t and action and value):
            continue
        app = _APP.search(ev)
        rows.append((t.group(1), action.group(1), unescape(value.group(1)),
                     app.group(1) if app else ""))
    return pd.DataFrame(rows, columns=["timestamp", "action", "value", "app"])


def load_keystrokes(root: Path = ULOG_DIR, refresh: bool = False) -> pd.DataFrame:
    """Keyboard events of every log, with participant and block, cached on disk."""
    if KEYS_CACHE.exists() and not refresh:
        return pd.read_csv(KEYS_CACHE, keep_default_na=False, parse_dates=["timestamp"])
    frames = []
    for path in sorted(Path(root).glob("*.xml")):
        m = _NAME.search(path.name)
        if not m:
            continue
        df = parse_ulog(path)
        df["participant_id"] = f"PP{int(m.group(1))}"
        df["block"] = int(m.group(2))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No uLog files in {root}. Run: python -m dataio.swell_raw")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], format="ISO8601", utc=True).dt.tz_localize(None)
    out = out.sort_values(["participant_id", "block", "timestamp"]).reset_index(drop=True)
    out.to_csv(KEYS_CACHE, index=False)
    return out


def load_questionnaire() -> pd.DataFrame:
    """Per-block self-ratings and topics. Never a model input: a check on the labels."""
    if not QUESTIONNAIRE.exists():
        raise FileNotFoundError(f"{QUESTIONNAIRE} missing. Run: python -m dataio.swell_raw")
    df = pd.read_csv(QUESTIONNAIRE, sep="\t")
    # Participants are numbered here; SP and MP are the experiment's two pilots.
    df = df[df["PP"].astype(str).str.fullmatch(r"\d+")].copy()
    df["PP"] = "PP" + df["PP"].astype(str)
    df["Blok"] = df["Blok"].astype(int)
    for col in ["Stress", "TemporalDemand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _block_conditions() -> dict:
    df = pd.read_csv(config.SWELL_PATH)[["PP", "Blok", "Condition"]]
    df = df[df["Condition"] != "R"].drop_duplicates(["PP", "Blok"])
    return {(p, int(b)): c for p, b, c in zip(df["PP"], df["Blok"], df["Condition"])}


def _key_name(action: str, value: str) -> str:
    """uLog key names -> the names features.typing_features expects."""
    if action == "key combination":
        return "ctrl_l"
    base = value.split(" + ")[-1]
    return {
        "Space": " ", "Return": "enter", "Backspace": "backspace", "Delete": "delete",
        "Capital": "caps_lock", "Tab": "tab", "Escape": "esc",
        "Left arrow": "arrow_key", "Right arrow": "arrow_key",
        "Up arrow": "arrow_key", "Down arrow": "arrow_key",
    }.get(base, value.lower() if action == "special key" else value)


def rebuild_text(actions, values) -> str:
    """The text typed, from the keys alone. Pastes are not typing, so not included."""
    out: list[str] = []
    for action, value in zip(actions, values):
        if action == "character":
            out.append(value)
        elif value.endswith("Space") and action != "key combination":
            out.append(" ")
        elif value == "Return":
            out.append("\n")
        elif value == "Backspace":
            if out:
                out.pop()
        elif value == "Ctrl + Backspace":
            while out and out[-1].isspace():
                out.pop()
            while out and not out[-1].isspace():
                out.pop()
    return "".join(out)


def _language(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    nl = sum(w in _DUTCH for w in words)
    return "nl" if nl > sum(w in _ENGLISH for w in words) else "en"


def load_paired(english_only: bool = True) -> pd.DataFrame:
    """
    One row per 5-minute Word window: participant_id, session_id, block,
    condition, task_type, text, stress_label and the typing features. The same
    shape as loaders.build_paired_frame, so run_paired's experiment accepts it.
    """
    from features.typing_features import features_per_session

    keys = windowed_word_keys()
    feats = features_per_session(keys[["session_id", "timestamp", "key", "event"]],
                                 min_presses=MIN_WINDOW_PRESSES)
    feats = feats.drop(columns=["stress_label"], errors="ignore")

    meta = []
    for sid, grp in keys.groupby("session_id"):
        meta.append({
            "session_id": sid,
            "participant_id": grp["participant_id"].iloc[0],
            "block": int(grp["block"].iloc[0]),
            "condition": grp["condition"].iloc[0],
            "text": rebuild_text(grp["action"], grp["value"]),
        })
    meta = pd.DataFrame(meta)
    out = meta.merge(feats, on="session_id", how="inner")
    out = out[out["text"].str.split().str.len() >= MIN_WINDOW_WORDS].copy()

    full = out.groupby("participant_id")["text"].agg(" ".join).map(_language)
    out["language"] = out["participant_id"].map(full)
    if english_only:
        out = out[out["language"] == "en"]
    out["task_type"] = "free"
    out["stress_label"] = out["condition"].str.lower().map(config.SWELL_CONDITION_MAP)
    return out.reset_index(drop=True)


def windowed_word_keys() -> pd.DataFrame:
    """Word keystrokes with condition, 5-minute session_id and normalised key name."""
    keys = load_keystrokes()
    keys = keys[keys["app"] == "WINWORD"].copy()
    # uLog records every Space twice: the key, then a whitespace "character".
    keys = keys[~((keys["action"] == "character") & keys["value"].str.fullmatch(r"\s*"))]

    conditions = _block_conditions()
    keys["condition"] = [conditions.get((p, b)) for p, b in zip(keys["participant_id"], keys["block"])]
    keys = keys.dropna(subset=["condition"])

    start = keys.groupby(["participant_id", "block"])["timestamp"].transform("min")
    window = ((keys["timestamp"] - start).dt.total_seconds()
              // config.SESSION_WINDOW_SECONDS).astype(int)
    keys["session_id"] = (keys["participant_id"] + "_B" + keys["block"].astype(str)
                          + "_W" + window.astype(str))
    keys["event"] = "press"
    keys["key"] = [_key_name(a, v) for a, v in zip(keys["action"], keys["value"])]
    return keys


if __name__ == "__main__":
    print(f"Downloading SWELL-KW raw logs to {ULOG_DIR} ...")
    print(f"{download()} file(s) downloaded.")
