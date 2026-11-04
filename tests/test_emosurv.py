"""Regression tests for the EmoSurv loader: each one guards a way the labels could leak."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
from dataio import emosurv

HEADER = "userId;emotionIndex;index;keyCode;keyDown;keyUp;D1U1;D1U2;D1D2;U1D2;U1U2;D1U3;D1D3;answer\n"


def _line(user, emotion, index, key, up="1,58E+12", dwell="90", gap="150", answer=""):
    return f"{user};{emotion};{index};{key};1,58E+12;{up};{dwell};0;{gap};0;0;0;300;{answer}\n"


def _write(folder: Path, lines: list[str]) -> None:
    (folder / emosurv.FILES["fixed"]).write_text(HEADER + "".join(lines))


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        patcher = mock.patch.object(config, "DATA_DIR", self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_corrupted_timing_is_missing_not_huge(self):
        _write(self.folder, [_line(1, "N", 1, "a", gap="-1,58E+12", dwell="1,58E+12")])
        keys = emosurv.load_keys("fixed")
        self.assertTrue(np.isnan(keys["gap_ms"].iloc[0]))
        self.assertTrue(np.isnan(keys["dwell_ms"].iloc[0]))

    def test_answer_column_is_never_loaded(self):
        _write(self.folder, [_line(1, "H", 1, "a", answer="R")])
        self.assertNotIn("answer", emosurv.load_keys("fixed").columns)

    def test_auto_repeat_is_not_a_press_but_keeps_its_time(self):
        _write(self.folder, [_line(1, "N", 1, "a", gap="100"), _line(1, "N", 2, "\\b", up="", gap="40"),
                             _line(1, "N", 3, "b", gap="100")])
        keys = emosurv.load_keys("fixed")
        self.assertEqual(keys["is_press"].tolist(), [True, False, True])
        feats = emosurv.block_features(keys)
        self.assertEqual(feats["backspace_rate"], 0.0)
        self.assertEqual(feats["gap_mean"], 140.0)


class TextTests(unittest.TestCase):
    def test_backspace_and_stored_punctuation(self):
        self.assertEqual(emosurv.rebuild_text(pd.Series(["h", "i", "x", "backspace", "¾", "shift"])), "hi.")


class BaselineTests(unittest.TestCase):
    def _blocks(self):
        return pd.DataFrame({"participant_id": ["a", "a", "b", "c"], "emotion": ["N", "A", "N", "C"],
                             "text_type": "free", "text": "x", "gap_mean": [100.0, 130.0, 90.0, 80.0]})

    def test_neutral_rows_cannot_reach_a_model(self):
        # they would all be exactly zero, which gives the label away
        self.assertNotIn("N", set(emosurv.from_baseline(self._blocks())["emotion"]))

    def test_subtracts_the_persons_own_neutral(self):
        self.assertEqual(emosurv.from_baseline(self._blocks())["gap_mean"].tolist(), [30.0])

    def test_person_without_neutral_is_dropped(self):
        self.assertNotIn("c", set(emosurv.from_baseline(self._blocks())["participant_id"]))

    def test_features_exclude_label_identity_and_text(self):
        cols = emosurv.feature_columns(self._blocks())
        self.assertEqual(cols, ["gap_mean"])


@unittest.skipUnless(emosurv.available(), "EmoSurv files not in data/")
class RealDataTests(unittest.TestCase):
    def test_fixed_passages_differ_by_emotion(self):
        # the reason fixed text is only reported as confounded
        blocks = emosurv.load_blocks("fixed")
        openings = blocks.groupby("emotion")["text"].agg(lambda s: s.str[:8].mode().iloc[0])
        self.assertEqual(openings.nunique(), len(openings))


if __name__ == "__main__":
    unittest.main()
