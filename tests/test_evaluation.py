"""Regression tests for the extended evaluation and the SWELL raw-log parser."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataio import swell_raw
from modeling import extended_eval as ev


class MetricTests(unittest.TestCase):
    def test_perfect_probabilities_score_perfectly(self):
        y = ["a", "b", "a", "b"]
        proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        m = ev.full_metrics(y, proba, ["a", "b"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["mcc"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["chance_f1"], 0.5)

    def test_calibration_error_is_zero_when_confidence_matches_accuracy(self):
        # stated 75% confidence, right 3 times out of 4
        proba = np.array([[0.75, 0.25]] * 4)
        self.assertAlmostEqual(ev.expected_calibration_error(np.array([0, 0, 0, 1]), proba), 0.0)


class BootstrapTests(unittest.TestCase):
    def test_person_level_interval_is_wider_when_people_differ(self):
        # one person always right, one always wrong: rows look stable, people do not
        y = np.array(["a", "b"] * 100)
        good = np.where(y == "a", 0.9, 0.1)
        proba = np.column_stack([good, 1 - good])
        proba[100:] = proba[100:, ::-1]
        groups = np.repeat(["p1", "p2"], 100)
        rows = ev.bootstrap_ci(y, proba, ["a", "b"], None, 200)
        people = ev.bootstrap_ci(y, proba, ["a", "b"], groups, 200)
        self.assertGreater(people["acc_hi"] - people["acc_lo"], rows["acc_hi"] - rows["acc_lo"])


class PermutationTests(unittest.TestCase):
    def test_block_labels_move_as_whole_units(self):
        y = np.array(["N"] * 3 + ["T"] * 3 + ["N"] * 2 + ["T"] * 2)
        groups = np.array(["p1"] * 6 + ["p2"] * 4)
        blocks = np.array(["b1"] * 3 + ["b2"] * 3 + ["c1"] * 2 + ["c2"] * 2)
        for seed in range(20):
            out = ev.permute_labels(y, groups, blocks, np.random.default_rng(seed))
            for b in np.unique(blocks):
                self.assertEqual(len(set(out[blocks == b])), 1)
            for g in np.unique(groups):
                self.assertEqual(sorted(out[groups == g]), sorted(y[groups == g]))

    def test_holm_is_never_smaller_than_raw_and_keeps_order(self):
        raw = np.array([0.01, 0.04, 0.03])
        adj = ev.holm_adjust(raw)
        self.assertTrue((adj >= raw).all())
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[1], 0.06)
        self.assertAlmostEqual(adj[2], 0.06)


class McNemarTests(unittest.TestCase):
    def test_identical_models_do_not_differ(self):
        y = ["a", "b", "a"]
        m = ev.mcnemar(y, ["a", "a", "a"], ["a", "a", "a"])
        self.assertEqual(m["p_value"], 1.0)


class SwellRawTests(unittest.TestCase):
    LOG = """<?xml version="1.0"?><Log>
<Event><EventType>Keyboard</EventType><EventAction>character</EventAction>
<TimeStamp>2012-10-08T11:40:17.0000000Z</TimeStamp><KeyboardValue>a &amp; b</KeyboardValue>
<Control><ControlApplication>WINWORD</ControlApplication></Control></Event>
<Event><EventType>Mouse</EventType><EventAction>clicked</EventAction>
<TimeStamp>2012-10-08T11:40:18.0000000Z</TimeStamp></Event>
</Log>"""

    def test_parser_reads_keyboard_events_only_and_unescapes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a_pp1_c1_uLog.xml"
            path.write_text(self.LOG)
            df = swell_raw.parse_ulog(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "value"], "a & b")
        self.assertEqual(df.loc[0, "app"], "WINWORD")

    def test_rebuild_applies_backspace_and_ignores_pastes(self):
        actions = ["character", "character", "special key", "special key", "character",
                   "key combination"]
        values = ["h", "x", "Backspace", "Space", "i", "Ctrl + V"]
        self.assertEqual(swell_raw.rebuild_text(actions, values), "h i")

    def test_ctrl_backspace_removes_the_last_word(self):
        actions = ["character", "special key", "character", "character", "special key"]
        values = ["a", "Space", "b", "c", "Ctrl + Backspace"]
        self.assertEqual(swell_raw.rebuild_text(actions, values), "a ")


if __name__ == "__main__":
    unittest.main()
