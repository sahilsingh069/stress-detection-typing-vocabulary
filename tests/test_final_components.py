"""Regression tests for the SWELL loader, paired data, grouped evaluation,
early fusion and the demo server."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config
from dataio import loaders
from features.typing_features import features_per_session
from modeling import classifiers, evaluation


def _events(rows):
    df = pd.DataFrame(rows, columns=["session_id", "event", "key", "code", "t"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    return df


class DwellPairingTests(unittest.TestCase):
    def test_code_pairs_press_and_release_when_character_changes(self):
        rows = []
        for i in range(12):
            t = i * 200
            rows += [("s1", "press", "1", "Digit1", t), ("s1", "release", "!", "Digit1", t + 50)]
        feats = features_per_session(_events(rows), min_presses=10)
        self.assertEqual(len(feats), 1)
        self.assertAlmostEqual(feats.iloc[0]["mean_dwell_ms"], 50.0, places=3)

    def test_stroke_id_gives_exact_dwell_for_masked_overlapping_keys(self):
        # every key is "$" and each press happens before the previous release
        rows = []
        for i in range(12):
            t = i * 60
            rows += [("s1", "press", "$", i, t), ("s1", "release", "$", i, t + 100)]
        df = _events(rows).rename(columns={"code": "stroke_id"})
        feats = features_per_session(df, min_presses=10)
        self.assertAlmostEqual(feats.iloc[0]["mean_dwell_ms"], 100.0, places=3)

    def test_short_sessions_are_dropped(self):
        rows = [("s1", "press", "a", "KeyA", i * 100) for i in range(5)]
        self.assertTrue(features_per_session(_events(rows), min_presses=10).empty)


class SwellLoaderTests(unittest.TestCase):
    def test_questionnaire_columns_never_become_features(self):
        df = loaders.load_swell(frame=loaders.load_synthetic_swell_raw(n_participants=3, minutes=5))
        self.assertNotIn("Stress", df.columns)
        self.assertIn("participant", df.columns)
        self.assertTrue(all(c.startswith("Sn") for c in df.columns
                            if c not in ("participant", "stress_label")))
        self.assertEqual(set(df["stress_label"]), {"Low", "Medium", "High"})

    def test_rest_periods_are_dropped(self):
        raw = loaders.load_synthetic_swell_raw(n_participants=2, minutes=3)
        df = loaders.load_swell(frame=raw)
        self.assertEqual(len(df), (raw["Condition"] != "R").sum())

    def test_heart_rate_only_file_is_rejected_with_explanation(self):
        hrv = pd.DataFrame({"MEAN_RR": [800.0, 810.0], "RMSSD": [20.0, 21.0],
                            "condition": ["no stress", "time pressure"]})
        with self.assertRaisesRegex(ValueError, "heart-rate"):
            loaders.load_swell(frame=hrv)


class PairedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = loaders.load_synthetic_paired_frame(n_participants=6)

    def test_paired_frame_has_text_label_and_typing_features(self):
        for col in ["participant_id", "session_id", "task_type", "text", "stress_label",
                    *config.OWN_TYPING_FEATURES]:
            self.assertIn(col, self.frame.columns)

    def test_typing_allow_list_excludes_leaky_columns(self):
        leaky = {"n_keystrokes", "duration_s", "condition", "block", "time_limit_s",
                 "self_report_intensity", "self_report_pressure", "word_count"}
        self.assertFalse(leaky & set(config.OWN_TYPING_FEATURES))

    def test_collector_csv_round_trip_keeps_literal_na_text(self):
        sessions, keys = loaders.load_synthetic_paired(n_participants=1)
        sessions.loc[0, "typed_text"] = "NA"
        with tempfile.TemporaryDirectory() as d:
            sessions.to_csv(Path(d) / "sessions_x.csv", index=False)
            keys.to_csv(Path(d) / "keystrokes_x.csv", index=False)
            loaded = loaders.load_own_sessions(d)
            paired = loaders.load_own_paired(d)
        self.assertEqual(loaded.loc[0, "text"], "NA")
        self.assertGreater(len(paired), 0)


class GroupedEvaluationTests(unittest.TestCase):
    def test_no_participant_is_in_train_and_test(self):
        y = pd.Series(["Low", "High"] * 20)
        groups = np.repeat([f"p{i}" for i in range(8)], 5)
        cv = evaluation.group_cv(y, groups)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_single_participant_cannot_be_group_evaluated(self):
        self.assertIsNone(evaluation.group_cv(pd.Series(["Low", "High"]), ["p1", "p1"]))


class EarlyFusionTests(unittest.TestCase):
    def test_early_fusion_pipeline_trains_on_both_modalities(self):
        import run_paired

        df = loaders.load_synthetic_paired_frame(n_participants=6)
        df = df[df["task_type"] == "chat"].reset_index(drop=True)
        X, ling_cols, typing_cols = run_paired.build_matrix(df)
        self.assertNotIn("word_count", ling_cols)
        pipe = classifiers.build_early_fusion_pipeline(
            classifiers.get_estimators()["LogisticRegression"], ling_cols, typing_cols)
        pipe.fit(X, df["stress_label"])
        proba = pipe.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), df["stress_label"].nunique()))

    def test_paired_experiment_reports_every_mode(self):
        import run_paired

        df = loaders.load_synthetic_paired_frame(n_participants=6)
        with tempfile.TemporaryDirectory() as d:
            table = run_paired.run_experiment(df, 0.5, Path(d))
        modes = set(table["mode"])
        for m in ["typing_only", "text_only", "early_fusion", "late_fusion_avg(w=0.5)",
                  "late_fusion_maxconf", "stacking", "typing_only_all_tasks"]:
            self.assertIn(m, modes)


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old_demo = config.DEMO_DIR
        config.DEMO_DIR = Path(cls.tmp.name)
        import train_demo
        train_demo.main("synthetic")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
        cls.server = importlib.import_module("server")
        cls.server.MODELS = cls.server.Models(config.DEMO_DIR)
        cls.client = cls.server.app.test_client()

    @classmethod
    def tearDownClass(cls):
        config.DEMO_DIR = cls._old_demo
        cls.tmp.cleanup()

    def _events(self, text):
        out, t = [], 0.0
        for ch in text:
            t += 150
            code = "Space" if ch == " " else f"Key{ch.upper()}"
            out += [{"ev": "down", "key": ch, "code": code, "t": t},
                    {"ev": "up", "key": ch, "code": code, "t": t + 90}]
        return out

    def test_prediction_includes_both_modalities_and_fusion(self):
        text = "i feel rushed and stressed about my deadline today"
        r = self.client.post("/api/predict", json={"text": text, "events": self._events(text)})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        for part in ["typing", "text", "fused", "early"]:
            self.assertIsNotNone(body[part], part)
            self.assertAlmostEqual(sum(body[part]["probs"].values()), 1.0, places=3)

    def test_too_little_input_returns_warnings_not_errors(self):
        body = self.client.post("/api/predict", json={"text": "hi", "events": []}).get_json()
        self.assertIsNone(body["typing"])
        self.assertIsNone(body["text"])
        self.assertEqual(len(body["warnings"]), 2)

    def test_malformed_events_are_rejected(self):
        r = self.client.post("/api/predict", json={"text": "a b c d", "events": "nope"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
