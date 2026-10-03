"""Regression tests for collected data and fusion input validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataio.loaders import load_own_keystrokes
from features.typing_features import prepare_feature_frame
from modeling.fusion import StackingFusion, weighted_average_fusion


class OwnKeystrokeLoaderTests(unittest.TestCase):
    def test_sessions_stay_separate_when_timestamps_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "keystrokes_example.csv"
            pd.DataFrame(
                [
                    ["p1", "p1_B1", "down", "a", 10, "Low"],
                    ["p1", "p1_B2", "down", "b", 10, "High"],
                ],
                columns=[
                    "participant_id", "session_id", "event", "key", "t_ms",
                    "stress_label",
                ],
            ).to_csv(path, index=False)

            loaded = load_own_keystrokes(path)

        self.assertEqual(loaded["user"].nunique(), 2)
        self.assertEqual(set(loaded["event"]), {"press"})

    def test_excluded_post_task_ratings_do_not_reach_typing_model(self):
        frame = pd.DataFrame(
            {
                "typing_speed_cpm": [120, 180],
                "self_report_intensity": [1, 5],
                "self_report_pressure": [1, 5],
                "stress_label": ["Low", "High"],
            }
        )

        features, _ = prepare_feature_frame(
            frame,
            excluded_columns=["self_report_intensity", "self_report_pressure"],
        )

        self.assertEqual(features.columns.tolist(), ["typing_speed_cpm"])


class FusionValidationTests(unittest.TestCase):
    def test_weighted_fusion_rejects_invalid_weight(self):
        probabilities = np.array([[0.3, 0.3, 0.4]])
        with self.assertRaises(ValueError):
            weighted_average_fusion(probabilities, probabilities, weight_typing=1.1)

    def test_stacking_rejects_mismatched_probability_shapes(self):
        fusion = StackingFusion()
        with self.assertRaises(ValueError):
            fusion.fit(np.ones((2, 3)), np.ones((3, 3)), ["Low", "High"])


if __name__ == "__main__":
    unittest.main()
