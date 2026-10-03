"""Regression tests for press/release keystroke feature extraction."""

import unittest

import pandas as pd

from features.typing_features import features_from_events


class TypingFeatureTests(unittest.TestCase):
    def test_press_release_events_keep_dwell_separate_from_flight(self):
        base = pd.Timestamp("2026-01-01 12:00:00")
        rows = []
        for index in range(20):
            press = base + pd.Timedelta(milliseconds=index * 100)
            release = press + pd.Timedelta(milliseconds=40)
            rows.extend(
                [
                    {"user": "u1", "timestamp": press, "key": "a", "event": "press", "stress_label": "Low"},
                    {"user": "u1", "timestamp": release, "key": "a", "event": "release", "stress_label": "Low"},
                ]
            )

        result = features_from_events(pd.DataFrame(rows), window_seconds=300)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["n_keystrokes"], 20)
        self.assertAlmostEqual(row["mean_dwell_ms"], 40.0, places=5)
        self.assertAlmostEqual(row["mean_flight_ms"], 100.0, places=5)


if __name__ == "__main__":
    unittest.main()
