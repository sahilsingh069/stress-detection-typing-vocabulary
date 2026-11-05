"""Nested cross-validation and text-independent timing features."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import keystroke_dynamics as kd
from modeling.nested import Option, Part, nested_proba


def _table(n_people=20, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for label in (0, 1):
            for _ in range(3):
                rows.append({"person": f"p{p}", "unit": f"p{p}_{label}", "good": rng.normal(signal * label),
                             "noise": rng.normal()})
    return pd.DataFrame(rows)


class NestedTests(unittest.TestCase):
    def _run(self, table, options):
        units = table.groupby("unit")["person"].first().to_frame()
        labels = [u.endswith("_1") for u in units.index]
        return nested_proba(options, units, np.array(labels).astype(int), 0), np.array(labels).astype(int)

    def test_picks_the_useful_option(self):
        t = _table()
        (proba, picks), y = self._run(t, [Option("good", [Part(t, ["good"], LogisticRegression())]),
                                          Option("noise", [Part(t, ["noise"], LogisticRegression())])])
        self.assertEqual(set(picks), {"good"})
        self.assertGreater((proba.argmax(1) == y).mean(), 0.9)

    def test_many_noise_options_stay_near_chance(self):
        # the point of nesting: choosing among noise must not look like skill
        t = _table(signal=0.0)
        for i in range(10):
            t[f"n{i}"] = np.random.default_rng(i).normal(size=len(t))
        (proba, _), y = self._run(t, [Option(f"n{i}", [Part(t, [f"n{i}"], LogisticRegression())]) for i in range(10)])
        self.assertLess((proba.argmax(1) == y).mean(), 0.7)

    def test_held_out_people_are_not_used_for_fitting(self):
        seen = []

        class Spy(LogisticRegression):
            def fit(self, X, y):
                seen.append(set(X.index))
                return super().fit(X, y)

        t = _table(n_people=10)
        self._run(t, [Option("a", [Part(t, ["good"], Spy())])])
        units = t.groupby("unit")["person"].first().to_frame()
        people = units["person"].nunique()
        self.assertTrue(all(t.loc[list(ix), "person"].nunique() < people for ix in seen))


class DynamicsTests(unittest.TestCase):
    def test_pair_ratio_ignores_which_letters_were_typed(self):
        table = pd.Series({"ab": 100.0, "cd": 400.0})
        same_pace_a = kd.pair_features(list("ababab"), [100.0] * 5, pd.Series({"ab": 100.0, "ba": 100.0}), "x")
        same_pace_b = kd.pair_features(list("cdcdcd"), [400.0] * 5, pd.Series({"cd": 400.0, "dc": 400.0}), "x")
        self.assertAlmostEqual(same_pace_a["x_median"], same_pace_b["x_median"])
        self.assertIn("ab", table.index)

    def test_too_few_pairs_gives_missing_not_zero(self):
        self.assertTrue(np.isnan(kd.pair_features(list("ab"), [100.0], pd.Series({"ab": 100.0}), "x")["x_median"]))

    def test_steady_rhythm_has_small_steps(self):
        steady = kd.rhythm_features([100.0] * 30)
        rng = np.random.default_rng(0)
        jumpy = kd.rhythm_features(rng.uniform(50, 900, 30))
        self.assertLess(steady["rhythm_step_median"], jumpy["rhythm_step_median"])


if __name__ == "__main__":
    unittest.main()
