"""The extension's JavaScript engine must match Python. Needs Node (set ENGINE_NODE to choose one)."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import burst_features, text_features

ROOT = config.PROJECT_ROOT
EXT = ROOT / "extension"
EDGE = ["", "   ", "ok", "I am NOT happy!!! Why???", "this is the shit, kind of great but very sad",
        "never so good... at least not at all", "isn't it GREAT?! :) hmm", "café naïve résumé 123 a_b",
        "deadline deadline deadline pressure", "cut the mustard yeah right kiss of death", "the the the",
        "I can't sleep, I'm so stressed and anxious about my exams.\nHelp me!", "sort of bad, kind of awful, very very bad",
        "least bad, very least good, at least fine", "emoji 😀 stress 😀"]


def _node():
    node = os.environ.get("ENGINE_NODE") or shutil.which("node")
    if not node:
        return None
    try:
        return node if subprocess.run([node, "-v"], capture_output=True).returncode == 0 else None
    except OSError:
        return None


NODE = _node()
RUNNER = """
const fs = require('fs');
const load = (f) => (0, eval)(fs.readFileSync(f, 'utf8'));
load(process.argv[2]); load(process.argv[3]);
const v = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const out = {
  ling: v.texts.map((t) => StressEngine.linguistic(t)),
  text: v.texts.map((t) => StressEngine.textProbs(t).High),
  burst: v.bursts.map((b) => StressEngine.burstFeatures(b.t, b.back)),
  typing: v.bursts.map((b) => StressEngine.typingProbs(b.t, b.back)),
};
fs.writeFileSync(process.argv[5], JSON.stringify(out, (k, x) => (typeof x === 'number' && !Number.isFinite(x) ? null : x)));
"""


@unittest.skipUnless(NODE and (EXT / "model.js").exists(), "needs Node and extension/model.js")
class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_extension_model as ex
        cls.text_model, cls.ling_cols, cls.typing_model = ex.fit_models()
        texts = list(EDGE)
        if config.DREADDIT_TEST_PATH.exists():
            texts += pd.read_csv(config.DREADDIT_TEST_PATH)["text"].astype(str).tolist()[:300]
        rng = np.random.default_rng(0)
        bursts = []
        for n in [15, 16, 25, 60, 120, 400]:
            for scale in (60, 140, 400):
                times = np.cumsum(np.round(rng.exponential(scale, n) + 20, 2))
                bursts.append({"t": times.tolist(), "back": (rng.random(n) < 0.08).tolist()})
        bursts.append({"t": (np.arange(30) * 100.0).tolist(), "back": [False] * 30})   # perfectly steady
        cls.texts, cls.bursts = texts, bursts
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "v.json").write_text(json.dumps({"texts": texts, "bursts": bursts}))
            (tmp / "r.js").write_text(RUNNER)
            subprocess.run([NODE, str(tmp / "r.js"), str(EXT / "model.js"), str(EXT / "engine.js"),
                            str(tmp / "v.json"), str(tmp / "o.json")], check=True)
            cls.js = json.loads((tmp / "o.json").read_text())

    def test_exported_model_is_current(self):
        src = (EXT / "model.js").read_text()
        table = json.loads(src[src.index("= ") + 2:].rstrip().rstrip(";"))
        np.testing.assert_allclose(table["typing"]["coef"], self.typing_model.named_steps["m"].coef_)

    def test_linguistic_features_match(self):
        for i, t in enumerate(self.texts):
            py = text_features.extract_text_features(t)
            for k, v in py.items():
                self.assertAlmostEqual(self.js["ling"][i][k], v, places=9, msg=f"{k} on {t[:60]!r}")

    def test_text_probability_matches(self):
        ling = text_features.build_text_features(pd.Series(self.texts))[self.ling_cols]
        X = pd.concat([pd.DataFrame({"text": self.texts}), ling], axis=1)
        p = self.text_model.predict_proba(X)[:, list(self.text_model.classes_).index("High")]
        np.testing.assert_allclose(self.js["text"], p, atol=1e-9)

    def test_burst_features_match(self):
        for i, b in enumerate(self.bursts):
            py = burst_features.burst_features(b["t"], b["back"])
            for k in burst_features.FEATURES:
                got = self.js["burst"][i][k]
                if np.isnan(py[k]):
                    self.assertIsNone(got, k)
                else:
                    self.assertAlmostEqual(got, py[k], places=8, msg=k)

    def test_typing_probabilities_match(self):
        X = pd.DataFrame([burst_features.burst_features(b["t"], b["back"]) for b in self.bursts])[burst_features.FEATURES]
        p = self.typing_model.predict_proba(X)
        got = np.array([[row[c] for c in self.typing_model.classes_] for row in self.js["typing"]])
        np.testing.assert_allclose(got, p, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
