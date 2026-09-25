"""Keep the September 25 chart tied to the checked-in measurements."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "fast_runtime", ROOT / "scripts/render_fast_runtime.py"
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class FastRuntimeChartTests(unittest.TestCase):
    def test_svg_is_current_and_uses_release_medians(self):
        previous = (ROOT / "benchmarks/2026-09-18/summary.json").read_bytes()
        current = (ROOT / "benchmarks/2026-09-25/summary.json").read_bytes()
        hashes = (hashlib.sha256(previous).hexdigest(), hashlib.sha256(current).hexdigest())
        expected = renderer.render(json.loads(previous), json.loads(current), hashes)
        self.assertEqual((ROOT / "docs/images/fast-256k-progress.svg").read_text(), expected)
        release = renderer.series(json.loads(previous), json.loads(current))[-1]["points"]
        self.assertEqual([p[0] for p in release], [131072, 260096])
        data = json.loads(current)["release_image_validation"]
        for tokens in (131072, 260096):
            runs = data[f"input_{tokens}_output_2048"]["measured_runs"]
            self.assertEqual(len(runs), 3)


if __name__ == "__main__":
    unittest.main()
