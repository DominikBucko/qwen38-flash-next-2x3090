"""Keep the published prefill figures tied to the checked-in measurements."""

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
    "prefill_progress", ROOT / "scripts/render_prefill_progress.py"
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class PrefillProgressTests(unittest.TestCase):
    def test_counts_reconcile_and_svg_exports_are_current(self):
        raw = (ROOT / "benchmarks/2026-09-18/summary.json").read_bytes()
        data = json.loads(raw)
        renderer.validate(data)
        digest = hashlib.sha256(raw).hexdigest()
        for filename, render in [
            ("prefill-long-context.svg", renderer.long_context),
            ("prefill-agent-cache.svg", renderer.agent_cache),
        ]:
            with self.subTest(filename=filename):
                self.assertEqual(
                    (ROOT / "docs/images" / filename).read_text(), render(data, digest)
                )


if __name__ == "__main__":
    unittest.main()
