#!/usr/bin/env python3
"""CPU correctness tests for the persistent KV tier journal.

journal.py is stdlib-only by design (it runs in the scheduler process), so
this test imports it directly and needs neither torch nor the runtime image.

Run: python3 -m unittest discover -s tests -p 'test_kv_tier_journal_cpu.py' -v
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

JOURNAL = (
    Path(__file__).parent.parent
    / "runtime/vllm-overlay/v1/simple_kv_offload/journal.py"
)


def load_journal():
    spec = importlib.util.spec_from_file_location("kvt_journal", JOURNAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JournalReplayTest(unittest.TestCase):
    def setUp(self):
        self.j = load_journal()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.journal")
        self.k1 = b"1" * 36
        self.k2 = b"2" * 36
        self.k3 = b"3" * 36

    def tearDown(self):
        self.tmp.cleanup()

    def replay(self, num_slots=8):
        return self.j.JournalReader.replay(self.path, num_slots=num_slots)

    def test_add_free_and_last_add_wins(self):
        w = self.j.JournalWriter(self.path)
        w.add_batch([(self.k1, 5), (self.k2, 7)])
        w.free_batch([5])          # slot 5's claim (k1) dies
        w.add_batch([(self.k3, 5)])  # slot 5 now belongs to k3
        w.close()
        mapping, stats = self.replay()
        self.assertEqual(mapping, {self.k2: 7, self.k3: 5})
        self.assertEqual(stats["freed"], 1)
        self.assertEqual(stats["bad_crc"], 0)

    def test_torn_tail_degrades_to_miss(self):
        w = self.j.JournalWriter(self.path)
        w.add_batch([(self.k1, 1)])
        w.close()
        clean, _ = self.replay()
        with open(self.path, "ab") as f:
            f.write(b"\x00\x01\x02")  # half a record
        mapping, stats = self.replay()
        self.assertEqual(mapping, clean)
        self.assertGreaterEqual(stats["torn_tail"], 1)

    def test_corrupt_record_is_refused_not_fatal(self):
        w = self.j.JournalWriter(self.path)
        w.add_batch([(self.k1, 1), (self.k2, 2)])
        w.close()
        with open(self.path, "rb") as f:
            data = bytearray(f.read())
        data[30] ^= 0xFF  # flip a byte inside the first record
        with open(self.path, "wb") as f:
            f.write(bytes(data))
        mapping, stats = self.replay()
        self.assertEqual(stats["bad_crc"], 1)
        self.assertNotIn(self.k1, mapping)

    def test_out_of_range_slot_refused(self):
        w = self.j.JournalWriter(self.path)
        w.add_batch([(self.k1, 99)])
        w.close()
        mapping, stats = self.replay(num_slots=8)
        self.assertEqual(mapping, {})
        self.assertEqual(stats["out_of_range"], 1)


if __name__ == "__main__":
    unittest.main()
