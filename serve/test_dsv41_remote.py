"""DeepSeek-V4.1's remote stage through the server's engine arguments: a config written the Qwen way (--layer-split K
--split-device 0 --remote-stage HOST:PORT) reaches strata-dsv41 as --remote-stage HOST:PORT --stage-end K, and every
other config's arguments are what they were.  No GPU, no engine.

    python -m unittest serve.test_dsv41_remote -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import server, stages  # noqa: E402
from serve.server import deepseek_remote_args, engine_args  # noqa: E402

BASE = ["--pack", "P", "--max-context", "8192", "--eos", "1", "--dspark", "auto"]
QWEN_WAY = ["--layer-split", "20", "--split-device", "0", "--remote-stage", "10.10.10.2:7841"]


class RemoteArgs(unittest.TestCase):
    def test_the_qwen_spelling_becomes_the_engines(self):
        self.assertEqual(deepseek_remote_args(BASE + QWEN_WAY),
                         BASE + ["--layer-split", "20", "--remote-stage", "10.10.10.2:7841", "--stage-end", "20"])

    def test_through_engine_args(self):
        cfg = {"arch": "deepseek41", "args": BASE + QWEN_WAY}
        args = engine_args(cfg)
        self.assertNotIn("--split-device", args)
        self.assertEqual(args[args.index("--stage-end") + 1], "20")
        self.assertEqual(args[args.index("--remote-stage") + 1], "10.10.10.2:7841")
        self.assertEqual(args[:len(BASE)], BASE)

    def test_the_engines_own_spelling_is_kept(self):
        own = BASE + ["--remote-stage", "h:7841", "--stage-end", "24"]
        self.assertEqual(deepseek_remote_args(own), own)
        # --stage-end wins over --layer-split (the person asked for it outright)
        both = own + ["--layer-split", "20"]
        self.assertEqual(deepseek_remote_args(both).count("--stage-end"), 1)
        self.assertEqual(deepseek_remote_args(both)[deepseek_remote_args(both).index("--stage-end") + 1], "24")

    def test_no_k_to_take_is_left_for_the_engine_to_say(self):
        for split in (["--layer-split", "auto"], []):
            args = BASE + split + ["--remote-stage", "h:7841"]
            self.assertNotIn("--stage-end", deepseek_remote_args(args), split)
        # a list of splits (several cards): the first is the worker's first layer, as the Workers card reads it
        self.assertEqual(deepseek_remote_args(BASE + ["--layer-split", "20,30", "--remote-stage", "h:1"])[-2:],
                         ["--stage-end", "20"])

    def test_nothing_changes_without_a_remote_stage(self):
        for args in (BASE, BASE + ["--layer-split", "20", "--split-device", "0"]):
            self.assertEqual(deepseek_remote_args(list(args)), args)
        self.assertEqual(engine_args({"arch": "deepseek41", "args": list(BASE)}), BASE)

    def test_the_qwen_engine_is_untouched(self):
        # only a deepseek41 config is translated: Qwen's --split-device is its own flag
        args = engine_args({"arch": "qwen4exp", "args": ["--pack", "P"] + QWEN_WAY})
        self.assertEqual(args, ["--pack", "P"] + QWEN_WAY)
        self.assertEqual(engine_args({"args": ["--pack", "P"] + QWEN_WAY}), ["--pack", "P"] + QWEN_WAY)

    def test_the_workers_card_still_reads_the_address_and_k(self):
        addr, k = stages.remote_args(engine_args({"arch": "deepseek41", "args": BASE + QWEN_WAY}))
        self.assertEqual((addr, k), ("10.10.10.2:7841", 20))


if __name__ == "__main__":
    unittest.main()
