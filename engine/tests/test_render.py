import json
import unittest
from pathlib import Path

from dairy_engine.render import format_timestamp, render_transcript

CASES = json.loads((Path(__file__).resolve().parents[2] / "test/fixtures/render-cases.json").read_text(encoding="utf-8"))


class RenderTest(unittest.TestCase):
    def test_same_as_node(self):
        """与 src/render.mjs 用同一份样例（test/fixtures/render-cases.json），输出逐字一致。"""
        for c in CASES:
            with self.subTest(c["name"]):
                self.assertEqual(render_transcript([dict(u) for u in c["utts"]]), c["expected"])

    def test_timestamp(self):
        self.assertEqual(format_timestamp(28.55), "00:28")
        self.assertEqual(format_timestamp(3600), "1:00:00")
        self.assertEqual(format_timestamp(-1), "00:00")

    def test_input_unchanged(self):
        utts = [{"start": 0, "end": 1, "label": "我", "text": "a"}, {"start": 1.5, "end": 2, "label": "我", "text": "b"}]
        render_transcript(utts)
        self.assertEqual(utts[0]["text"], "a")


if __name__ == "__main__":
    unittest.main()
