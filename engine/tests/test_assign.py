import unittest

from dairy_engine.assign import (Timeline, activity_regions, assign_sentence, assign_sentences, chunk_region,
                                 exclusive_pieces, two_speaker_regions)


def sent(chars, start=0.0, step=0.3):
    """合成句子：每个字占 step 秒。"""
    units = [[c, round(start + i * step, 3), round(start + (i + 1) * step, 3)] for i, c in enumerate(chars)]
    return {"start": units[0][1], "end": units[-1][2], "text": "".join(chars), "units": units}


class RegionTest(unittest.TestCase):
    def test_activity_and_overlap(self):
        spans = [(0, 5, "A"), (4, 8, "B"), (6, 7, "C"), (10, 12, "A")]
        regions = activity_regions(spans)
        self.assertEqual(regions, [(0, 4, frozenset("A")), (4, 5, frozenset("AB")), (5, 6, frozenset("B")),
                                   (6, 7, frozenset("BC")), (7, 8, frozenset("B")), (10, 12, frozenset("A"))])
        self.assertEqual(two_speaker_regions(regions, 0.5), [(4, 5, ("A", "B")), (6, 7, ("B", "C"))])
        self.assertEqual(exclusive_pieces(regions, 1.5), [(0, 4, "A"), (10, 12, "A")])

    def test_three_speakers_not_two(self):
        regions = activity_regions([(0, 3, "A"), (0, 3, "B"), (1, 2, "C")])
        self.assertEqual(two_speaker_regions(regions, 0.5), [(0, 1, ("A", "B")), (2, 3, ("A", "B"))])

    def test_short_overlap_dropped(self):
        self.assertEqual(two_speaker_regions(activity_regions([(0, 2, "A"), (1.6, 3, "B")]), 0.5), [])

    def test_same_speaker_touching_spans_merge(self):
        self.assertEqual(activity_regions([(0, 1, "A"), (1, 2, "A")]), [(0, 2, frozenset("A"))])

    def test_chunk_region(self):
        self.assertEqual(chunk_region(0, 10), [(0, 10)])
        self.assertEqual(chunk_region(0, 25), [(0, 25 / 3), (25 / 3, 50 / 3), (50 / 3, 25)])
        self.assertTrue(all(e - s <= 10 for s, e in chunk_region(3, 103.7)))


class AssignTest(unittest.TestCase):
    def tl(self, spans):
        return Timeline(activity_regions(spans))

    def test_single_speaker(self):
        out = assign_sentence(sent("今天天气很好"), self.tl([(0, 10, "A")]))
        self.assertEqual([(p["speaker"], p["text"]) for p in out], [("A", "今天天气很好")])

    def test_split_at_speaker_change(self):
        # 前 6 字 A（0–1.8 s），后 6 字 B（1.8–3.6 s）。
        out = assign_sentence(sent("你吃饭了没有我吃过了谢谢"), self.tl([(0, 1.8, "A"), (1.8, 4, "B")]))
        self.assertEqual([(p["speaker"], p["text"], p["start"], p["end"]) for p in out],
                         [("A", "你吃饭了没有", 0.0, 1.8), ("B", "我吃过了谢谢", 1.8, 3.6)])

    def test_short_interjection_not_split(self):
        # 中间 2 个字被判给 B：太短，并回前一段，整句不切。
        out = assign_sentence(sent("我们今天嗯嗯去公园玩吧"), self.tl([(0, 1.2, "A"), (1.2, 1.8, "B"), (1.8, 4, "A")]))
        self.assertEqual([(p["speaker"], p["text"]) for p in out], [("A", "我们今天嗯嗯去公园玩吧")])

    def test_overlap_units_undecided_follow_previous(self):
        # 1.8–2.4 s 两人同时在说，这两个字未定，并入前一段 A。
        out = assign_sentence(sent("你吃饭了没有我吃过了谢谢你"), self.tl([(0, 2.4, "A"), (1.8, 4, "B")]))
        self.assertEqual([(p["speaker"], p["text"]) for p in out], [("A", "你吃饭了没有我吃"), ("B", "过了谢谢你")])

    def test_no_speaker_majority_rule(self):
        # 没有任何字落在单人区：按重叠时长，B 占 0.9/1.2 ≥ 一半。
        s = sent("好的好的", step=0.3)
        out = assign_sentence(s, self.tl([(0, 1.2, "A"), (0.3, 1.2, "B"), (0.0, 1.2, "C")]))
        self.assertEqual(out[0]["speaker"], "A")
        out = assign_sentence(s, self.tl([(5, 6, "A")]))
        self.assertIsNone(out[0]["speaker"])
        out = assign_sentence(s, self.tl([(0, 0.5, "A"), (0.5, 0.55, "B")]))
        self.assertIsNone(out[0]["speaker"], "重叠不足句长一半")

    def test_assign_sentences_flat(self):
        out = assign_sentences([sent("一二三四五六"), sent("七八九", start=10)], [(0, 2, "A"), (10, 11, "B")])
        self.assertEqual([p["speaker"] for p in out], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
