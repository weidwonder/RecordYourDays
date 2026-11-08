import unittest

from dairy_engine.compose import compose
from dairy_engine import label as L
from dairy_engine.label import OTHER, choose_routes, decide, label_one
from dairy_engine.render import render_transcript
from dairy_engine.vec import l2

ME, PARTNER, CHILD = l2([1, 0, 0, 0]), l2([0, 1, 0, 0]), l2([0, 0, 1, 0])
PEOPLE = [{"name": "我", "threshold": 0.35, "vec": ME}, {"name": "伴侣", "threshold": 0.4, "vec": PARTNER},
          {"name": "孩子", "threshold": 0.35, "vec": CHILD}]
PARAMS = {"confident": 0.5, "margin": 0.1, "route_min_score": 0.25, "route_min_margin": 0.1}


def mix(a, b, w):
    return l2([w * x + (1 - w) * y for x, y in zip(a, b)])


class DecideTest(unittest.TestCase):
    def test_decide(self):
        self.assertEqual(decide(ME, PEOPLE)["label"], "我")
        self.assertEqual(decide(l2([0, 0, 0, 1]), PEOPLE)["label"], OTHER)
        self.assertEqual(decide(None, PEOPLE), {"label": OTHER, "top": -1, "second": -1})

    def test_label_one_rules(self):
        clusters = {"A": "伴侣"}
        self.assertEqual(label_one(ME, "A", clusters, PEOPLE), "我", "逐句有把握时推翻簇")
        self.assertEqual(label_one(mix(ME, PARTNER, 0.5), "A", clusters, PEOPLE), "伴侣", "逐句不明确时随簇")
        self.assertEqual(label_one(mix(ME, CHILD, 0.55), None, clusters, PEOPLE), "我", "没归属时用逐句")
        self.assertEqual(label_one(None, "X", clusters, PEOPLE), OTHER)


class UnknownLabelTest(unittest.TestCase):
    def tearDown(self):
        L.set_unknown_label(L.DEFAULT_UNKNOWN_LABEL)

    def test_configurable(self):
        L.set_unknown_label("未知")
        self.assertEqual(decide(l2([0, 0, 0, 1]), PEOPLE)["label"], "未知")
        utts = [{"start": 0, "end": 1, "label": "未知", "text": "a"}, {"start": 1.5, "end": 2, "label": "未知", "text": "b"}]
        self.assertEqual(render_transcript(utts), "[00:00] 未知 a\n[00:01] 未知 b\n")
        for bad in ("", "a b", "<x>", None):
            with self.assertRaises(ValueError):
                L.set_unknown_label(bad)


class RouteTest(unittest.TestCase):
    cents = {"A": ME, "B": PARTNER}

    def test_crossed_routes(self):
        r = choose_routes([mix(PARTNER, CHILD, 0.8), mix(ME, CHILD, 0.8)], ("A", "B"), self.cents, 0.25, 0.1)
        self.assertTrue(r["ok"])
        self.assertEqual(r["order"], ("B", "A"))

    def test_ambiguous_and_low_score_skipped(self):
        same = mix(ME, PARTNER, 0.5)
        self.assertEqual(choose_routes([same, same], ("A", "B"), self.cents, 0.25, 0.1)["reason"], "ambiguous")
        weak = [mix(ME, CHILD, 0.2), mix(PARTNER, CHILD, 0.2)]
        self.assertEqual(choose_routes(weak, ("A", "B"), self.cents, 0.25, 0.1)["reason"], "low-score")
        self.assertEqual(choose_routes([ME, None], ("A", "B"), self.cents, 0.25, 0.1)["reason"], "no-embedding")
        self.assertEqual(choose_routes([ME, PARTNER], ("A", "C"), self.cents, 0.25, 0.1)["reason"], "no-centroid")


def piece(text, start, step, speaker):
    units = [[c, round(start + i * step, 3), round(start + (i + 1) * step, 3)] for i, c in enumerate(text)]
    return {"start": units[0][1], "end": units[-1][2], "text": text, "units": units, "speaker": speaker}


class ComposeTest(unittest.TestCase):
    def cache(self):
        pieces = [piece("我们先吃饭吧", 0, 0.5, "A"), piece("好啊好啊你说什么", 10, 0.5, "B")]
        return {"pieces": pieces, "piece_embs": [mix(ME, PARTNER, 0.6), mix(PARTNER, ME, 0.6)],
                "excl": [[0, 3, "A"], [10, 12, "B"]], "excl_embs": [ME, PARTNER]}

    def test_without_overlap(self):
        utts, st = compose(self.cache(), PEOPLE, PARAMS, None)
        self.assertEqual([(u["label"], u["text"]) for u in utts], [("我", "我们先吃饭吧"), ("伴侣", "好啊好啊你说什么")])
        self.assertEqual(st["speakers"], 2)

    def test_overlap_accepted_replaces_mixed_text(self):
        # 12–14 s 两人重叠：这段里的混音字（第 5–8 个字）被去掉，换成两路各一行。
        chunk = {"start": 12, "end": 14, "pair": ["A", "B"],
                 "routes": [{"text": "你说什么", "emb": mix(PARTNER, CHILD, 0.8)}, {"text": "我说吃饭", "emb": mix(ME, CHILD, 0.8)}]}
        utts, st = compose(self.cache(), PEOPLE, PARAMS, [chunk])
        self.assertEqual([(u["start"], u["label"], u["text"]) for u in utts],
                         [(0, "我", "我们先吃饭吧"), (10, "伴侣", "好啊好啊"), (12, "伴侣", "你说什么"), (12, "我", "我说吃饭")])
        self.assertEqual((st["overlap_chunks"], st["overlap_accepted"], st["overlap_accepted_seconds"]), (1, 1, 2))

    def test_overlap_low_confidence_keeps_mix(self):
        same = mix(ME, PARTNER, 0.5)
        chunks = [{"start": 12, "end": 14, "pair": ["A", "B"], "routes": [{"text": "x", "emb": same}, {"text": "y", "emb": same}]},
                  {"start": 1, "end": 2, "pair": ["A", "B"], "routes": [{"text": "", "emb": ME}, {"text": "y", "emb": PARTNER}]}]
        utts, st = compose(self.cache(), PEOPLE, PARAMS, chunks)
        self.assertEqual([u["text"] for u in utts], ["我们先吃饭吧", "好啊好啊你说什么"])
        self.assertEqual(st["overlap_skipped"], {"ambiguous": 1, "empty-text": 1})

    def test_chunk_covering_whole_piece_removes_it(self):
        chunk = {"start": 9.9, "end": 15, "pair": ["A", "B"],
                 "routes": [{"text": "甲", "emb": ME}, {"text": "乙", "emb": PARTNER}]}
        utts, _ = compose(self.cache(), PEOPLE, PARAMS, [chunk])
        self.assertEqual([u["text"] for u in utts], ["我们先吃饭吧", "甲", "乙"])


if __name__ == "__main__":
    unittest.main()
