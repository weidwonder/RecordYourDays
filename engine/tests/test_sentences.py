import unittest

from dairy_engine.sentences import join_units, make_units, plan_segments, punc_symbols, segment_sentences

PUNC = ["<unk>", "_", "，", "。", "？", "、"]


class SentenceTest(unittest.TestCase):
    def test_split_on_sentence_end(self):
        # 「你好，吃了吗？好的。」10 个 token；时间戳相对片段起点 100 s。
        text = "你 好 吃 了 吗 好 的"
        ts = [[0, 200], [200, 400], [600, 800], [800, 1000], [1000, 1200], [2000, 2200], [2200, 2400]]
        arr = [1, 2, 1, 1, 4, 1, 3]
        out = segment_sentences(text, ts, arr, PUNC, 100.0, 103.0)
        self.assertEqual([s["text"] for s in out], ["你好，吃了吗？", "好的。"])
        self.assertEqual((out[0]["start"], out[0]["end"]), (100.0, 101.2))
        self.assertEqual((out[1]["start"], out[1]["end"]), (102.0, 102.4))
        self.assertEqual(out[1]["units"], [["好", 102.0, 102.2], ["的。", 102.2, 102.4]])

    def test_tail_without_end_punc_kept(self):
        out = segment_sentences("好 的 那", [[0, 100], [100, 200], [200, 300]], [1, 3, 1], PUNC, 0, 1)
        self.assertEqual([s["text"] for s in out], ["好的。", "那"])

    def test_negative_and_out_of_range_stamps_clamped(self):
        units = make_units(["大", "v"], [[-90, 30], [30, 5000]], None, 10.0, 11.0)
        self.assertEqual(units, [["大", 10.0, 10.03], ["v", 10.03, 11.0]])

    def test_mismatched_stamps_fall_back_to_segment(self):
        out = segment_sentences("你 好", [[0, 100]], [1, 3], PUNC, 5.0, 6.5)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["start"], out[0]["end"], out[0]["text"]), (5.0, 6.5, "你好。"))

    def test_mismatched_punc_ignored(self):
        out = segment_sentences("你 好", [[0, 100], [100, 200]], [1], PUNC, 0, 1)
        self.assertEqual(out[0]["text"], "你好")

    def test_english_words_spaced(self):
        self.assertEqual(join_units(["打", "开", "hello", "world，", "好"]), "打开hello world，好")
        self.assertEqual(join_units(["ok，", "fine"]), "ok，fine")

    def test_punc_symbols(self):
        self.assertEqual(punc_symbols([1, 2, 3, 4, 5, 0, 9], PUNC), ["", "，", "。", "？", "、", "", ""])

    def test_plan_segments(self):
        raw = [(0, 1), (1, 2.5), (3.0, 4), (6, 7), (7, 30)]
        # 前三段间隔 ≤ 0.6 合并；6–7 与 7–30 合并会超过 12 秒，不合并；7–30 等长切成两段。
        self.assertEqual(plan_segments(raw, 0.6, 12), [(0, 4), (6, 7), (7, 18.5), (18.5, 30)])
        self.assertEqual(plan_segments([], 0.6, 12), [])

    def test_empty(self):
        self.assertEqual(segment_sentences("", [], None, PUNC, 0, 1), [])


if __name__ == "__main__":
    unittest.main()
