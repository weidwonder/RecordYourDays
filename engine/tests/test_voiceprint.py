import json
import tempfile
import unittest
from pathlib import Path

from dairy_engine import config as C
from dairy_engine.compose import compose
from dairy_engine.label import decide
from dairy_engine.pipeline import Day, downstream_keys
from dairy_engine.vec import l2
from dairy_engine.voiceprint import load_voiceprints, normalize_speakers

AX = [l2([1 if i == k else 0 for i in range(6)]) for k in range(6)]  # 我 伴侣 孩子 同事A 同事B 陌生人
OLD = [{"name": "我", "threshold": 0.35, "enroll": [{"file": "raw/2026-11-01/a.MP3", "start": 1, "end": 2}]}]
CLIP = {"clip": "voiceprints/clips/20261103_081500_1000-3000.wav", "source": "raw/2026-11-03/20261103_081500.MP3", "start": 1, "end": 3}
SPEAKERS = OLD + [
    {"name": "伴侣", "role": "family", "threshold": 0.4, "enroll": [CLIP]},
    {"name": "同事A", "role": "known", "threshold": 0.4, "enroll": [CLIP]},
    {"name": "同事B", "role": "known", "display": "老B", "threshold": 0.4, "enroll": [CLIP]},
]
VECS = {"我": AX[0], "伴侣": AX[1], "同事A": AX[3], "同事B": AX[4]}
PARAMS = {"confident": 0.5, "margin": 0.1, "route_min_score": 0.25, "route_min_margin": 0.1}


def mix(pairs):
    return l2([sum(w for k, w in pairs if k == i) for i in range(6)])


class SpeakersTest(unittest.TestCase):
    def test_old_format_defaults(self):
        s = normalize_speakers(OLD)
        self.assertEqual((s[0]["role"], s[0]["display"]), ("family", "我"))

    def test_new_format_and_errors(self):
        s = normalize_speakers(SPEAKERS)
        self.assertEqual([x["display"] for x in s], ["我", "伴侣", "同事A", "老B"])
        with self.assertRaises(ValueError):
            normalize_speakers([{**OLD[0], "role": "friend"}])
        with self.assertRaises(ValueError):
            normalize_speakers([{**OLD[0], "enroll": [{"clip": "../x.wav"}]}])
        with self.assertRaises(ValueError):
            normalize_speakers([{**OLD[0], "display": "a b"}])


def piece(text, start, speaker):
    units = [[c, start + i * 0.5, start + (i + 1) * 0.5] for i, c in enumerate(text)]
    return {"start": units[0][1], "end": units[-1][2], "text": text, "units": units, "speaker": speaker}


class KnownPeopleTest(unittest.TestCase):
    def people(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "speakers.json").write_text(json.dumps(SPEAKERS, ensure_ascii=False))
            Path(d, "vp.json").write_text(json.dumps(VECS))
            return load_voiceprints(d, C.load(d), override=Path(d, "vp.json"))

    def test_display_label(self):
        people = self.people()
        self.assertEqual(decide(mix([(4, 0.9), (1, 0.3)]), people)["label"], "老B")
        self.assertEqual(decide(mix([(3, 0.35), (1, 0.3), (5, 0.9)]), people)["label"], "其他")

    def test_colleague_clusters_match_themselves(self):
        # 三个簇的声音相近，都带一点伴侣的成分；登记了两位同事后各自认成自己。
        cache = {"pieces": [piece("今天开会", 0, "W"), piece("项目进度", 10, "C1"), piece("下午再说", 20, "C2")],
                 "piece_embs": [None, mix([(1, 0.5), (3, 0.45)]), mix([(1, 0.5), (4, 0.45)])],
                 "excl": [[0, 5, "W"], [10, 15, "C1"], [20, 25, "C2"]],
                 "excl_embs": [mix([(1, 0.9), (3, 0.2)]), mix([(3, 0.8), (1, 0.5)]), mix([(4, 0.8), (1, 0.5)])]}
        people = self.people()
        utts, _ = compose(cache, people, PARAMS, None)
        self.assertEqual([u["label"] for u in utts], ["伴侣", "同事A", "老B"])
        family_only = [p for p in people if p["role"] == "family"]
        utts, _ = compose(cache, family_only, PARAMS, None)
        self.assertEqual([u["label"] for u in utts], ["伴侣", "伴侣", "伴侣"])


class RelabelOnlyTest(unittest.TestCase):
    def setup_root(self, d):
        root = Path(d)
        (root / "speakers.json").write_text(json.dumps(SPEAKERS, ensure_ascii=False))
        (root / "vp.json").write_text(json.dumps(VECS))
        cache = root / "cache" / "2026-11-03"
        cache.mkdir(parents=True)
        keys = downstream_keys("a1", "d1", C.load(root))
        pieces = [piece("我们先吃饭吧", 0, "A"), piece("好的好的", 10, "B")]
        files = {
            "asr": {"sentences": [{"start": 0, "end": 3, "text": "我们先吃饭吧"}, {"start": 10, "end": 12, "text": "好的好的"}], "key": "a1"},
            "diar": {"spans": [[0, 3, "A"], [10, 12, "B"]], "key": "d1"},
            "embed": {"pieces": pieces, "piece_embs": [AX[0], mix([(3, 0.9), (1, 0.3)])], "excl": [[0, 3, "A"], [10, 12, "B"]],
                      "excl_embs": [AX[0], mix([(3, 0.9), (1, 0.3)])], "key": keys["embed"]},
            "overlap": {"chunks": [], "key": keys["overlap"]},
        }
        for stage, v in files.items():
            (cache / f"gpu-20261103_081500.{stage}.json").write_text(json.dumps(v, ensure_ascii=False))
        return root, cache

    def test_relabel_without_audio(self):
        with tempfile.TemporaryDirectory() as d:
            root, cache = self.setup_root(d)
            Day(root, "2026-11-03", False, lambda m: None, root / "vp.json").relabel()
            text = (root / "transcripts/2026-11-03/20261103_081500.md").read_text(encoding="utf-8")
            self.assertEqual(text, "[00:00] 我 我们先吃饭吧\n[00:10] 同事A 好的好的\n")
            self.assertEqual(json.loads((cache / "gpu-stats.json").read_text())["labels"], {"我": 1, "同事A": 1})

    def test_relabel_refuses_mismatched_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root, cache = self.setup_root(d)
            p = cache / "gpu-20261103_081500.embed.json"
            p.write_text(json.dumps({**json.loads(p.read_text()), "key": "stale"}))
            with self.assertRaises(ValueError):
                Day(root, "2026-11-03", False, lambda m: None, root / "vp.json").relabel()
            (cache / "gpu-20261103_081500.diar.json").unlink()
            with self.assertRaises(FileNotFoundError):
                Day(root, "2026-11-03", False, lambda m: None, root / "vp.json").relabel()


if __name__ == "__main__":
    unittest.main()
