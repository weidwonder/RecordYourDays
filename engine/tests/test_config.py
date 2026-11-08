import json
import tempfile
import unittest
from pathlib import Path

from dairy_engine import config as C
from dairy_engine import models as M


class ConfigTest(unittest.TestCase):
    def write(self, d, data):
        Path(d, "dairy.config.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_defaults_without_file(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = C.load(d, env={})
            self.assertEqual(cfg["modelRoot"], "~/.cache/record-your-days/models")
            self.assertEqual(cfg["unknownLabel"], "其他")

    def test_gpu_section_remote_ignored_env_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"unknownLabel": "未知", "gpu": {"minFreeMiB": 5000, "envDir": "/e", "remote": {"host": "gpu"}}})
            cfg = C.load(d, env={"DAIRY_ENGINE_MODEL_ROOT": "/m"})
            self.assertEqual((cfg["minFreeMiB"], cfg["envDir"], cfg["modelRoot"], cfg["unknownLabel"]), (5000, "/e", "/m", "未知"))
            self.assertNotIn("remote", cfg)
            self.assertEqual(C.load(d, env={"DAIRY_ENGINE_ENV_DIR": "/x"})["envDir"], "/x")

    def test_unknown_gpu_key(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"gpu": {"typo": 1}})
            with self.assertRaises(ValueError):
                C.load(d, env={})

    def test_example_config_loads(self):
        example = (Path(__file__).resolve().parents[2] / "dairy.config.example.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "dairy.config.json").write_text(example, encoding="utf-8")
            cfg = C.load(d, env={})
            self.assertEqual((cfg["minFreeMiB"], cfg["unknownLabel"]), (10000, "其他"))


class ModelPlanTest(unittest.TestCase):
    def test_every_model_has_a_source(self):
        self.assertEqual(set(M.SOURCES), {"paraformer", "vad", "punc", "diarizen", "diarizenEmbedding", "mossformer", "campplus"})
        for key, (source, repo, filename, _) in M.SOURCES.items():
            self.assertIn(source, ("modelscope", "huggingface", "url"))
            if filename:
                self.assertEqual(Path(C.DEFAULTS[key]).name, filename, key)

    def test_plan_skips_existing_and_unknown_paths(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {**C.DEFAULTS, "modelRoot": d, "punc": "custom/punc"}
            para = C.model_path(cfg, "paraformer")
            para.mkdir(parents=True)
            for f in ("model.pt", "config.yaml"):
                (para / f).write_text("x")
            cam = C.model_path(cfg, "campplus")
            cam.parent.mkdir(parents=True)
            cam.write_text("x")
            states = {k: s for k, _, s in M.plan(cfg)}
            self.assertEqual(states["paraformer"], "have")
            self.assertEqual(states["campplus"], "have")
            self.assertEqual(states["vad"], "download")
            self.assertEqual(states["punc"], "unknown")
            (para / "model.pt").write_text("")
            self.assertEqual({k: s for k, _, s in M.plan(cfg)}["paraformer"], "download", "空文件不算已下载")


if __name__ == "__main__":
    unittest.main()
