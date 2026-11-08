"""命令行：python -m dairy_engine transcribe <YYYY-MM-DD> [--force | --relabel-only [--voiceprints 文件]] [--root 目录] | check | models"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def check(root):
    """环境自检：输出 JSON（版本、GPU、模型文件是否齐全）。"""
    from . import config as C

    cfg = C.load(root)
    out = {"models": {}}
    for k in ("paraformer", "vad", "punc", "diarizen", "diarizenEmbedding", "mossformer", "campplus"):
        out["models"][k] = C.model_path(cfg, k).exists()
    try:
        import torch, funasr, sherpa_onnx  # noqa: E401

        out.update(torch=torch.__version__, cuda=torch.cuda.is_available(), gpus=torch.cuda.device_count(),
                   funasr=funasr.__version__, sherpa_onnx=sherpa_onnx.__version__)
    except Exception as e:  # noqa: BLE001
        out["error"] = type(e).__name__
    env = Path(cfg["envDir"]).expanduser()
    out["diarizen_venv"] = (env / "diarizen-venv/bin/python").exists()
    print(json.dumps(out, ensure_ascii=False))
    return 0 if all(out["models"].values()) and out.get("cuda") and out["diarizen_venv"] else 1


def main(argv):
    ap = argparse.ArgumentParser(prog="dairy_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("day")
    t.add_argument("--force", action="store_true")
    t.add_argument("--relabel-only", action="store_true", help="只用缓存重算认人与渲染，不读录音、不用 GPU")
    t.add_argument("--voiceprints", help="现成的声纹 JSON（{人名: 向量}），只配合 --relabel-only")
    t.add_argument("--root", default=".")
    c = sub.add_parser("check")
    c.add_argument("--root", default=".")
    m = sub.add_parser("models", help="下载 config 里缺的模型到 modelRoot，已有的跳过")
    m.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    os.umask(0o077)
    if a.cmd == "check":
        return check(a.root)
    if a.cmd == "models":
        from . import config as C
        from .models import download_missing

        return download_missing(C.load(a.root), log)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.day):
        ap.error("日期格式应为 YYYY-MM-DD")
    if a.relabel_only and a.force:
        ap.error("--relabel-only 与 --force 不能同时使用")
    if a.voiceprints and not a.relabel_only:
        ap.error("--voiceprints 只能配合 --relabel-only")
    from .pipeline import NoGpu, transcribe_day

    log(f"GPU 方案转写 {a.day}{' --force' if a.force else ''}{' --relabel-only' if a.relabel_only else ''}")
    try:
        transcribe_day(a.root, a.day, a.force, log, a.relabel_only, a.voiceprints)
    except NoGpu as e:
        log(str(e))
        return 3
    except Exception as e:  # noqa: BLE001
        # 只记异常类型与栈位置，不记异常消息：消息可能夹带输入内容。
        frames = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__)]
        log(f"失败：{type(e).__name__} @ {' < '.join(reversed(frames[-6:]))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
