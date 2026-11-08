"""GPU 方案的模型下载：config.py 里每个模型的公开来源，下载到 modelRoot，已有的跳过。

下载计划（plan）只用标准库；真正下载时才导入 modelscope / huggingface_hub（都在主环境的锁文件里）。
只认缺省路径：dairy.config.json 改了某个模型的路径时，不知道它从哪来，缺了就提示自行下载。
"""
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from . import config as C

# 键 → (来源, 仓库或网址, 单个文件名或 None（整个仓库）, 判断「已下载」要有的文件)
SOURCES = {
    "paraformer": ("modelscope", "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", None, ("model.pt", "config.yaml")),
    "vad": ("modelscope", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", None, ("model.pt", "config.yaml")),
    "punc": ("modelscope", "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", None, ("model.pt", "config.yaml")),
    "diarizen": ("huggingface", "BUT-FIT/diarizen-wavlm-large-s80-md-v2", None, ("pytorch_model.bin", "config.toml")),
    "diarizenEmbedding": ("huggingface", "pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin", ()),
    "mossformer": ("modelscope", "alibabasglab/MossFormer2_SS_16K", "last_best_checkpoint.pt", ()),
    "campplus": ("url", "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
                 "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", None, ()),
}


def _have(path, needs):
    if needs:
        return all((path / n).is_file() and (path / n).stat().st_size > 0 for n in needs)
    return path.is_file() and path.stat().st_size > 0


def plan(cfg):
    """[(键, 目标路径, 状态)]，状态为 have（已有）、download（要下载）、unknown（改过路径且缺失，不知道来源）。"""
    out = []
    for key, (_, _, _, needs) in SOURCES.items():
        path = C.model_path(cfg, key)
        if _have(path, needs):
            out.append((key, path, "have"))
        else:
            out.append((key, path, "download" if cfg[key] == C.DEFAULTS[key] else "unknown"))
    return out


def _fetch(source, repo, filename, dest):
    """下载到 dest（整个仓库时 dest 是目录，单个文件时 dest 是文件）。先下到同目录的临时位置，齐了再改名。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        if source == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download

            kw = {"allow_patterns": [filename]} if filename else {}
            snapshot_download(repo, local_dir=str(tmp), **kw)
        elif source == "huggingface":
            if filename:
                from huggingface_hub import hf_hub_download

                hf_hub_download(repo, filename, local_dir=str(tmp))
            else:
                from huggingface_hub import snapshot_download

                snapshot_download(repo, local_dir=str(tmp))
        else:
            with urllib.request.urlopen(repo, timeout=60) as r, open(tmp / dest.name, "wb") as f:
                shutil.copyfileobj(r, f)
            filename = dest.name
        got = tmp / filename if filename else tmp
        if not got.exists():
            raise FileNotFoundError(f"下载结果里没有 {filename}")
        if not filename:
            shutil.rmtree(got / ".cache", ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        os.replace(got, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_missing(cfg, log=print):
    """下载缺的模型；返回退出码（有不知道来源的缺失模型时为 1）。"""
    bad = 0
    for key, path, state in plan(cfg):
        if state == "have":
            log(f"[跳过] {key} 已存在：{path}")
            continue
        if state == "unknown":
            log(f"[缺失] {key}：{path} 不是缺省路径，不知道来源，请自行下载")
            bad = 1
            continue
        source, repo, filename, _ = SOURCES[key]
        log(f"[下载] {key} ← {source} {repo}{f' ({filename})' if filename else ''}")
        _fetch(source, repo, filename, path)
        log(f"  完成 → {path}")
    return bad
