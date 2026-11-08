"""GPU 方案参数：缺省值 + dairy.config.json 的 gpu 覆盖（gpu.remote 是 Node 侧的 ssh 配置，这里不读）。
改动影响缓存的参数会进缓存键。环境变量 DAIRY_ENGINE_ENV_DIR、DAIRY_ENGINE_MODEL_ROOT 覆盖 envDir、modelRoot
（Node 侧按本机或 gpu.remote 的配置传入）。"""
import json
import os
from pathlib import Path

DEFAULT_UNKNOWN_LABEL = "其他"
DEFAULTS = {
    # 环境与模型（运行转写的那台机器上的路径，~ 展开为 $HOME）；模型来源见 models.py
    "envDir": "~/.cache/record-your-days/engine-env",
    "modelRoot": "~/.cache/record-your-days/models",
    "paraformer": "modelscope/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": "modelscope/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "modelscope/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "diarizen": "huggingface/diarizen-wavlm-large-s80-md-v2",
    "diarizenEmbedding": "huggingface/pyannote-wespeaker-voxceleb-resnet34-LM/pytorch_model.bin",
    "mossformer": "modelscope/MossFormer2_SS_16K/last_best_checkpoint.pt",
    "campplus": "sherpa-onnx/speaker-embed/3dspeaker_campplus_sv_zh_en_16k-common_advanced/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    # ASR
    "asrBatch": 16,
    "vadMaxSec": 12.0,
    "vadMergeGap": 0.6,
    # 分离
    "diarizenBatch": 8,
    # 声纹（CPU）
    "embedWorkers": 12,
    "embedThreads": 2,
    "minEmbSec": 0.3,
    # 认人（与 src/label.mjs 一致）
    "confident": 0.5,
    "margin": 0.1,
    # 重叠支路
    "overlap": True,
    "overlapMinSec": 0.5,
    "overlapMaxSec": 10.0,
    "routeMinScore": 0.25,
    "routeMinMargin": 0.1,
    # 选卡
    "gpus": None,
    "minFreeMiB": 10000,
}


def load(root, env=None):
    env = os.environ if env is None else env
    cfg = dict(DEFAULTS)
    cfg["unknownLabel"] = DEFAULT_UNKNOWN_LABEL
    p = Path(root) / "dairy.config.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        user = {k: v for k, v in (data.get("gpu") or {}).items() if k != "remote"}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"dairy.config.json 的 gpu 有未知参数：{sorted(unknown)}")
        cfg.update(user)
        cfg["unknownLabel"] = data.get("unknownLabel", DEFAULT_UNKNOWN_LABEL)
    for key, var in (("envDir", "DAIRY_ENGINE_ENV_DIR"), ("modelRoot", "DAIRY_ENGINE_MODEL_ROOT")):
        if env.get(var):
            cfg[key] = env[var]
    return cfg


def model_path(cfg, key):
    return Path(cfg["modelRoot"]).expanduser() / cfg[key]


def label_params(cfg):
    return {"confident": cfg["confident"], "margin": cfg["margin"],
            "route_min_score": cfg["routeMinScore"], "route_min_margin": cfg["routeMinMargin"]}
