"""GPU / CPU 计算阶段。只在 GPU 方案的 Python 环境（engine/setup_gpu.sh）里导入。

GPU 阶段在每卡一个的 worker 进程里跑（模型在 worker 内常驻，跨文件复用）；
DiariZen 在独立子 venv 的子进程里跑；CAM++ 声纹在 CPU 进程池里算。
worker 与子进程都把 stdout/stderr 指向 /dev/null 并禁止网络连接，第三方库的日志不会带出识别文字。
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .sentences import join_units, plan_segments, segment_sentences

SAMPLE_RATE = 16000
_W = {}


def deny_network(*args, **kwargs):
    raise RuntimeError("转写期间禁止网络连接")


def lock_down():
    """屏蔽第三方日志、禁止联网、只用本地模型。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network


# ---------- 音频 ----------

def decode_to_f32(src, dst):
    """ffmpeg 解码为 16 kHz 单声道 float32 裸数据文件，返回采样点数。"""
    tmp = f"{dst}.part"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-f", "f32le", tmp], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, dst)
    return os.path.getsize(dst) // 4


def load_f32(path):
    import numpy as np

    return np.memmap(path, dtype="<f4", mode="r")


def decode_range(src, start, end):
    """只解码 [start, end) 秒（旧写法的声纹登记片段用），返回 numpy 数组。src/clips.mjs 裁 clip 用同样的定位与重采样参数。"""
    import numpy as np

    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def decode_file(src):
    """整个文件解码为 16 kHz 单声道（声纹 clip 用）。clip 本身就是 16 kHz 单声道 float32，解出来与裁剪时逐点相同。"""
    import numpy as np

    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.frombuffer(p.stdout, dtype="<f4").copy()


# ---------- CAM++ 声纹（sherpa-onnx CPU，与 speakers.json 登记时同一模型与前端） ----------

class Campplus:
    def __init__(self, model, threads):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=threads, provider="cpu", debug=False)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, samples):
        import numpy as np

        stream = self.ex.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        if not self.ex.is_ready(stream):
            return None
        v = np.asarray(self.ex.compute(stream), dtype=np.float64)
        n = float(np.linalg.norm(v)) or 1.0
        return [round(float(x) / n, 6) for x in v]


_P = {}


def _embed_init(model, threads, audio_path):
    lock_down()
    _P["emb"] = Campplus(model, threads)
    _P["audio"] = load_f32(audio_path)


def _embed_task(ranges):
    a = _P["audio"]
    return [_P["emb"](a[round(s * SAMPLE_RATE):round(e * SAMPLE_RATE)]) for s, e in ranges]


def embed_ranges(audio_path, ranges, cfg):
    """多进程算一批区间的 embedding；顺序与 ranges 一致，算不出（太短）为 None。"""
    import multiprocessing as mp

    if not ranges:
        return []
    batch = 64
    parts = [ranges[i:i + batch] for i in range(0, len(ranges), batch)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(cfg["embedWorkers"], initializer=_embed_init,
                  initargs=(str(C.model_path(cfg, "campplus")), cfg["embedThreads"], str(audio_path))) as pool:
        out = []
        for r in pool.map(_embed_task, parts):
            out.extend(r)
    return out


# ---------- GPU worker ----------

def worker_init(gpu, cfg):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "4"
    lock_down()
    _W.update(gpu=gpu, cfg=cfg)


def _funmodel(key):
    from funasr import AutoModel

    return AutoModel(model=str(C.model_path(_W["cfg"], key)), device="cuda:0", disable_update=True, disable_pbar=True,
                     disable_log=True, ncpu=4)


def _asr_models():
    if "asr" not in _W:
        import torch
        import yaml

        torch.set_num_threads(4)
        _W["vad"] = _funmodel("vad")
        _W["asr"] = _funmodel("paraformer")
        _W["punc"] = _funmodel("punc")
        conf = yaml.safe_load((C.model_path(_W["cfg"], "punc") / "config.yaml").read_text(encoding="utf-8"))
        _W["punc_list"] = conf["model_conf"]["punc_list"]
    return _W["vad"], _W["asr"], _W["punc"], _W["punc_list"]


def _monitored(fn):
    from .gpu import VramMonitor

    with VramMonitor(_W["gpu"]) as mon:
        t0 = time.perf_counter()
        result = fn(mon)
        seconds = time.perf_counter() - t0
    return {"result": result, "seconds": seconds, "peak_mib": mon.peak_mib, "gpu": _W["gpu"]}


VAD_BLOCK_SEC = 1800


def _vad_segments(vad, audio):
    """FSMN VAD 按 30 分钟一块跑（整天音频一次送入显存占用过大），返回原始段（秒）。"""
    import numpy as np

    segs = []
    block = VAD_BLOCK_SEC * SAMPLE_RATE
    for b0 in range(0, len(audio), block):
        chunk = np.asarray(audio[b0:b0 + block], dtype=np.float32)
        if len(chunk) < SAMPLE_RATE // 10:
            continue
        out = vad.generate(input=chunk, cache={}, disable_pbar=True)
        segs.extend((b0 / SAMPLE_RATE + s / 1000, b0 / SAMPLE_RATE + e / 1000) for s, e in (out[0]["value"] if out else []))
    return segs


def recognize(asr, punc, punc_list, clips, batch, timestamps=True):
    """一批音频片段 → [(text, timestamp, punc_array, punc_text)]，顺序与输入一致。按时长排序后分批，减少补零。"""
    order = sorted(range(len(clips)), key=lambda i: -len(clips[i]))
    res = [None] * len(clips)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        kw = {"pred_timestamp": True} if timestamps else {}
        out = asr.generate(input=[clips[i] for i in idx], cache={}, batch_size=len(idx), disable_pbar=True, **kw)
        if len(out) != len(idx):
            raise RuntimeError("批量识别输出数量与输入不一致")
        for i, o in zip(idx, out):
            res[i] = (str(o.get("text", "")).strip(), o.get("timestamp") or [], None, "")
    for i, (text, ts, _, _) in enumerate(res):
        if text:
            p = punc.generate(input=text, disable_pbar=True)[0]
            arr = p.get("punc_array")
            res[i] = (text, ts, arr.tolist() if arr is not None else None, str(p.get("text", "")))
    return res


def run_asr(audio_path):
    def go(mon):
        import numpy as np

        vad, asr, punc, punc_list = _asr_models()
        audio = load_f32(audio_path)
        t0 = time.perf_counter()
        cfg = _W["cfg"]
        segs = plan_segments(_vad_segments(vad, audio), cfg["vadMergeGap"], cfg["vadMaxSec"])
        t_vad = time.perf_counter() - t0
        clips = [np.asarray(audio[round(s * SAMPLE_RATE):round(e * SAMPLE_RATE)], dtype=np.float32) for s, e in segs]
        keep = [i for i, c in enumerate(clips) if len(c) >= SAMPLE_RATE // 20]
        out = recognize(asr, punc, punc_list, [clips[i] for i in keep], cfg["asrBatch"])
        sentences = []
        for i, (text, ts, arr, _) in zip(keep, out):
            if text:
                sentences.extend(segment_sentences(text, ts, arr, punc_list, segs[i][0], segs[i][1]))
        sentences.sort(key=lambda s: s["start"])
        return {"sentences": sentences, "vad_segments": len(segs), "vad_seconds": t_vad}

    return _monitored(go)


def run_diar(audio_path, out_path):
    """在 DiariZen 子 venv 里跑整文件分离，结果写 out_path。"""
    def go(mon):
        cfg = _W["cfg"]
        env_dir = Path(cfg["envDir"]).expanduser()
        env = dict(os.environ, PYTHONPATH=f"{Path(__file__).resolve().parent.parent}:{env_dir / 'DiariZen'}")
        cmd = [str(env_dir / "diarizen-venv/bin/python"), "-m", "dairy_engine.diarizen_run", "--audio", str(audio_path),
               "--out", str(out_path), "--model", str(C.model_path(cfg, "diarizen")),
               "--embedding", str(C.model_path(cfg, "diarizenEmbedding")), "--batch", str(cfg["diarizenBatch"])]
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mon.pids.add(p.pid)
        code = p.wait()
        if code != 0:
            raise RuntimeError(f"DiariZen 子进程失败（退出码 {code}）")
        return json.loads(Path(out_path).read_text())

    return _monitored(go)


class Separator:
    """MossFormer2_SS_16K 二人盲分离（ClearerVoice 官方检查点，严格加载）。"""

    def __init__(self, ckpt):
        import torch
        from types import SimpleNamespace
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K

        self.torch = torch
        args = SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512, num_mossformer_layer=24,
                               encoder_kernel_size=16, num_spks=2)
        self.model = MossFormer2_SS_16K(args)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.model.load_state_dict(state, strict=True)
        self.model.eval().cuda()

    def __call__(self, audio):
        import numpy as np

        with self.torch.inference_mode():
            v = self.model(self.torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).cuda())
        return [s.reshape(-1).detach().cpu().numpy()[:len(audio)] for s in v]


def run_overlap(audio_path, chunks):
    """chunks: [(start, end, (A, B))] → [{start, end, pair, routes: [{text, emb}, {text, emb}]}]。"""
    def go(mon):
        import numpy as np

        cfg = _W["cfg"]
        _, asr, punc, punc_list = _asr_models()
        if "sep" not in _W:
            _W["sep"] = Separator(C.model_path(cfg, "mossformer"))
            _W["cam"] = Campplus(C.model_path(cfg, "campplus"), 4)
        audio = load_f32(audio_path)
        out = []
        t_sep = t_asr = t_emb = 0.0
        # 每 32 块一组：逐块分离，两路识别跨块按 asrBatch 成批。
        for g in range(0, len(chunks), 32):
            group = chunks[g:g + 32]
            t = time.perf_counter()
            sources = []
            for s, e, _ in group:
                sources.extend(_W["sep"](np.asarray(audio[round(s * SAMPLE_RATE):round(e * SAMPLE_RATE)], dtype=np.float32)))
            t_sep += time.perf_counter() - t
            t = time.perf_counter()
            rec = recognize(asr, punc, punc_list, sources, cfg["asrBatch"], timestamps=False)
            t_asr += time.perf_counter() - t
            t = time.perf_counter()
            embs = [_W["cam"](x) for x in sources]
            t_emb += time.perf_counter() - t
            for k, (s, e, pair) in enumerate(group):
                routes = [{"text": (pt or join_units(text.split())) if text else "", "emb": v}
                          for (text, _, _, pt), v in zip(rec[2 * k:2 * k + 2], embs[2 * k:2 * k + 2])]
                out.append({"start": round(s, 3), "end": round(e, 3), "pair": list(pair), "routes": routes})
        return {"chunks": out, "separation_seconds": t_sep, "asr_seconds": t_asr, "embedding_seconds": t_emb}

    return _monitored(go)


def worker_ping():
    import torch

    return {"cuda": torch.cuda.is_available(), "pid": os.getpid(), "python": sys.version.split()[0]}
