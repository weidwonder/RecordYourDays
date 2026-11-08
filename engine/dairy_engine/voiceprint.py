"""speakers.json → 每人一个声纹。校验、缺省值与哈希同 src/speakers.mjs；向量用 sherpa-onnx Python 版 CAM++ 算，
缓存在 cache/gpu-voiceprints-<hash>.json（与 CPU 方案的 voiceprints-<hash>.json 分开）。

注册片段两种写法：{clip, source, start, end} 取 voiceprints/clips/ 下整个 clip（16 kHz 单声道 wav）；
旧写法 {file, start, end} 取录音里的一段。"""
import hashlib
import json
import os
import re
from pathlib import Path

from . import config as C
from .vec import mean_emb

ROLES = ("family", "known")
CLIP_RE = re.compile(r"voiceprints/clips/[a-zA-Z0-9_.-]+\.wav")


def normalize_speakers(speakers):
    """校验并补全缺省值：role 缺省 family，display 缺省为名字。"""
    if not isinstance(speakers, list) or not speakers:
        raise ValueError("speakers.json 应为非空数组")
    names = set()
    out = []
    for s in speakers:
        if not isinstance(s.get("name"), str) or not s["name"] or s["name"] in names:
            raise ValueError("speakers.json：名字为空或重复")
        names.add(s["name"])
        role = s.get("role", "family")
        if role not in ROLES:
            raise ValueError(f"speakers.json：{s['name']} 的 role 只能是 family 或 known")
        display = s.get("display", s["name"])
        if not isinstance(display, str) or not display or any(c.isspace() for c in display):
            raise ValueError(f"speakers.json：{s['name']} 的 display 不能为空或含空白")
        if not isinstance(s.get("threshold"), (int, float)):
            raise ValueError(f"speakers.json：{s['name']} 缺少数值 threshold")
        if not isinstance(s.get("enroll"), list) or not s["enroll"]:
            raise ValueError(f"speakers.json：{s['name']} 没有注册片段")
        for e in s["enroll"]:
            if "clip" in e:
                ok = isinstance(e["clip"], str) and CLIP_RE.fullmatch(e["clip"])
            else:
                ok = isinstance(e.get("file"), str) and e.get("start", 0) < e.get("end", 0)
            if not ok:
                raise ValueError(f"speakers.json：{s['name']} 的注册片段无效")
        out.append({**s, "role": role, "display": display})
    return out


def load_speakers(root):
    text = (Path(root) / "speakers.json").read_text(encoding="utf-8")
    return normalize_speakers(json.loads(text)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def people_from(speakers, vecs):
    """[{name, label, role, threshold, vec}]，label 是转写稿里显示的标签。"""
    return [{"name": s["name"], "label": s["display"], "role": s["role"], "threshold": s["threshold"], "vec": vecs[s["name"]]}
            for s in speakers]


def load_voiceprints(root, cfg, log=print, override=None):
    """override：现成的 {人名: 向量} JSON 文件（本机没有 sherpa-onnx 时由 Node 侧算好传入），不读写缓存。"""
    speakers, h = load_speakers(root)
    if override is not None:
        vecs = json.loads(Path(override).read_text())
        missing = [s["name"] for s in speakers if s["name"] not in vecs]
        if missing:
            raise ValueError("声纹文件缺人")
        return people_from(speakers, vecs)
    cache = Path(root) / "cache" / f"gpu-voiceprints-{h}.json"
    vecs = json.loads(cache.read_text()) if cache.exists() else None
    if vecs is None:
        from .stages import Campplus, decode_file, decode_range

        cam = Campplus(C.model_path(cfg, "campplus"), 4)
        vecs = {}
        for s in speakers:
            vs = []
            for e in s["enroll"]:
                samples = decode_file(Path(root) / e["clip"]) if "clip" in e else decode_range(Path(root) / e["file"], e["start"], e["end"])
                v = cam(samples)
                if not v:
                    raise ValueError(f"注册片段太短，算不出 embedding：{s['name']} {e.get('clip') or e['file']}")
                vs.append(v)
            vecs[s["name"]] = mean_emb(vs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{cache}.part"
        Path(tmp).write_text(json.dumps(vecs))
        os.replace(tmp, cache)
        log(f"声纹已按 speakers.json 重算 → cache/{cache.name}")
    return people_from(speakers, vecs)
