"""一天的录音 → 转写稿（GPU 方案）。

阶段与缓存（cache/<日期>/，前缀 gpu-，不与 CPU 方案缓存混用）：
  asr      FSMN VAD + Paraformer + CT 标点 → 句子（GPU）           gpu-<名>.asr.json
  diar     DiariZen 整文件说话人分离（GPU，子 venv）                gpu-<名>.diar.json
  embed    句子归属 + CAM++ 声纹（CPU，依赖 asr、diar）              gpu-<名>.emb.json
  overlap  恰好两人重叠区 → MossFormer2 → Paraformer → CAM++（GPU） gpu-<名>.overlap.json
认人与渲染每次现算。每个缓存带 key，上游重算或参数变了，下游 key 对不上就自动重算。

多张卡并行：每卡一个 worker 进程，按依赖关系挑能跑的阶段，耗时长的先跑；同一天的不同文件、同一文件的
ASR 与分离都可以分到不同的卡上同时跑。
"""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

from . import config as C
from .assign import activity_regions, assign_sentences, chunk_region, exclusive_pieces, two_speaker_regions
from .compose import compose
from .label import set_unknown_label
from .render import render_transcript

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus"}
VERSION = {"asr": 1, "diar": 1, "embed": 1, "overlap": 1}
# 估算各 GPU 阶段的 RTF，只用于排序（实测值）。
COST = {"diar": 0.06, "asr": 0.013, "overlap": 0.01}


class NoGpu(RuntimeError):
    pass


def _h(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write(path, value):
    tmp = f"{path}.part"
    Path(tmp).write_text(json.dumps(value, ensure_ascii=False))
    os.replace(tmp, path)


def downstream_keys(asr, diar, cfg):
    """embed、overlap 的键只由上游键和参数决定；只重算认人时据此核对缓存是否成套，不用读录音。"""
    embed = _h("embed", VERSION["embed"], asr, diar, cfg["campplus"], cfg["minEmbSec"])
    overlap = _h("overlap", VERSION["overlap"], diar, cfg["overlapMinSec"], cfg["overlapMaxSec"], cfg["mossformer"],
                 cfg["paraformer"], cfg["punc"], cfg["campplus"])
    return {"embed": embed, "overlap": overlap}


def stage_keys(src_sha, cfg):
    asr = _h("asr", VERSION["asr"], src_sha, cfg["paraformer"], cfg["vad"], cfg["punc"], cfg["vadMaxSec"], cfg["vadMergeGap"])
    diar = _h("diar", VERSION["diar"], src_sha, cfg["diarizen"], cfg["diarizenEmbedding"])
    return {"asr": asr, "diar": diar, **downstream_keys(asr, diar, cfg)}


def overlap_chunks(spans, cfg):
    regions = two_speaker_regions(activity_regions([tuple(s) for s in spans]), cfg["overlapMinSec"])
    return [(s, e, pair) for rs, re_, pair in regions for s, e in chunk_region(rs, re_, cfg["overlapMaxSec"])]


def embed_stage(audio_path, sentences, spans, cfg):
    """句子归属（可能切分）+ 每段声纹 + 每个说话人独占片段的声纹。CPU 多进程。"""
    from .stages import embed_ranges

    spans = [tuple(s) for s in spans]
    pieces = assign_sentences(sentences, spans)
    excl = exclusive_pieces(activity_regions(spans), cfg["minEmbSec"])
    t0 = time.perf_counter()
    embs = embed_ranges(audio_path, [(p["start"], p["end"]) for p in pieces] + [(s, e) for s, e, _ in excl], cfg)
    return {"pieces": pieces, "piece_embs": embs[:len(pieces)], "excl": [list(x) for x in excl],
            "excl_embs": embs[len(pieces):], "seconds": time.perf_counter() - t0}


class Day:
    def __init__(self, root, day, force, log, voiceprints=None):
        self.root, self.day, self.force, self.log, self.voiceprints = Path(root), day, force, log, voiceprints
        self.cfg = C.load(root)
        set_unknown_label(self.cfg["unknownLabel"])
        self.raw = self.root / "raw" / day
        self.cache = self.root / "cache" / day
        self.out = self.root / "transcripts" / day
        self.tmp = self.root / "tmp" / f"engine-{day}"
        self.stats = {"day": day, "engine": "gpu", "files": {}, "stage_seconds": {}, "gpus": {}}

    def cpath(self, base, stage):
        return self.cache / f"gpu-{base}.{stage}.json"

    def plan(self):
        files = sorted(p for p in self.raw.iterdir() if p.suffix.lower() in AUDIO_EXT) if self.raw.is_dir() else []
        if not files:
            raise FileNotFoundError(f"raw/{self.day}/ 下没有音频")
        self.files = {}
        for f in files:
            base = f.stem
            keys = stage_keys(_sha256(f), self.cfg)
            have = {}
            for stage in ("asr", "diar", "embed", "overlap"):
                c = None if self.force else _read(self.cpath(base, stage))
                have[stage] = c if c and c.get("key") == keys[stage] else None
            need = {s for s, v in have.items() if v is None}
            # ASR 或分离重算过，下游必须跟着重算。
            if need & {"asr", "diar"}:
                need.add("embed")
            if "diar" in need:
                need.add("overlap")
            if not self.cfg["overlap"]:
                need.discard("overlap")
            self.files[base] = {"src": f, "keys": keys, "have": have, "need": need, "audio": None, "samples": None}
        return self.files

    def plan_cached(self):
        """只重算认人：不读录音、不跑任何阶段，按 cache/<日期>/ 里成套的缓存（asr、diar、embed，开着重叠支路时还有 overlap）。"""
        bases = sorted(p.name[len("gpu-"):-len(".embed.json")] for p in self.cache.glob("gpu-*.embed.json")) if self.cache.is_dir() else []
        if not bases:
            raise FileNotFoundError(f"cache/{self.day}/ 下没有 GPU 方案缓存，不能只重算认人")
        self.files = {}
        for base in bases:
            have = {stage: _read(self.cpath(base, stage)) for stage in ("asr", "diar", "embed", "overlap")}
            if not (have["asr"] and have["diar"] and have["embed"]):
                raise FileNotFoundError(f"{base} 的 GPU 方案缓存不全，不能只重算认人")
            keys = downstream_keys(have["asr"].get("key"), have["diar"].get("key"), self.cfg)
            if have["embed"].get("key") != keys["embed"]:
                raise ValueError(f"{base} 的 embed 缓存与 asr/diar 缓存不配套，不能只重算认人")
            if self.cfg["overlap"] and (not have["overlap"] or have["overlap"].get("key") != keys["overlap"]):
                raise ValueError(f"{base} 的 overlap 缓存缺失或不配套，不能只重算认人")
            self.files[base] = {"src": None, "keys": keys, "have": have, "need": set(), "audio": None, "samples": None}
        return self.files

    def relabel(self):
        """只重算认人与渲染，几秒钟；缓存不全时报错，不转写。"""
        t_day = time.perf_counter()
        from .voiceprint import load_voiceprints

        t = time.perf_counter()
        people = load_voiceprints(self.root, self.cfg, self.log, self.voiceprints)
        self.stats["stage_seconds"]["voiceprints"] = time.perf_counter() - t
        self.plan_cached()
        self.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log("只重算认人（--relabel-only）")
        self.render_all(people, t_day)

    def run(self):
        t_day = time.perf_counter()
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        from .voiceprint import load_voiceprints

        t = time.perf_counter()
        people = load_voiceprints(self.root, self.cfg, self.log, self.voiceprints)
        self.stats["stage_seconds"]["voiceprints"] = time.perf_counter() - t
        self.plan()
        if any(f["need"] for f in self.files.values()):
            try:
                self.compute()
            finally:
                shutil.rmtree(self.tmp, ignore_errors=True)
        else:
            self.log("ASR、分离、声纹、重叠支路均用缓存")
        self.render_all(people, t_day)

    def render_all(self, people, t_day):
        t = time.perf_counter()
        total = {}
        for base, f in self.files.items():
            ov = f["have"]["overlap"]["chunks"] if self.cfg["overlap"] else None
            utts, st = compose(f["have"]["embed"], people, C.label_params(self.cfg), ov)
            text = render_transcript(utts)
            tmp = self.out / f"{base}.md.part"
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.out / f"{base}.md")
            lines = [ln for ln in text.splitlines() if ln.startswith("[")]
            labels = {}
            for ln in lines:
                lab = ln.split("] ", 1)[1].split(" ", 1)[0] if "] " in ln else ""
                if lab.startswith("<"):
                    continue
                labels[lab] = labels.get(lab, 0) + 1
            fs = self.stats["files"].setdefault(base, {})
            fs.update(st, lines=len(lines), labels=labels, sentences=len(f["have"]["asr"]["sentences"]),
                      pieces=len(f["have"]["embed"]["pieces"]), diar_spans=len(f["have"]["diar"]["spans"]),
                      computed_now=sorted(f["need"]),
                      stages={k: (f["have"][k] or {}).get("meta") for k in ("asr", "diar", "embed", "overlap")})
            for k, v in labels.items():
                total[k] = total.get(k, 0) + v
            self.log(f"  {base}.md：{len(lines)} 行，{' '.join(f'{k} {v}' for k, v in sorted(labels.items()))}；"
                     f"分离 {st['speakers']} 个说话人；重叠支路 {st['overlap_accepted']}/{st['overlap_chunks']} 块采用")
        self.stats["stage_seconds"]["label_render"] = time.perf_counter() - t
        self.stats["total_seconds"] = time.perf_counter() - t_day
        self.stats["labels"] = total
        _write(self.cache / "gpu-stats.json", self.stats)
        # 最近一次运行的摘要：只有日期、时间与耗时，不含转写内容。
        (self.root / "cache" / ".last-gpu-run").write_text(f"{self.day}，{time.strftime('%Y-%m-%d %H:%M:%S')} 完成，耗时 {self.stats['total_seconds']:.0f}s\n")
        self.log(f"{self.day} 共 {len(self.files)} 个文件，{self.stats['total_seconds']:.0f}s")

    def gpus(self):
        from .gpu import pick_gpus, query_gpus

        allowed = set(self.cfg["gpus"]) if self.cfg["gpus"] is not None else None
        gs = query_gpus()
        picked = pick_gpus(gs, self.cfg["minFreeMiB"], allowed)
        busy = [g["index"] for g in gs if g["index"] not in picked]
        if busy:
            self.log(f"跳过有其他进程占用或显存不足的卡：{busy}")
        if not picked:
            raise NoGpu("没有空闲 GPU（nvidia-smi 显示都被占用），未转写")
        return picked

    def decode(self):
        from .stages import decode_to_f32

        self.tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
        todo = [(b, f) for b, f in self.files.items() if f["need"]]
        t0 = time.perf_counter()

        def one(item):
            b, f = item
            path = self.tmp / f"{b}.f32"
            n = decode_to_f32(f["src"], path)
            return b, path, n

        with cf.ThreadPoolExecutor(4) as ex:
            for b, path, n in ex.map(one, todo):
                self.files[b]["audio"], self.files[b]["samples"] = path, n
                self.stats["files"].setdefault(b, {})["audio_seconds"] = n / 16000
        self.stats["stage_seconds"]["decode"] = time.perf_counter() - t0
        self.log(f"  解码 {len(todo)} 个文件 {time.perf_counter() - t0:.0f}s")

    def compute(self):
        from . import stages

        gpu_needed = any(f["need"] & {"asr", "diar", "overlap"} for f in self.files.values())
        gpus = self.gpus() if gpu_needed else []
        self.decode()
        ctx = mp.get_context("spawn")
        workers = {g: cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=stages.worker_init, initargs=(g, self.cfg)) for g in gpus}
        cpu = cf.ThreadPoolExecutor(2)
        idle = set(gpus)
        running = {}
        done = {(b, s) for b, f in self.files.items() for s in ("asr", "diar", "embed", "overlap") if s not in f["need"]}
        pending = {(b, s) for b, f in self.files.items() for s in f["need"]}
        sec = self.stats["stage_seconds"]

        def ready(job):
            b, s = job
            if s == "embed":
                return (b, "asr") in done and (b, "diar") in done
            if s == "overlap":
                return (b, "diar") in done
            return True

        def cost(job):
            b, s = job
            return COST.get(s, 0) * self.files[b]["samples"]

        try:
            while pending or running:
                for job in sorted((j for j in pending if ready(j)), key=cost, reverse=True):
                    b, s = job
                    f = self.files[b]
                    if s == "embed":
                        fut = cpu.submit(embed_stage, f["audio"], f["have"]["asr"]["sentences"], f["have"]["diar"]["spans"], self.cfg)
                        running[fut] = (job, None)
                    elif idle:
                        g = min(idle)
                        idle.discard(g)
                        if s == "asr":
                            fut = workers[g].submit(stages.run_asr, str(f["audio"]))
                        elif s == "diar":
                            fut = workers[g].submit(stages.run_diar, str(f["audio"]), str(self.tmp / f"{b}.diar.json"))
                        else:
                            chunks = overlap_chunks(f["have"]["diar"]["spans"], self.cfg)
                            fut = workers[g].submit(stages.run_overlap, str(f["audio"]), chunks)
                        running[fut] = (job, g)
                    else:
                        continue
                    pending.discard(job)
                    self.log(f"  开始 {s} {b}" + (f"（GPU {running[fut][1]}）" if running[fut][1] is not None else "（CPU）"))
                if not running:
                    raise RuntimeError("调度出错：没有可运行的阶段")
                finished, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    (b, s), g = running.pop(fut)
                    if g is not None:
                        idle.add(g)
                    r = fut.result()
                    f = self.files[b]
                    if s == "embed":
                        meta = {"seconds": r.pop("seconds")}
                        value = r
                    else:
                        value = r["result"]
                        meta = {"seconds": r["seconds"], "gpu": g, "peak_mib": r["peak_mib"]}
                        gs = self.stats["gpus"].setdefault(str(g), {"peak_mib": 0, "busy_seconds": 0.0})
                        gs["peak_mib"] = max(gs["peak_mib"], r["peak_mib"] or 0)
                        gs["busy_seconds"] += r["seconds"]
                    # 耗时等计数随缓存保存，之后用缓存的运行也能查到当初的耗时。
                    for k in ("separation_seconds", "asr_seconds", "embedding_seconds", "vad_segments", "vad_seconds"):
                        if k in value:
                            meta[k] = value.pop(k)
                    meta["computed"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    value["meta"] = meta
                    seconds = meta["seconds"]
                    sec[s] = sec.get(s, 0.0) + seconds
                    value["key"] = f["keys"][s]
                    _write(self.cpath(b, s), value)
                    f["have"][s] = value
                    done.add((b, s))
                    self.log(f"  完成 {s} {b} {seconds:.0f}s" + (f"，显存峰值 {r['peak_mib']} MiB" if g is not None else ""))
        finally:
            for w in workers.values():
                w.shutdown(cancel_futures=True)
            cpu.shutdown(cancel_futures=True)


def transcribe_day(root, day, force=False, log=print, relabel_only=False, voiceprints=None):
    d = Day(root, day, force, log, voiceprints)
    d.relabel() if relabel_only else d.run()
