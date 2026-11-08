"""公开集复现：用 GPU 方案的阶段函数跑 AliMeeting Eval 前四场，结果见 alimeeting-2026-11.json。

报告：
- DER（collar 0 / 0.25，口径同 pyannote）：DiariZen 整场分离结果 vs 标注。
- 句子级 CER：ASR 远场子集（ali-asr.json，非重叠标注句）作参考；
  假设文字取 GPU 方案整场 VAD+Paraformer+标点后、字级时间戳中点落在该参考句起止内的字。
  这同时考查识别与时间戳，区别于用真值句边界切片后单独识别的 CER。
- 对照：同一 ASR 代码直接识别真值句边界切出的片段（oracle_segment_cer），确认模型本身的识别率。
- 重叠支路：恰好两人重叠区的块数与时长、两路识别耗时。

只读公开数据（--data 目录：ali-meetings.json 列出会议的音频路径、时长与标注 spans，ali-asr.json 列出参考句片段），结果写 JSON。
用法：PYTHONPATH=engine <envDir>/venv/bin/python engine/eval/alimeeting.py --data <目录> --out <json> [--gpu 0]
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import diarization_metrics  # noqa: E402

def normalized(text):
    from opencc import OpenCC

    text = re.sub(r"<\|.*?\|>|<\s*[A-Za-z_][A-Za-z0-9_|-]*\s*>", "", text)
    text = OpenCC("t2s").convert(unicodedata.normalize("NFKC", text)).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "N"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="AliMeeting 评测数据目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    out_path = Path(a.out).resolve()
    data = Path(a.data).expanduser()
    import soundfile as sf
    from rapidfuzz.distance import Levenshtein

    from dairy_engine import config as C
    from dairy_engine import stages
    from dairy_engine.pipeline import overlap_chunks

    cfg = C.load(a.root)
    stages.worker_init(a.gpu, cfg)
    meetings = json.loads((data / "ali-meetings.json").read_text())
    refs = json.loads((data / "ali-asr.json").read_text())
    by_meeting = {}
    for r in refs:
        mid, idx = r["id"].rsplit("-", 1)
        by_meeting.setdefault(mid, []).append(int(idx))
    result = {"meetings": [], "config": {k: cfg[k] for k in ("asrBatch", "vadMaxSec", "vadMergeGap", "diarizenBatch", "overlapMinSec", "overlapMaxSec")}}
    errors = chars = 0
    with tempfile.TemporaryDirectory(prefix="dairy-engine-eval-") as td:
        for m in meetings:
            audio, rate = sf.read(m["path"], dtype="float32")
            assert rate == 16000
            f32 = Path(td) / f"{m['id']}.f32"
            audio.astype("<f4").tofile(f32)
            asr = stages.run_asr(str(f32))
            diar = stages.run_diar(str(f32), str(Path(td) / f"{m['id']}.diar.json"))
            spans = diar["result"]["spans"]
            chunks = overlap_chunks(spans, cfg)
            ov = stages.run_overlap(str(f32), chunks)
            ref_spans = [(s["start"], s["end"], s["speaker"]) for s in m["spans"]]
            hyp = [tuple(s) for s in spans]
            units = [u for s in asr["result"]["sentences"] for u in s["units"]]
            m_err = m_chars = 0
            for i in by_meeting.get(m["id"], []):
                ref = m["spans"][i]
                hyp_text = "".join(u[0] for u in units if ref["start"] <= (u[1] + u[2]) / 2 < ref["end"])
                r_n, h_n = normalized(ref["text"]), normalized(hyp_text)
                m_err += Levenshtein.distance(r_n, h_n)
                m_chars += len(r_n)
            errors += m_err
            chars += m_chars
            result["meetings"].append({
                "id": m["id"], "audio_seconds": m["duration"],
                "asr_seconds": asr["seconds"], "asr_peak_mib": asr["peak_mib"], "sentences": len(asr["result"]["sentences"]),
                "diar_seconds": diar["seconds"], "diar_peak_mib": diar["peak_mib"],
                "predicted_speakers": len({s[2] for s in hyp}), "reference_speakers": len({s[2] for s in ref_spans}),
                "collar0": diarization_metrics(ref_spans, hyp, m["duration"], 0),
                "collar025": diarization_metrics(ref_spans, hyp, m["duration"], 0.25),
                "cer_sentences": len(by_meeting.get(m["id"], [])), "cer_errors": m_err, "cer_chars": m_chars,
                "overlap_chunks": len(chunks), "overlap_seconds": sum(e - s for s, e, _ in chunks),
                "overlap_run_seconds": ov["seconds"], "overlap_peak_mib": ov["peak_mib"],
            })
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    _, asr_model, punc, punc_list = stages._asr_models()
    clips = [sf.read(r["path"], dtype="float32")[0] for r in refs]
    texts = stages.recognize(asr_model, punc, punc_list, clips, cfg["asrBatch"])
    result["oracle_segment_cer"] = (sum(Levenshtein.distance(normalized(r["text"]), normalized(t[0])) for r, t in zip(refs, texts))
                                    / sum(len(normalized(r["text"])) for r in refs))
    ms = result["meetings"]
    for c in ("collar0", "collar025"):
        ref_total = sum(x[c]["reference_speaker_seconds"] for x in ms)
        result[f"der_{c}"] = sum(x[c]["der"] * x[c]["reference_speaker_seconds"] for x in ms) / ref_total
    result["cer"] = errors / chars
    result["cer_sentences"] = sum(x["cer_sentences"] for x in ms)
    result["cer_chars"] = chars
    result["audio_seconds"] = sum(x["audio_seconds"] for x in ms)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
