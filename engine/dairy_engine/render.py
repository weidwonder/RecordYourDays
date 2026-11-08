"""带标签的句子 → 转写稿 markdown。逐条移植 src/render.mjs，两边共用 test/fixtures/render-cases.json。"""
import math

from . import label
NO_SPEECH_WHOLE = "<no speech detected>"
NO_SPEECH_SEGMENT = "<no speech detected in this segment>"
MERGE_GAP_SEC = 2
SILENCE_MARK_SEC = 30


def format_timestamp(sec):
    """秒 → mm:ss，满一小时为 h:mm:ss。"""
    t = max(0, math.floor(sec))
    h, rest = divmod(t, 3600)
    mm, ss = divmod(rest, 60)
    return f"{h}:{mm:02d}:{ss:02d}" if h > 0 else f"{mm:02d}:{ss:02d}"


def merge_adjacent(utts):
    """按开始时间（稳定）排序后，把间隔 ≤ 2 s 的相邻同标签句合并；「其他」不合并。"""
    out = []
    for u in sorted(utts, key=lambda x: x["start"]):
        last = out[-1] if out else None
        if last and last["label"] != label.OTHER and last["label"] == u["label"] and u["start"] - last["end"] <= MERGE_GAP_SEC:
            last["end"] = u["end"]
            last["text"] = last["text"] + u["text"]
        else:
            out.append(dict(u))
    return out


def render_transcript(utts):
    """utts: [{start, end, text, label}]。每行 `[时间] 标签 文本`；超过 30 s 的静音插一行标记。"""
    if not utts:
        return f"{NO_SPEECH_WHOLE}\n"
    lines = []
    prev_end = 0
    for u in merge_adjacent(utts):
        if u["start"] - prev_end > SILENCE_MARK_SEC:
            lines.append(f"[{format_timestamp(prev_end)}] {NO_SPEECH_SEGMENT}")
        lines.append(f"[{format_timestamp(u['start'])}] {u['label']} {u['text']}")
        prev_end = u["end"]
    return "\n".join(lines) + "\n"
