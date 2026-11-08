"""Paraformer 字级时间戳 + CT 标点 → 句子。

Paraformer 的 text 以空格分隔 token（汉字一个一个、英文按词），timestamp 与 token 一一对应（毫秒，相对送入的片段）。
CT 标点的 punc_array 也与这些 token 一一对应，每个 token 后面跟一个标点（或不跟）。
句子在句末标点（。？！）之后断开，片段（VAD 段）边界也断开。

每句保留逐 token 的「单元」[文本（含其后标点）, 起, 止]，后续按说话人切分句子、剔除重叠区文字时都以单元为准。
"""
import math
import re

SENTENCE_END = set("。？！?!")
_ASCII_WORD = re.compile(r"^[A-Za-z0-9'][A-Za-z0-9'.-]*$")


def _round(x):
    return round(float(x), 3)


def join_units(texts):
    """拼接单元文本：相邻两个英文/数字词之间补空格，汉字之间不补。"""
    out = ""
    prev_ascii = False
    for t in texts:
        if not t:
            continue
        cur_ascii = bool(_ASCII_WORD.match(t[0]))
        if out and prev_ascii and cur_ascii:
            out += " "
        out += t
        bare = t.rstrip("，。？！、,.?!")
        prev_ascii = bool(bare) and bool(_ASCII_WORD.match(bare)) and len(bare) == len(t)
    return out


def make_units(tokens, stamps_ms, puncs, seg_start, seg_end):
    """token 与时间戳、标点对齐成单元 [文本, 起秒, 止秒]（绝对时间）。

    时间戳数量对不上时不猜测对齐：把整段文字当作一个单元，时间取片段起止。
    时间戳被夹到片段内，并保证单调不减。
    """
    if not tokens:
        return []
    if len(stamps_ms) != len(tokens):
        text = join_units([t + (puncs[i] if puncs and len(puncs) == len(tokens) else "") for i, t in enumerate(tokens)])
        return [[text, _round(seg_start), _round(seg_end)]]
    units = []
    last = seg_start
    for i, tok in enumerate(tokens):
        s = min(max(seg_start + stamps_ms[i][0] / 1000, last), seg_end)
        e = min(max(seg_start + stamps_ms[i][1] / 1000, s), seg_end)
        p = puncs[i] if puncs and len(puncs) == len(tokens) else ""
        units.append([tok + p, _round(s), _round(e)])
        last = s
    return units


def sentence_from_units(units):
    return {"start": units[0][1], "end": max(u[2] for u in units), "text": join_units([u[0] for u in units]), "units": units}


def split_sentences(units):
    """单元在句末标点后断句。返回 [{start, end, text, units}]，空文本句丢弃。"""
    out = []
    cur = []
    for u in units:
        cur.append(u)
        if u[0] and u[0][-1] in SENTENCE_END:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [sentence_from_units(c) for c in out if join_units([u[0] for u in c]).strip()]


def punc_symbols(punc_array, punc_list):
    """CT 标点的 id 数组 → 每个 token 后的标点字符串（'_' 与 '<unk>' 表示无标点）。"""
    out = []
    for i in punc_array:
        sym = punc_list[int(i)] if 0 <= int(i) < len(punc_list) else ""
        out.append("" if sym in ("_", "<unk>") else sym)
    return out


def segment_sentences(text, stamps_ms, punc_array, punc_list, seg_start, seg_end):
    """一个 VAD 片段的识别结果 → 句子。punc_array 为 None 时不加标点。"""
    tokens = text.split()
    puncs = punc_symbols(punc_array, punc_list) if punc_array is not None else None
    if puncs is not None and len(puncs) != len(tokens):
        puncs = None
    return split_sentences(make_units(tokens, stamps_ms, puncs, seg_start, seg_end))


def plan_segments(raw, merge_gap, max_sec):
    """FSMN VAD 原始段 → 送识别的段。

    FSMN 常把连续说话切成首尾相接的碎段（切点不在停顿处），单独识别会丢上下文、切坏字词。
    相邻段间隔 ≤ merge_gap 且合并后不超过 max_sec 就合并；仍超过 max_sec 的段等长切开。
    """
    merged = []
    for s, e in raw:
        if merged and s - merged[-1][1] <= merge_gap and e - merged[-1][0] <= max_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        n = max(1, math.ceil((e - s) / max_sec - 1e-9))
        step = (e - s) / n
        out.extend((s + i * step, e if i == n - 1 else s + (i + 1) * step) for i in range(n))
    return out
