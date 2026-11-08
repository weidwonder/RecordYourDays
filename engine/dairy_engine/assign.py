"""DiariZen 说话区间的整理与句子归属。

spans 为 [(start, end, speaker)]，speaker 是 DiariZen 的匿名编号（只在本文件内有意义），允许互相重叠。

句子归属规则：
1. 每个单元（字 / 英文词）取时间中点，中点处恰好一人在说就记为这人，没人或多人同时在说记为「未定」。
2. 按单元顺序把同一人的连续单元归成一段，「未定」单元并入前一段（开头的并入后一段）。
3. 字数少于 SPLIT_MIN_UNITS 或时长短于 SPLIT_MIN_SEC 的段并入相邻段（优先并入前一段），再合并相邻同人段。
4. 合并后仍有两段以上：在换人处把句子切开，每段各自归给它的说话人。
5. 只剩一段或没有任何已定单元：整句按重叠时长归给重叠最长的说话人，重叠不足句长一半时不归任何人（None），
   之后只按这句自己的声纹认人。这条与 CPU 方案 src/diarize.mjs 的 assignSpeakers 一致。
"""
import math

from .sentences import join_units

SPLIT_MIN_UNITS = 4
SPLIT_MIN_SEC = 1.0
MIN_OVERLAP_RATIO = 0.5


def activity_regions(spans):
    """把可能重叠的说话区间切成互不重叠的小段：[(start, end, frozenset(说话人))]，只含有人说话的段，按时间排序。"""
    events = []
    for s, e, spk in spans:
        if e > s:
            events.append((s, 1, spk))
            events.append((e, -1, spk))
    events.sort(key=lambda x: (x[0], x[1]))
    active = {}
    out = []
    prev = None
    for t, kind, spk in events:
        if prev is not None and t > prev and active:
            cur = frozenset(k for k, n in active.items() if n > 0)
            if cur:
                if out and out[-1][2] == cur and abs(out[-1][1] - prev) < 1e-9:
                    out[-1] = (out[-1][0], t, cur)
                else:
                    out.append((prev, t, cur))
        active[spk] = active.get(spk, 0) + kind
        if active[spk] <= 0:
            active.pop(spk)
        prev = t
    return out


def exclusive_pieces(regions, min_dur=0.0):
    """只有一人在说的段：[(start, end, speaker)]，时长不足 min_dur 的丢掉。用于算每个说话人的平均声纹。"""
    return [(s, e, next(iter(a))) for s, e, a in regions if len(a) == 1 and e - s >= min_dur]


def two_speaker_regions(regions, min_dur=0.5):
    """恰好两人同时说话的区间，同一对人相邻的段合并；返回 [(start, end, (A, B))]，A、B 排好序。"""
    out = []
    for s, e, a in regions:
        if len(a) != 2:
            continue
        pair = tuple(sorted(a))
        if out and out[-1][2] == pair and abs(out[-1][1] - s) < 1e-9:
            out[-1] = (out[-1][0], e, pair)
        else:
            out.append((s, e, pair))
    return [r for r in out if r[1] - r[0] >= min_dur]


def chunk_region(start, end, max_len=10.0):
    """把区间等分成不超过 max_len 秒的块。"""
    n = max(1, math.ceil((end - start) / max_len - 1e-9))
    step = (end - start) / n
    return [(start + i * step, end if i == n - 1 else start + (i + 1) * step) for i in range(n)]


class Timeline:
    """按时间查询某一时刻在说话的人。"""

    def __init__(self, regions):
        self.regions = regions
        self.starts = [r[0] for r in regions]

    def at(self, t):
        import bisect

        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.regions[i][0] <= t < self.regions[i][1]:
            return self.regions[i][2]
        return frozenset()

    def overlap_by_speaker(self, start, end):
        """[start, end) 内每个说话人的说话时长。"""
        import bisect

        out = {}
        i = max(0, bisect.bisect_right(self.starts, start) - 1)
        while i < len(self.regions) and self.regions[i][0] < end:
            s, e, a = self.regions[i]
            d = min(e, end) - max(s, start)
            if d > 0:
                for spk in a:
                    out[spk] = out.get(spk, 0.0) + d
            i += 1
        return out


def _runs(units, owners):
    runs = []
    for u, o in zip(units, owners):
        if runs and (o is None or o == runs[-1]["spk"]):
            runs[-1]["units"].append(u)
        elif not runs and o is None:
            runs.append({"spk": None, "units": [u]})
        elif runs and runs[-1]["spk"] is None:
            runs[-1]["spk"] = o
            runs[-1]["units"].append(u)
        else:
            runs.append({"spk": o, "units": [u]})
    return runs


def _small(run):
    us = run["units"]
    return len(us) < SPLIT_MIN_UNITS or us[-1][2] - us[0][1] < SPLIT_MIN_SEC


def _merge_runs(runs):
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if _small(r):
                j = i - 1 if i > 0 else i + 1
                lo, hi = sorted((i, j))
                keep = runs[j]["spk"]
                runs[lo] = {"spk": keep, "units": runs[lo]["units"] + runs[hi]["units"]}
                del runs[hi]
                changed = True
                break
        merged = []
        for r in runs:
            if merged and merged[-1]["spk"] == r["spk"]:
                merged[-1]["units"] = merged[-1]["units"] + r["units"]
            else:
                merged.append(r)
        runs[:] = merged
    return runs


def whole_sentence_speaker(timeline, start, end):
    dur = end - start
    by = timeline.overlap_by_speaker(start, end)
    if not by or dur <= 0:
        return None
    spk, ov = max(sorted(by.items()), key=lambda kv: kv[1])
    return spk if ov >= MIN_OVERLAP_RATIO * dur else None


def piece(units, speaker):
    return {"start": units[0][1], "end": max(u[2] for u in units), "text": join_units([u[0] for u in units]),
            "units": units, "speaker": speaker}


def assign_sentence(sentence, timeline):
    """一句 → 一个或多个 piece：{start, end, text, units, speaker}。规则见模块说明。"""
    units = sentence["units"]
    owners = []
    for u in units:
        a = timeline.at((u[1] + u[2]) / 2)
        owners.append(next(iter(a)) if len(a) == 1 else None)
    runs = _merge_runs(_runs(units, owners)) if any(o is not None for o in owners) else []
    if len(runs) >= 2:
        return [piece(r["units"], r["spk"]) for r in runs]
    return [piece(units, whole_sentence_speaker(timeline, sentence["start"], sentence["end"]))]


def assign_sentences(sentences, spans):
    timeline = Timeline(activity_regions(spans))
    out = []
    for s in sentences:
        out.extend(assign_sentence(s, timeline))
    return out
