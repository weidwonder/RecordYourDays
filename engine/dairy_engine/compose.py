"""缓存的中间结果 + 声纹 → 带标签的句子。每次运行都现算，改了 speakers.json 只需重跑这一步。"""
import bisect

from .label import choose_routes, cluster_labels, label_one, label_pieces, speaker_centroids
from .sentences import join_units


def _drop_units(utts, chunks):
    """去掉中点落在任一已采用重叠块内的单元；单元删光的句子整句去掉。"""
    if not chunks:
        return utts
    chunks = sorted(chunks)
    starts = [c[0] for c in chunks]

    def inside(t):
        i = bisect.bisect_right(starts, t) - 1
        return i >= 0 and chunks[i][0] <= t < chunks[i][1]

    out = []
    for u in utts:
        if u["end"] <= chunks[0][0] or u["start"] >= chunks[-1][1]:
            out.append(u)
            continue
        kept = [x for x in u["units"] if not inside((x[1] + x[2]) / 2)]
        if len(kept) == len(u["units"]):
            out.append(u)
        elif kept:
            out.append({**u, "start": kept[0][1], "end": max(x[2] for x in kept), "text": join_units([x[0] for x in kept]), "units": kept})
    return out


def compose(cache, people, params, overlap=None):
    """cache: {pieces, piece_embs, excl, excl_embs}；overlap: 重叠支路缓存的 chunks 列表或 None（关闭）。

    返回 (utts, stats)。utts: [{start, end, text, label}]；stats 只含计数。
    """
    excl = [tuple(x) for x in cache["excl"]]
    centroids = speaker_centroids(excl, cache["excl_embs"])
    clusters = cluster_labels(centroids, people)
    pieces = cache["pieces"]
    labels = label_pieces(pieces, cache["piece_embs"], clusters, people, params["confident"], params["margin"])
    utts = [{"start": p["start"], "end": p["end"], "text": p["text"], "units": p["units"], "label": lab} for p, lab in zip(pieces, labels)]
    stats = {"speakers": len({x[2] for x in excl} | {p["speaker"] for p in pieces if p["speaker"] is not None}),
             "overlap_chunks": 0, "overlap_seconds": 0.0, "overlap_accepted": 0, "overlap_accepted_seconds": 0.0, "overlap_skipped": {}}
    if overlap is None:
        return [_public(u) for u in utts], stats
    accepted = []
    extra = []
    for c in overlap:
        stats["overlap_chunks"] += 1
        stats["overlap_seconds"] += c["end"] - c["start"]
        texts = [r["text"] for r in c["routes"]]
        if len(texts) != 2 or not all(t.strip() for t in texts):
            reason = "empty-text"
        else:
            res = choose_routes([r["emb"] for r in c["routes"]], tuple(c["pair"]), centroids, params["route_min_score"], params["route_min_margin"])
            reason = None if res["ok"] else res["reason"]
        if reason:
            stats["overlap_skipped"][reason] = stats["overlap_skipped"].get(reason, 0) + 1
            continue
        stats["overlap_accepted"] += 1
        stats["overlap_accepted_seconds"] += c["end"] - c["start"]
        accepted.append((c["start"], c["end"]))
        for r, spk in zip(c["routes"], res["order"]):
            label = label_one(r["emb"], spk, clusters, people, params["confident"], params["margin"])
            extra.append({"start": c["start"], "end": c["end"], "text": r["text"], "label": label})
    utts = _drop_units(utts, accepted)
    return [_public(u) for u in utts] + extra, stats


def _public(u):
    return {"start": u["start"], "end": u["end"], "text": u["text"], "label": u["label"]}
