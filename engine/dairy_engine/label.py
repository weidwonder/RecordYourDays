"""认人与重叠支路选路。认人规则移植自 src/label.mjs，参数一致。"""
from .vec import cos, mean_emb

DEFAULT_UNKNOWN_LABEL = "其他"
# 没认出来的人的标签；set_unknown_label 改它，decide 与 render 在调用时读取。
OTHER = DEFAULT_UNKNOWN_LABEL
# 逐句结果足够有把握时才推翻簇映射：最高分 ≥ CONFIDENT 且领先第二名 ≥ MARGIN。
CONFIDENT = 0.5
MARGIN = 0.1


def set_unknown_label(label):
    """设置没认出来的人的标签（dairy.config.json 的 unknownLabel），规则同 src/label.mjs 的 checkLabel。"""
    global OTHER
    if not isinstance(label, str) or not label or any(c.isspace() for c in label) or label.startswith("<"):
        raise ValueError("unknownLabel 不能为空、含空白或以 < 开头")
    OTHER = label


def decide(emb, people):
    """一个 embedding 对所有登记的人（family 与 known）打分：取最高分者，达到其本人阈值则为其显示名，否则「其他」。

    people: [{name, label?, threshold, vec}]，label 缺省为 name。"""
    if not emb:
        return {"label": OTHER, "top": -1, "second": -1}
    best = None
    top = float("-inf")
    second = -1
    for p in people:
        s = cos(emb, p["vec"])
        if s > top:
            if best is not None:
                second = top
            top = s
            best = p
        elif s > second:
            second = s
    return {"label": best.get("label", best["name"]) if top >= best["threshold"] else OTHER, "top": top, "second": second}


def speaker_centroids(pieces, embs):
    """每个 DiariZen 说话人：只有他一人在说的片段的 embedding 按时长加权平均。pieces: [(start, end, speaker)]。"""
    groups = {}
    for (s, e, spk), v in zip(pieces, embs):
        if not v:
            continue
        g = groups.setdefault(spk, ([], []))
        g[0].append(v)
        g[1].append(e - s)
    return {spk: mean_emb(vs, ws) for spk, (vs, ws) in groups.items()}


def cluster_labels(centroids, people):
    return {spk: decide(v, people)["label"] for spk, v in centroids.items()}


def label_one(emb, speaker, clusters, people, confident=CONFIDENT, margin=MARGIN):
    """默认随所属说话人的标签；逐句判定有把握时用逐句结果；没有归属（或归属的人没有声纹）时也用逐句结果。"""
    own = decide(emb, people)
    if own["top"] >= confident and own["top"] - own["second"] >= margin:
        return own["label"]
    if speaker is not None and speaker in clusters:
        return clusters[speaker]
    return own["label"]


def label_pieces(pieces, embs, clusters, people, confident=CONFIDENT, margin=MARGIN):
    return [label_one(v, p["speaker"], clusters, people, confident, margin) for p, v in zip(pieces, embs)]


def choose_routes(route_embs, pair, centroids, min_score, min_margin):
    """重叠支路选路：两路分离结果各自对应 pair 里的哪个说话人。

    route_embs: 两路的 embedding；pair: (A, B)；centroids: 说话人平均声纹。
    两种配对各算余弦之和，取大的那种；它领先另一种 ≥ min_margin 且配上的两对余弦都 ≥ min_score 才算有把握。
    返回 {ok, order, reason, scores}；order[i] 是第 i 路对应的说话人。
    """
    a, b = pair
    if any(not v for v in route_embs):
        return {"ok": False, "reason": "no-embedding"}
    if a not in centroids or b not in centroids:
        return {"ok": False, "reason": "no-centroid"}
    s = [[cos(v, centroids[spk]) for spk in (a, b)] for v in route_embs]
    straight = s[0][0] + s[1][1]
    crossed = s[0][1] + s[1][0]
    if straight >= crossed:
        order, matched = (a, b), (s[0][0], s[1][1])
    else:
        order, matched = (b, a), (s[0][1], s[1][0])
    margin = abs(straight - crossed)
    if margin < min_margin:
        return {"ok": False, "reason": "ambiguous", "margin": margin}
    if min(matched) < min_score:
        return {"ok": False, "reason": "low-score", "margin": margin}
    return {"ok": True, "order": order, "margin": margin, "matched": matched}
