"""已 L2 归一化向量的运算，与 src/embedding.mjs 一致。"""
import math


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a, b):
    """两个已归一化向量的余弦。"""
    return sum(x * y for x, y in zip(a, b))


def mean_emb(vs, ws=None):
    """按权重（缺省等权）平均后再归一化。"""
    out = [0.0] * len(vs[0])
    for k, v in enumerate(vs):
        w = ws[k] if ws else 1.0
        for i, x in enumerate(v):
            out[i] += w * x
    return l2(out)
