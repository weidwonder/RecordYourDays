// 给每句定标签（登记过的人的显示名，或没认出来的人的标签，缺省「其他」）。
import { assignSpeakers } from './diarize.mjs';
import { cos, meanEmb } from './embedding.mjs';

export const DEFAULT_UNKNOWN_LABEL = '其他';
/** 没认出来的人的标签。live binding：setUnknownLabel 之后各处读到的都是新值。 */
export let OTHER = DEFAULT_UNKNOWN_LABEL;

/** 标签不能为空、不能含空白（转写行按空格分出标签），也不能以 < 开头（与无语音标记混淆）。 */
export function checkLabel(label, what = '标签') {
  if (typeof label !== 'string' || !label || /\s/.test(label) || label.startsWith('<')) throw new Error(`${what}不能为空、含空白或以 < 开头：${label}`);
  return label;
}

/** 设置没认出来的人的标签（dairy.config.json 的 unknownLabel），在认人和渲染之前调用。 */
export function setUnknownLabel(label) {
  OTHER = checkLabel(label, 'unknownLabel ');
}
// 逐句结果足够有把握时才推翻簇映射：最高分 ≥ CONFIDENT 且领先第二名 ≥ MARGIN。
const CONFIDENT = 0.5;
const MARGIN = 0.1;

/**
 * 一个 embedding 对所有登记的人（family 与 known）打分：取最高分者，达到其本人阈值则为其显示名，否则「其他」。
 * people: [{ name, label?, threshold, vec }]，label 缺省为 name。返回的 name 是最高分者的名字（不论是否达到阈值）。
 * emb 为 null 时 top/second 为 -1。
 */
export function decide(emb, people) {
  if (!emb) return { label: OTHER, top: -1, second: -1 };
  let best;
  let top = -Infinity;
  let second = -1;
  for (const p of people) {
    const s = cos(emb, p.vec);
    if (s > top) {
      if (best) second = top;
      top = s;
      best = p;
    } else if (s > second) second = s;
  }
  return { label: top >= best.threshold ? (best.label ?? best.name) : OTHER, top, second, name: best.name };
}

/** 每个簇用其 span embedding 按时长加权求中心再打分。返回 Map<簇编号, 标签>；没有可用 embedding 的簇不在其中。 */
export function mapClusters(spans, spanEmbs, people) {
  const groups = new Map();
  spans.forEach((s, i) => {
    if (!spanEmbs[i]) return;
    const g = groups.get(s.speaker) ?? { vs: [], ws: [] };
    g.vs.push(spanEmbs[i]);
    g.ws.push(s.end - s.start);
    groups.set(s.speaker, g);
  });
  const out = new Map();
  for (const [k, g] of groups) out.set(k, decide(meanEmb(g.vs, g.ws), people).label);
  return out;
}

/** 一句的标签：逐句判定有把握时用逐句结果，否则随所归簇（cluster 为 null 或簇没有标签时也用逐句结果）。 */
export function labelOne(emb, cluster, clusterLabel, people) {
  const own = decide(emb, people);
  if (own.top >= CONFIDENT && own.top - own.second >= MARGIN) return own.label;
  return cluster !== null && cluster !== undefined && clusterLabel.has(cluster) ? clusterLabel.get(cluster) : own.label;
}

/**
 * 返回与 utts 等长的标签数组。默认随所归簇的标签；逐句判定有把握时用逐句结果；
 * 没归上簇（或簇无 embedding）的句子也用逐句结果。
 * 逐句优先的原因：VAD 句常跨两人，按最大重叠归簇会把一句明显是甲的话归进乙的簇；
 * 簇优先的原因：短句、远场句的逐句分数不稳，簇中心聚合了更多语音。
 */
export function labelUtterances(utts, uttEmbs, spans, spanEmbs, people) {
  const clusterLabel = mapClusters(spans, spanEmbs, people);
  const clusters = assignSpeakers(utts, spans);
  return utts.map((u, i) => labelOne(uttEmbs[i], clusters[i], clusterLabel, people));
}
