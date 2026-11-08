// 登记声纹的计算部分（纯函数，单测用合成数据）：DiariZen 说话人簇的统计、挑登记片段、登记片段的相似度检查。
// 输入是 GPU 方案缓存 cache/<日期>/gpu-<文件名>.diar.json / .embed.json 的内容：
//   spans: [[start, end, speaker]]（DiariZen，允许重叠）；pieces: [{ start, end, text, speaker }]（句子归属后的段）；
//   piece_embs / excl_embs：CAM++ 声纹；excl: [[start, end, speaker]] 只有这一人在说的片段。
import { cos, meanEmb } from './embedding.mjs';
import { decide, labelOne } from './label.mjs';
import { formatTimestamp } from './render.mjs';

/** "95"、"95.5"、"1:35"、"0:01:35"、"1:02:03.5" → 秒。 */
export function parseTime(s) {
  const str = String(s).trim();
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(str)) throw new Error(`时间格式应为秒数或 h:mm:ss：${s}`);
  return str.split(':').reduce((acc, x) => acc * 60 + Number(x), 0);
}

/** 可能重叠的说话区间 → 互不重叠的小段 [start, end, Set(说话人)]，只含有人说话的段。与 engine 的 activity_regions 一致。 */
export function activityRegions(spans) {
  const events = [];
  for (const [s, e, spk] of spans) if (e > s) events.push([s, 1, spk], [e, -1, spk]);
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const active = new Map();
  const out = [];
  let prev = null;
  for (const [t, kind, spk] of events) {
    if (prev !== null && t > prev && active.size) {
      const cur = [...active.keys()].sort();
      const last = out[out.length - 1];
      if (last && last[2].join('\u0000') === cur.join('\u0000') && Math.abs(last[1] - prev) < 1e-9) last[1] = t;
      else out.push([prev, t, cur]);
    }
    const n = (active.get(spk) ?? 0) + kind;
    if (n <= 0) active.delete(spk); else active.set(spk, n);
    prev = t;
  }
  return out;
}

/** 每个说话人的平均声纹：只有他一人在说的片段按时长加权。返回 Map<说话人, 向量>。 */
export function speakerCentroids(excl, exclEmbs) {
  const groups = new Map();
  excl.forEach(([s, e, spk], i) => {
    if (!exclEmbs[i]) return;
    const g = groups.get(spk) ?? { vs: [], ws: [] };
    g.vs.push(exclEmbs[i]);
    g.ws.push(e - s);
    groups.set(spk, g);
  });
  return new Map([...groups].map(([spk, g]) => [spk, meanEmb(g.vs, g.ws)]));
}

/**
 * 干净的单人句：归了说话人、时长在 [minSec, maxSec]、期间没有别的说话人在说（DiariZen 判为单人、无重叠），
 * 且这人自己的说话区间盖住至少一半。返回 piece 下标数组。
 */
export function soloPieces(pieces, spans, { minSec = 1.5, maxSec = Infinity } = {}) {
  const regions = activityRegions(spans);
  const firstAfter = (t) => { let lo = 0; let hi = regions.length; while (lo < hi) { const m = (lo + hi) >> 1; if (regions[m][1] <= t) lo = m + 1; else hi = m; } return lo; };
  const out = [];
  pieces.forEach((p, i) => {
    const dur = p.end - p.start;
    if (p.speaker === null || p.speaker === undefined || dur < minSec || dur > maxSec) return;
    let own = 0;
    for (let k = firstAfter(p.start); k < regions.length && regions[k][0] < p.end; k += 1) {
      const [s, e, who] = regions[k];
      const d = Math.min(e, p.end) - Math.max(s, p.start);
      if (d <= 0) continue;
      if (who.length !== 1 || who[0] !== p.speaker) return;
      own += d;
    }
    if (own >= 0.5 * dur) out.push(i);
  });
  return out;
}

/**
 * 按时间段挑登记片段：完全落在 [start, end] 内的干净单人句（≥ minSec）。
 * 一段里常有几个人轮流说，缺省只取单人说话时长最多的那个簇；cluster 指定簇，allSpeakers 取全部。
 * 返回 { picked: [下标], bySpeaker: [{ speaker, count, seconds }]（按时长降序）, speaker }。
 */
export function pickRange(pieces, spans, start, end, { minSec = 1.5, cluster, allSpeakers = false } = {}) {
  const inside = soloPieces(pieces, spans, { minSec }).filter((i) => pieces[i].start >= start - 1e-6 && pieces[i].end <= end + 1e-6);
  const by = new Map();
  for (const i of inside) {
    const g = by.get(pieces[i].speaker) ?? { speaker: pieces[i].speaker, count: 0, seconds: 0 };
    g.count += 1;
    g.seconds += pieces[i].end - pieces[i].start;
    by.set(pieces[i].speaker, g);
  }
  const bySpeaker = [...by.values()].sort((a, b) => b.seconds - a.seconds);
  const speaker = allSpeakers ? null : (cluster ?? bySpeaker[0]?.speaker ?? null);
  return { picked: inside.filter((i) => speaker === null || pieces[i].speaker === speaker), bySpeaker, speaker };
}

/**
 * 把一个簇登记成某人：挑最干净的 max 个片段——干净单人句、时长在 [minSec, maxSec]、
 * 与簇中心声纹的余弦最高。返回 [{ i, score }]，按时间排序。
 */
export function pickCluster(pieces, pieceEmbs, spans, speaker, centroid, { max = 20, minSec = 1.5, maxSec = 10 } = {}) {
  return soloPieces(pieces, spans, { minSec, maxSec })
    .filter((i) => pieces[i].speaker === speaker && pieceEmbs[i])
    .map((i) => ({ i, score: cos(pieceEmbs[i], centroid) }))
    .sort((a, b) => b.score - a.score || a.i - b.i)
    .slice(0, max)
    .sort((a, b) => a.i - b.i);
}

const pad2 = (n) => String(n).padStart(2, '0');

/** 区间列表按（名义）钟点分小时累计秒数：clock 是文件开始的当天秒数。返回 [{ hour: '09–10', seconds }]，按时间排序。 */
export function hourly(intervals, clock) {
  const acc = new Map();
  for (const [s, e] of intervals) {
    let t = clock + s;
    const stop = clock + e;
    while (t < stop) {
      const h = Math.floor(t / 3600);
      const next = Math.min(stop, (h + 1) * 3600);
      acc.set(h, (acc.get(h) ?? 0) + next - t);
      t = next;
    }
  }
  return [...acc].sort((a, b) => a[0] - b[0]).map(([h, sec]) => ({ hour: `${pad2(h % 24)}–${pad2((h + 1) % 24)}`, seconds: Math.round(sec) }));
}

/**
 * 代表句：与簇中心最像的句子，干净单人句优先，彼此至少隔 gapSec 秒，取 n 条；不够就放宽间隔。
 * 返回 [{ i, score }]（按时间排序）。
 */
export function representatives(pieces, pieceEmbs, spans, speaker, centroid, { n = 5, gapSec = 120 } = {}) {
  const solo = new Set(soloPieces(pieces, spans, { minSec: 1.5 }));
  const score = (i) => (centroid ? cos(pieceEmbs[i], centroid) : 0);
  const ranked = pieces.map((_, i) => i).filter((i) => pieces[i].speaker === speaker && pieceEmbs[i])
    .map((i) => ({ i, score: score(i), solo: solo.has(i) }))
    .sort((a, b) => (b.solo - a.solo) || b.score - a.score || a.i - b.i);
  for (const gap of [gapSec, gapSec / 4, 0]) {
    const out = [];
    for (const c of ranked) {
      if (out.length >= n) break;
      if (out.every((o) => Math.abs(pieces[o.i].start - pieces[c.i].start) >= gap)) out.push(c);
    }
    if (out.length >= Math.min(3, ranked.length) || gap === 0) return out.sort((a, b) => a.i - b.i).map(({ i, score: sc }) => ({ i, score: sc }));
  }
  return [];
}

const round3 = (x) => Math.round(x * 1000) / 1000;

/**
 * 一个文件的簇报告。data: { spans, pieces, pieceEmbs, excl, exclEmbs }；people: 声纹（含 label）；clock: 文件开始的当天秒数。
 * 不含句子文字，withText 为真时代表句附上文字。
 */
export function clusterReport(data, people, { clock = 0, withText = false } = {}) {
  const { spans, pieces, pieceEmbs, excl, exclEmbs } = data;
  const cents = speakerCentroids(excl, exclEmbs);
  const clusterLabel = new Map([...cents].map(([spk, v]) => [spk, decide(v, people).label]));
  const speakers = [...new Set(spans.map((s) => s[2]))].sort((a, b) => String(a).localeCompare(String(b), 'en', { numeric: true }));
  return speakers.map((spk) => {
    const own = spans.filter((s) => s[2] === spk);
    const mine = pieces.map((_, i) => i).filter((i) => pieces[i].speaker === spk);
    const lines = {};
    for (const i of mine) {
      const lab = labelOne(pieceEmbs[i], spk, clusterLabel, people);
      lines[lab] = (lines[lab] ?? 0) + 1;
    }
    const c = cents.get(spk) ?? null;
    const top = c ? people.map((p) => ({ name: p.name, score: round3(cos(c, p.vec)) })).sort((a, b) => b.score - a.score).slice(0, 3) : [];
    const reps = representatives(pieces, pieceEmbs, spans, spk, c);
    return {
      speaker: spk,
      seconds: Math.round(own.reduce((a, s) => a + s[1] - s[0], 0)),
      soloSeconds: Math.round(excl.filter((x) => x[2] === spk).reduce((a, x) => a + x[1] - x[0], 0)),
      sentences: mine.length,
      label: clusterLabel.get(spk) ?? null,
      lineLabels: lines,
      top,
      hours: hourly(own, clock),
      samples: reps.map(({ i, score }) => ({ at: `[${formatTimestamp(pieces[i].start)}]`, start: round3(pieces[i].start), end: round3(pieces[i].end),
        score: round3(score), ...(withText ? { text: pieces[i].text } : {}) })),
    };
  });
}

/**
 * 登记片段的相似度检查。people: [{ name, clips: [{ id, vec }] }]。
 * 每人：片段两两余弦（均值、最小）、声纹中心与别人中心的最高余弦；
 * 每个片段：与本人其他片段中心的余弦（留一）、与别人中心的最高余弦。更像别人、或与本人太不像（< lowOwn）的片段算可疑。
 */
export function checkEnrollment(people, { lowOwn = 0.3, confusable = 0.6 } = {}) {
  const cents = new Map(people.map((p) => [p.name, meanEmb(p.clips.map((c) => c.vec))]));
  return people.map((p) => {
    const vs = p.clips.map((c) => c.vec);
    const pairs = [];
    for (let i = 0; i < vs.length; i += 1) for (let j = i + 1; j < vs.length; j += 1) pairs.push(cos(vs[i], vs[j]));
    const others = people.filter((q) => q.name !== p.name);
    const nearest = others.map((q) => ({ name: q.name, score: round3(cos(cents.get(p.name), cents.get(q.name))) })).sort((a, b) => b.score - a.score)[0] ?? null;
    const clips = p.clips.map((c, k) => {
      const rest = vs.filter((_, j) => j !== k);
      const own = rest.length ? round3(cos(c.vec, meanEmb(rest))) : null;
      const other = others.map((q) => ({ name: q.name, score: round3(cos(c.vec, cents.get(q.name))) })).sort((a, b) => b.score - a.score)[0] ?? null;
      const reasons = [];
      if (own !== null && other && other.score > own) reasons.push(`更像${other.name}`);
      if (own !== null && own < lowOwn) reasons.push('与本人其他片段不像');
      return { id: c.id, own, other, suspicious: reasons.length > 0, reasons };
    });
    const advice = [];
    for (const c of clips.filter((x) => x.suspicious)) advice.push(`片段 ${c.id}：${c.reasons.join('、')}，建议核对后移除`);
    if (p.clips.length < 3) advice.push(`只有 ${p.clips.length} 个片段，建议补到 5 个以上`);
    if (nearest && nearest.score >= confusable) advice.push(`声纹与${nearest.name}接近（${nearest.score}），容易混，建议多登记两人各自的清晰片段`);
    return {
      name: p.name,
      clips: p.clips.length,
      pairMean: pairs.length ? round3(pairs.reduce((a, b) => a + b, 0) / pairs.length) : null,
      pairMin: pairs.length ? round3(Math.min(...pairs)) : null,
      nearest,
      suspicious: clips.filter((x) => x.suspicious),
      advice,
    };
  });
}
