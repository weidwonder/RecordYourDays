import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assignSpeakers } from '../src/diarize.mjs';
import { OTHER, decide, labelUtterances, mapClusters } from '../src/label.mjs';

// 声纹取正交轴，第四轴代表陌生人；于是对单位向量 [a, b, c, d]，三人的余弦分别就是 a、b、c。
const people = [
  { name: '我', threshold: 0.35, vec: [1, 0, 0, 0] },
  { name: '伴侣', threshold: 0.4, vec: [0, 1, 0, 0] },
  { name: '孩子', threshold: 0.35, vec: [0, 0, 1, 0] },
];
/** 余弦为 a、b、c 的单位向量（剩余长度放到陌生人轴）。 */
const e = (a, b, c) => [a, b, c, Math.sqrt(1 - a * a - b * b - c * c)];

test('decide：取最高分者，达到其本人阈值才标名字', () => {
  assert.equal(decide(e(0.36, 0.1, 0.1), people).label, '我');
  assert.equal(decide(e(0.1, 0.4, 0.1), people).label, '伴侣');
  assert.equal(decide(e(0.1, 0.2, 0.34), people).label, OTHER);
});

test('decide：最高者不到阈值即「其他」，不退而看第二名', () => {
  // 伴侣 0.39 不到 0.40；我 0.36 虽达到 0.35，也不改判我（与实验方案 d 一致）。
  const d = decide(e(0.36, 0.39, 0.1), people);
  assert.equal(d.label, OTHER);
  assert.ok(Math.abs(d.top - 0.39) < 1e-9 && Math.abs(d.second - 0.36) < 1e-9);
});

test('decide：embedding 为 null 时是「其他」', () => {
  assert.deepEqual(decide(null, people), { label: OTHER, top: -1, second: -1 });
});

test('mapClusters：簇中心按 span 时长加权', () => {
  // 两段 1 s 的我 + 一段 8 s 的伴侣：等权平均会归我，按时长加权归伴侣。
  const spans = [
    { start: 0, end: 1, speaker: 0 },
    { start: 1, end: 2, speaker: 0 },
    { start: 2, end: 10, speaker: 0 },
  ];
  const m = mapClusters(spans, [e(1, 0, 0), e(1, 0, 0), e(0, 1, 0)], people);
  assert.equal(m.get(0), '伴侣');
});

test('mapClusters：中心不到阈值映射为「其他」；没有可用 embedding 的簇不映射', () => {
  const spans = [
    { start: 0, end: 2, speaker: 1 },
    { start: 2, end: 2.2, speaker: 2 },
  ];
  const m = mapClusters(spans, [e(0.2, 0.2, 0.2), null], people);
  assert.equal(m.get(1), OTHER);
  assert.equal(m.has(2), false);
});

test('assignSpeakers：取重叠最长的簇，重叠不足句长一半为 null', () => {
  const spans = [
    { start: 0, end: 2, speaker: 3 },
    { start: 2, end: 5, speaker: 4 },
  ];
  const utts = [{ start: 1, end: 4 }, { start: 4.5, end: 7 }, { start: 8, end: 9 }];
  assert.deepEqual(assignSpeakers(utts, spans), [4, null, null]);
});

test('labelUtterances：默认随簇；逐句最高分 ≥ 0.5 且领先 ≥ 0.1 时改用逐句结果', () => {
  const spans = [{ start: 0, end: 20, speaker: 0 }];
  const spanEmbs = [e(0, 0, 1)]; // 簇 0 → 孩子
  const utts = [
    { start: 0, end: 3 },
    { start: 4, end: 6 },
    { start: 7, end: 9 },
    { start: 10, end: 11 },
    { start: 12, end: 13 },
  ];
  const uttEmbs = [
    e(0.6, 0.3, 0.1), // 有把握 → 我
    e(0.55, 0.5, 0.1), // 领先 0.05 不足 → 随簇
    e(0.45, 0.1, 0.1), // 最高分不到 0.5 → 随簇
    e(0.1, 0.52, 0.3), // 0.52、领先 0.22 → 伴侣
    null, // 无 embedding → 随簇
  ];
  assert.deepEqual(labelUtterances(utts, uttEmbs, spans, spanEmbs, people), ['我', '孩子', '孩子', '伴侣', '孩子']);
});

test('labelUtterances：没归上簇或所归簇无 embedding 时用逐句结果', () => {
  const spans = [
    { start: 0, end: 5, speaker: 0 },
    { start: 5, end: 5.2, speaker: 1 },
  ];
  const spanEmbs = [e(0, 0, 1), null];
  const utts = [
    { start: 10, end: 12 }, // 无重叠 → 逐句伴侣 0.42
    { start: 5, end: 5.2 }, // 簇 1 无 embedding → 逐句我 0.3 不到阈值 → 其他
    { start: 20, end: 21 }, // 无簇、无 embedding → 其他
  ];
  const uttEmbs = [e(0.1, 0.42, 0.1), e(0.3, 0.1, 0.1), null];
  assert.deepEqual(labelUtterances(utts, uttEmbs, spans, spanEmbs, people), ['伴侣', OTHER, OTHER]);
});

test('unknownLabel：没认出来的人的标签可配置，渲染时同样不合并；非法值报错', async () => {
  const { setUnknownLabel, DEFAULT_UNKNOWN_LABEL } = await import('../src/label.mjs');
  const { renderTranscript } = await import('../src/render.mjs');
  try {
    setUnknownLabel('未知');
    assert.equal(decide(e(0.1, 0.2, 0.34), people).label, '未知');
    assert.equal(OTHER, '未知');
    assert.equal(renderTranscript([{ start: 0, end: 1, label: '未知', text: 'a' }, { start: 1.5, end: 2, label: '未知', text: 'b' }]), '[00:00] 未知 a\n[00:01] 未知 b\n');
    for (const bad of ['', '有 空格', '<x>', 3]) assert.throws(() => setUnknownLabel(bad), /unknownLabel/);
  } finally { setUnknownLabel(DEFAULT_UNKNOWN_LABEL); }
});
