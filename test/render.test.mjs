import assert from 'node:assert/strict';
import { test } from 'node:test';
import { formatTimestamp, mergeAdjacent, renderTranscript } from '../src/render.mjs';

const u = (start, end, label, text) => ({ start, end, label, text });

test('formatTimestamp：不满一小时 mm:ss，满一小时 h:mm:ss，向下取整', () => {
  assert.equal(formatTimestamp(0), '00:00');
  assert.equal(formatTimestamp(28.55), '00:28');
  assert.equal(formatTimestamp(3599.9), '59:59');
  assert.equal(formatTimestamp(3600), '1:00:00');
  assert.equal(formatTimestamp(9549.2), '2:39:09');
});

test('mergeAdjacent：间隔 ≤ 2 s 的同标签句合并，> 2 s 不合并', () => {
  const m = mergeAdjacent([u(0, 1, '我', '甲。'), u(3, 4, '我', '乙。'), u(6.5, 7, '我', '丙。')]);
  assert.deepEqual(m, [u(0, 4, '我', '甲。乙。'), u(6.5, 7, '我', '丙。')]);
});

test('mergeAdjacent：「其他」不合并，不同标签不合并，按开始时间排序且不改输入', () => {
  const input = [u(2, 3, '其他', 'b'), u(0, 1, '其他', 'a'), u(3.5, 4, '伴侣', 'c'), u(4.5, 5, '孩子', 'd')];
  const m = mergeAdjacent(input);
  assert.deepEqual(m.map((x) => x.text), ['a', 'b', 'c', 'd']);
  assert.equal(input[0].text, 'b');
});

test('renderTranscript：行格式、合并与超过 30 s 静音标记', () => {
  const md = renderTranscript([
    u(28.55, 32.1, '我', '今天先到这里。'),
    u(33.7, 36.4, '我', '好。'),
    u(36.8, 37.2, '其他', '是。'),
    u(37.5, 38, '其他', '嗯。'),
    u(67.5, 70, '孩子', '我的。'),
    u(3700, 3701, '伴侣', '好好。'),
  ]);
  assert.equal(
    md,
    [
      '[00:28] 我 今天先到这里。好。',
      '[00:36] 其他 是。',
      '[00:37] 其他 嗯。',
      '[01:07] 孩子 我的。',
      '[01:10] <no speech detected in this segment>',
      '[1:01:40] 伴侣 好好。',
    ].join('\n') + '\n'
  );
});

test('renderTranscript：开头静音超过 30 s 从 00:00 标起；恰好 30 s 不标', () => {
  assert.equal(renderTranscript([u(31, 32, '我', '嗯。')]), '[00:00] <no speech detected in this segment>\n[00:31] 我 嗯。\n');
  assert.equal(renderTranscript([u(30, 32, '我', '嗯。')]), '[00:30] 我 嗯。\n');
});

test('renderTranscript：没有句子时输出 <no speech detected>', () => {
  assert.equal(renderTranscript([]), '<no speech detected>\n');
});

test('renderTranscript：与 GPU 方案 engine/dairy_engine/render.py 共用的样例逐字一致', async () => {
  const { readFile } = await import('node:fs/promises');
  const cases = JSON.parse(await readFile(new URL('./fixtures/render-cases.json', import.meta.url), 'utf8'));
  for (const c of cases) assert.equal(renderTranscript(c.utts.map((x) => ({ ...x }))), c.expected, c.name);
});
