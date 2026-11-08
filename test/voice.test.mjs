// 登记声纹：speakers.json 新旧格式、片段挑选、相似度检查、显示名、多个 known 人的认人、登记命令（合成数据）。
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { clipArgs, cutClips, remoteClipScript } from '../src/clips.mjs';
import { activityRegions, checkEnrollment, clusterReport, hourly, parseTime, pickCluster, pickRange, soloPieces } from '../src/enroll.mjs';
import { l2 } from '../src/embedding.mjs';
import { OTHER, decide, labelUtterances } from '../src/label.mjs';
import { renderTranscript } from '../src/render.mjs';
import { clipEntry, clipReferences, formatSpeakers, parseSpeakers, rawReferences } from '../src/speakers.mjs';
import { keepLocalSet } from '../src/storage.mjs';
import { recordingInfo } from '../src/recorder.mjs';
import { commands, parseVoiceArgs, resolveRecording, summarize } from '../src/voice.mjs';
import { testConfig } from './helpers.mjs';

// 正交轴：0 我、1 伴侣、2 孩子、3 同事A、4 同事B、5 陌生人。
const axis = (k, n = 6) => Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));
const mix = (pairs, n = 6) => l2(Array.from({ length: n }, (_, i) => pairs.reduce((a, [k, w]) => a + (k === i ? w : 0), 0)));
const PEOPLE = [
  { name: '我', label: '我', role: 'family', threshold: 0.35, vec: axis(0) },
  { name: '伴侣', label: '伴侣', role: 'family', threshold: 0.4, vec: axis(1) },
  { name: '孩子', label: '孩子', role: 'family', threshold: 0.35, vec: axis(2) },
  { name: '同事A', label: '同事A', role: 'known', threshold: 0.4, vec: axis(3) },
  { name: '同事B', label: '老B', role: 'known', threshold: 0.4, vec: axis(4) },
];

const OLD = [
  { name: '我', threshold: 0.35, enroll: [{ file: 'raw/2026-11-01/20261101_105551.MP3', start: 28.55, end: 32.098 }] },
  { name: '伴侣', threshold: 0.4, enroll: [{ file: 'raw/2026-11-03/20261103_140000.MP3', start: 1, end: 3 }] },
];

test('speakers.json：旧格式照常能读，缺省 role=family、display=名字；新格式 clip 校验', () => {
  const s = parseSpeakers(JSON.stringify(OLD));
  assert.deepEqual(s.map((x) => [x.role, x.display]), [['family', '我'], ['family', '伴侣']]);
  const known = [...OLD, { name: '同事A', role: 'known', threshold: 0.4, enroll: [clipEntry('raw/2026-11-03/20261103_081500.MP3', 10.5, 13.25)] }];
  const k = parseSpeakers(JSON.stringify(known));
  assert.equal(k[2].display, '同事A');
  assert.deepEqual(k[2].enroll[0], { clip: 'voiceprints/clips/20261103_081500_10500-13250.wav', source: 'raw/2026-11-03/20261103_081500.MP3', start: 10.5, end: 13.25 });
  assert.throws(() => parseSpeakers(JSON.stringify([{ ...OLD[0], role: 'friend' }])), /role/);
  assert.throws(() => parseSpeakers(JSON.stringify([{ ...OLD[0], display: '有 空格' }])), /display/);
  assert.throws(() => parseSpeakers(JSON.stringify([{ ...OLD[0], enroll: [{ clip: '../raw/x.wav' }] }])), /注册片段无效/);
  assert.throws(() => parseSpeakers(JSON.stringify([OLD[0], OLD[0]])), /重复/);
});

test('speakers.json：写出格式每个片段一行，读回内容不变；旧写法录音要留本机，clip 来源不用', async () => {
  const mixed = [OLD[0], { ...OLD[1], enroll: [clipEntry('raw/2026-11-03/20261103_140000.MP3', 1, 3)] }];
  const text = formatSpeakers(mixed);
  assert.equal(text.split('\n').filter((l) => l.startsWith('      {')).length, 2);
  assert.deepEqual(parseSpeakers(text), parseSpeakers(JSON.stringify(mixed)));
  assert.deepEqual(rawReferences(parseSpeakers(text)), ['raw/2026-11-01/20261101_105551.MP3']);
  assert.deepEqual(clipReferences(parseSpeakers(text)), ['voiceprints/clips/20261103_140000_1000-3000.wav']);
  const dir = await mkdtemp(join(tmpdir(), 'dairy-voice-keep-'));
  try {
    await writeFile(join(dir, 'speakers.json'), text);
    assert.deepEqual([...await keepLocalSet(dir, { keepLocal: ['raw/2026-11-01/e2e.MP3'] })], ['raw/2026-11-01/20261101_105551.MP3', 'raw/2026-11-01/e2e.MP3']);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('显示名：认人取所有登记的人里最像的，标签用 display；没到阈值仍是「其他」', () => {
  assert.equal(decide(mix([[4, 0.9], [1, 0.3]]), PEOPLE).label, '老B');
  assert.equal(decide(mix([[3, 0.9], [1, 0.4]]), PEOPLE).label, '同事A');
  const weak = decide(mix([[3, 0.35], [1, 0.3], [5, 0.88]]), PEOPLE);
  assert.equal(weak.label, OTHER);
  assert.equal(weak.name, '同事A');
  assert.match(renderTranscript([{ start: 0, end: 1, text: '好的', label: '老B' }]), /^\[00:00\] 老B 好的$/m);
});

test('多个 known 人：几位同事的簇各自认成自己，不再整簇认成伴侣', () => {
  // 三个簇：伴侣、同事A、同事B；声纹都带一点伴侣的成分（声音相近）。
  const spans = [{ start: 0, end: 10, speaker: 0 }, { start: 10, end: 20, speaker: 1 }, { start: 20, end: 30, speaker: 2 }];
  const spanEmbs = [mix([[1, 0.9], [3, 0.2]]), mix([[3, 0.8], [1, 0.5]]), mix([[4, 0.8], [1, 0.5]])];
  const utts = [{ start: 1, end: 3 }, { start: 11, end: 13 }, { start: 21, end: 23 }];
  const uttEmbs = [null, mix([[1, 0.5], [3, 0.45]]), mix([[1, 0.5], [4, 0.45]])];
  assert.deepEqual(labelUtterances(utts, uttEmbs, spans, spanEmbs, PEOPLE), ['伴侣', '同事A', '老B']);
  // 只登记家人时，同事的簇都会被认成伴侣——这正是要解决的问题。
  assert.deepEqual(labelUtterances(utts, uttEmbs, spans, spanEmbs, PEOPLE.slice(0, 3)), ['伴侣', '伴侣', '伴侣']);
});

test('时间与录音名：秒数、h:mm:ss；文件名推出日期与开始钟点', () => {
  assert.equal(parseTime('95.5'), 95.5);
  assert.equal(parseTime('1:35'), 95);
  assert.equal(parseTime('7:37:14'), 7 * 3600 + 37 * 60 + 14);
  assert.throws(() => parseTime('1h2m'), /h:mm:ss/);
  const cfg = testConfig();
  assert.deepEqual(recordingInfo('20261103_081500', cfg), { day: '2026-11-03', clock: 8 * 3600 + 15 * 60 });
  assert.deepEqual(resolveRecording('20261103_081500', cfg), { base: '20261103_081500', day: '2026-11-03', clock: 29700, source: 'raw/2026-11-03/20261103_081500.MP3' });
  assert.equal(resolveRecording('raw/2026-11-03/20261103_081500.m4a', cfg).source, 'raw/2026-11-03/20261103_081500.m4a');
  assert.equal(resolveRecording('REC-2026-11-03-081500', testConfig({ recorder: { pattern: 'REC-YYYY-MM-DD-HHmmss', extensions: ['.wav'] } })).source,
    'raw/2026-11-03/REC-2026-11-03-081500.wav');
  assert.throws(() => resolveRecording('foo', cfg), /recorder\.pattern/);
});

// 合成的一段：A 说 0–10 秒，B 说 9–20 秒（9–10 秒重叠），A 又说 30–40 秒。
const SPANS = [[0, 10, 'A'], [9, 20, 'B'], [30, 40, 'A']];
const piece = (start, end, speaker, text = '句子') => ({ start, end, text, speaker });
const PIECES = [
  piece(0.5, 3, 'A'), // 干净
  piece(8, 9.5, 'A'), // 与 B 重叠
  piece(11, 12, 'B'), // 太短（1 秒）
  piece(12, 16, 'B'), // 干净
  piece(17, 19.5, 'B'), // 干净
  piece(31, 33, 'A'), // 干净
  piece(25, 27, null), // 没归上任何人
  piece(34, 39.9, 'A'), // 干净，较长
];

test('activityRegions 与 soloPieces：只取单人、无重叠、≥1.5 秒的句子', () => {
  assert.deepEqual(activityRegions(SPANS).map((r) => [r[0], r[1], r[2].join('')]), [[0, 9, 'A'], [9, 10, 'AB'], [10, 20, 'B'], [30, 40, 'A']]);
  assert.deepEqual(soloPieces(PIECES, SPANS), [0, 3, 4, 5, 7]);
  assert.deepEqual(soloPieces(PIECES, SPANS, { maxSec: 5 }), [0, 3, 4, 5]);
});

test('pickRange：时间段里缺省只取单人时长最多的簇，可指定簇或全取', () => {
  const r = pickRange(PIECES, SPANS, 0, 21);
  assert.deepEqual(r.picked, [3, 4]);
  assert.equal(r.speaker, 'B');
  assert.deepEqual(r.bySpeaker.map((g) => [g.speaker, g.count]), [['B', 2], ['A', 1]]);
  assert.deepEqual(pickRange(PIECES, SPANS, 0, 21, { cluster: 'A' }).picked, [0]);
  assert.deepEqual(pickRange(PIECES, SPANS, 0, 21, { allSpeakers: true }).picked, [0, 3, 4]);
  assert.deepEqual(pickRange(PIECES, SPANS, 13, 21).picked, [4], '没完全落在时间段内的句子不取');
});

test('pickCluster：按与簇中心的相似度挑最干净的 N 个，时长 1.5–10 秒，按时间排序', () => {
  const embs = PIECES.map(() => null);
  embs[0] = mix([[3, 0.6], [5, 0.8]]);
  embs[5] = mix([[3, 0.95], [5, 0.3]]);
  embs[7] = mix([[3, 0.8], [5, 0.6]]);
  embs[1] = axis(3); // 重叠句，再像也不取
  const got = pickCluster(PIECES, embs, SPANS, 'A', axis(3), { max: 2 });
  assert.deepEqual(got.map((x) => x.i), [5, 7]);
  assert.ok(got[0].score > got[1].score);
  assert.deepEqual(pickCluster(PIECES, embs, SPANS, 'A', axis(3), { max: 5, maxSec: 5 }).map((x) => x.i), [0, 5]);
});

test('hourly：按名义钟点分小时，跨整点拆开', () => {
  assert.deepEqual(hourly([[0, 120], [3500, 3700]], 9 * 3600 - 60), [{ hour: '08–09', seconds: 60 }, { hour: '09–10', seconds: 220 }, { hour: '10–11', seconds: 40 }]);
  assert.deepEqual(hourly([[0, 10]], 23 * 3600 + 3595), [{ hour: '23–00', seconds: 5 }, { hour: '00–01', seconds: 5 }]);
});

test('clusterReport：每簇时长、句数、当前标签、最像的 3 人、小时分布、代表句；缺省不带文字', () => {
  const embs = PIECES.map((p) => (p.speaker === 'A' ? mix([[3, 0.9], [1, 0.4]]) : p.speaker === 'B' ? axis(1) : null));
  const data = { spans: SPANS, pieces: PIECES, pieceEmbs: embs, excl: [[0, 9, 'A'], [10, 20, 'B'], [30, 40, 'A']], exclEmbs: [mix([[3, 0.9], [1, 0.4]]), axis(1), mix([[3, 0.9], [1, 0.4]])] };
  const [a, b] = clusterReport(data, PEOPLE, { clock: 10 * 3600 });
  assert.deepEqual([a.speaker, a.seconds, a.soloSeconds, a.sentences, a.label], ['A', 20, 19, 4, '同事A']);
  assert.deepEqual(a.top.map((t) => t.name), ['同事A', '伴侣', '我']);
  assert.deepEqual(a.hours, [{ hour: '10–11', seconds: 20 }]);
  assert.deepEqual(a.lineLabels, { 同事A: 4 });
  assert.equal(b.label, '伴侣');
  assert.ok(a.samples.length >= 3 && a.samples.every((s) => /^\[\d\d:\d\d\]$/.test(s.at) && s.text === undefined));
  const withText = clusterReport(data, PEOPLE, { withText: true });
  assert.ok(withText[0].samples.every((s) => s.text === '句子'));
  const text = summarize('clusters', { day: '2026-11-03', voiceprints: '本机', note: '', files: [{ file: 'x', clusters: [a, b] }] });
  assert.ok(!text.includes('句子'));
});

test('checkEnrollment：找出更像别人的片段，片段太少、两人太像给出建议', () => {
  const people = [
    { name: '伴侣', clips: [{ id: 'w1', vec: mix([[1, 1], [5, 0.2]]) }, { id: 'w2', vec: mix([[1, 1], [2, 0.2]]) }, { id: 'w3', vec: mix([[1, 1], [0, 0.1]]) }, { id: 'bad', vec: mix([[3, 1], [1, 0.3]]) }] },
    { name: '同事A', clips: [{ id: 'c1', vec: mix([[3, 1], [5, 0.1]]) }, { id: 'c2', vec: mix([[3, 1], [2, 0.1]]) }] },
  ];
  const [w, c] = checkEnrollment(people);
  assert.deepEqual(w.suspicious.map((x) => [x.id, x.other.name]), [['bad', '同事A']]);
  assert.ok(w.advice.some((a) => a.includes('bad') && a.includes('更像同事A')));
  assert.ok(c.advice.some((a) => a.includes('只有 2 个片段')));
  assert.ok(w.pairMin < w.pairMean);
  const close = checkEnrollment([{ name: 'X', clips: [{ id: 'x', vec: mix([[1, 1], [3, 0.9]]) }] }, { name: 'Y', clips: [{ id: 'y', vec: mix([[3, 1], [1, 0.9]]) }] }]);
  assert.ok(close[0].advice.some((a) => a.includes('接近')));
});

test('parseVoiceArgs：参数个数、选项值、互斥选项', () => {
  assert.deepEqual(parseVoiceArgs(['add', '同事A', '20261103_081500', '1:00', '1:30', '--display', '同事', '--json']).opts, { clip: [], display: '同事', json: true });
  assert.deepEqual(parseVoiceArgs(['remove', '伴侣', '--clip', 'a', '--clip', 'b']).opts.clip, ['a', 'b']);
  assert.equal(parseVoiceArgs(['add', 'x', 'r', '1', '2', '--all-speakers']).opts.allSpeakers, true);
  assert.throws(() => parseVoiceArgs(['add', 'x']), /需要 4 个参数/);
  assert.throws(() => parseVoiceArgs(['list', '--role', 'x']), /role/);
  assert.throws(() => parseVoiceArgs(['add', 'x', 'r', '1', '2', '--whole', '--cluster', '3']), /只能选一个/);
  assert.throws(() => parseVoiceArgs(['nope']), /用法/);
  assert.throws(() => parseVoiceArgs(['add-cluster', 'x', 'd', 'r', '1', '--max', '2.5']), /整数/);
});

test('clip 裁剪参数与 engine 的 decode_range 一致，远程脚本引用安全、缺来源时以 4 退出', () => {
  assert.deepEqual(clipArgs('a.MP3', 28.55, 32.098, 'o.wav'),
    ['-nostdin', '-v', 'error', '-y', '-ss', '28.550', '-t', '3.548', '-i', 'a.MP3', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_f32le', '-f', 'wav', 'o.wav']);
  const remote = { workDir: '~/record-your-days' };
  const script = remoteClipScript(remote, [clipEntry('raw/2026-11-03/20261103_081500.MP3', 1, 3)]);
  assert.match(script, /test -f 'raw\/2026-11-03\/20261103_081500\.MP3' \|\| exit 4/);
  assert.match(script, /mv 'voiceprints\/clips\/20261103_081500_1000-3000\.wav\.part' 'voiceprints\/clips\/20261103_081500_1000-3000\.wav'/);
  assert.throws(() => remoteClipScript(remote, [{ source: 'raw/x;rm', clip: 'voiceprints/clips/a.wav', start: 0, end: 1 }]), /raw\/<日期>/);
});

const hasFfmpeg = spawnSync('ffmpeg', ['-version']).status === 0;

/** 临时项目：speakers.json、合成的 GPU 方案缓存、一段 60 秒的合成录音。 */
async function project() {
  const root = await mkdtemp(join(tmpdir(), 'dairy-voice-'));
  await writeFile(join(root, 'dairy.config.json'), JSON.stringify({ timeZone: 'Asia/Shanghai', modelDir: join(root, 'no-models') }));
  await writeFile(join(root, 'speakers.json'), JSON.stringify(OLD));
  await mkdir(join(root, 'cache/2026-11-03'), { recursive: true });
  const spans = [[0, 10, '0'], [9, 20, '1'], [30, 40, '0']];
  const embs = PIECES.map((p) => (p.speaker === 'A' ? mix([[3, 0.9], [1, 0.4]]) : p.speaker === 'B' ? axis(1) : null));
  const pieces = PIECES.map((p) => ({ ...p, speaker: p.speaker === null ? null : p.speaker === 'A' ? '0' : '1' }));
  await writeFile(join(root, 'cache/2026-11-03/gpu-20261103_081500.diar.json'), JSON.stringify({ spans, key: 'k' }));
  await writeFile(join(root, 'cache/2026-11-03/gpu-20261103_081500.embed.json'), JSON.stringify({
    pieces, piece_embs: embs, excl: [[0, 9, '0'], [10, 20, '1'], [30, 40, '0']], excl_embs: [embs[0], axis(1), embs[0]], key: 'k' }));
  if (hasFfmpeg) {
    await mkdir(join(root, 'raw/2026-11-03'), { recursive: true });
    spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=300:d=60', '-ac', '1', '-ar', '16000', join(root, 'raw/2026-11-03/20261103_081500.wav')]);
  }
  return root;
}

test('登记命令：按时间段、按簇登记成新的 known 人，list、rename、remove；clip 在本机裁出且时长对', { skip: !hasFfmpeg && '没有 ffmpeg' }, async () => {
  const root = await project();
  const config = testConfig({ modelDir: join(root, 'no-models'), numThreads: 1 });
  try {
    const rec = '20261103_081500.wav';
    const dry = await commands.add(root, config, ['同事A', rec, '0', '0:21'], { clip: [], dryRun: true });
    assert.equal(dry.added, 2);
    assert.equal(JSON.parse(await readFile(join(root, 'speakers.json'), 'utf8')).length, 2, '--dry-run 不改 speakers.json');
    const a = await commands.add(root, config, ['同事A', rec, '0', '0:21'], { clip: [], local: true });
    assert.deepEqual([a.added, a.cluster, a.newPerson, a.cutOn], [2, '1', true, '本机']);
    assert.match(a.voiceprints, /没有重算/);
    const st = await stat(join(root, 'voiceprints/clips/20261103_081500_12000-16000.wav'));
    assert.ok(Math.abs((st.size - 44) / 64000 - 4) < 0.01);
    const again = await commands.add(root, config, ['同事A', rec, '0', '0:21'], { clip: [], local: true });
    assert.deepEqual([again.added, again.skipped], [0, 2]);
    await assert.rejects(commands.add(root, config, ['同事B', rec, '0', '0:21'], { clip: [], local: true }), /已登记给 同事A/);
    const c = await commands['add-cluster'](root, config, ['同事B', '2026-11-03', rec, '0'], { clip: [], local: true, max: '2', display: '老C' });
    assert.equal(c.added, 2);
    await commands.rename(root, config, ['同事A', '同事C'], { clip: [] });
    const { people } = await commands.list(root);
    assert.deepEqual(people.map((p) => [p.name, p.role, p.display, p.clips]), [['我', 'family', '我', 1], ['伴侣', 'family', '伴侣', 1], ['同事C', 'known', '同事C', 2], ['同事B', 'known', '老C', 2]]);
    assert.deepEqual(people[2].days, ['2026-11-03']);
    const rm1 = await commands.remove(root, config, ['同事C'], { clip: ['20261103_081500_12000-16000'] });
    assert.deepEqual([rm1.removedClips, rm1.left, rm1.unreferencedClips], [1, 1, 1]);
    await assert.rejects(commands.remove(root, config, ['伴侣'], { clip: [] }), /--yes/);
    await commands.remove(root, config, ['同事B'], { clip: [] });
    assert.deepEqual(parseSpeakers(await readFile(join(root, 'speakers.json'), 'utf8')).map((s) => s.name), ['我', '伴侣', '同事C']);
    await assert.rejects(commands.add(root, config, ['x', rec, '21', '29'], { clip: [], local: true }), /没有干净的单人句/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('迁移：旧写法片段裁成 clip、speakers.json 改引用 clip 并补全 role/display，不删录音', { skip: !hasFfmpeg && '没有 ffmpeg' }, async () => {
  const root = await project();
  const config = testConfig({ modelDir: join(root, 'no-models'), numThreads: 1 });
  try {
    await writeFile(join(root, 'speakers.json'), JSON.stringify([{ name: '伴侣', threshold: 0.4, enroll: [{ file: 'raw/2026-11-03/20261103_081500.wav', start: 1, end: 3.5 }] }]));
    const out = await commands.migrate(root, config, [], { clip: [], local: true });
    assert.deepEqual([out.migrated, out.cutOn], [1, '本机']);
    const [w] = parseSpeakers(await readFile(join(root, 'speakers.json'), 'utf8'));
    assert.deepEqual(w.enroll, [{ clip: 'voiceprints/clips/20261103_081500_1000-3500.wav', source: 'raw/2026-11-03/20261103_081500.wav', start: 1, end: 3.5 }]);
    assert.deepEqual([w.role, w.display], ['family', '伴侣']);
    assert.ok(await stat(join(root, 'raw/2026-11-03/20261103_081500.wav')));
    assert.equal((await commands.migrate(root, config, [], { clip: [], local: true })).migrated, 0, '再跑一遍什么也不做');
    await assert.rejects(cutClips(root, [clipEntry('raw/2026-11-03/none.MP3', 0, 1)], { conn: null, remote: {} }), /本机没有/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
