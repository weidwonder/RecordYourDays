// dairy.config.json：缺省值、示例文件、校验与环境变量；录音笔文件名规则与时区换算。
import assert from 'node:assert/strict';
import { readFile, rm, writeFile, mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { loadConfig, normalizeConfig, ROOT } from '../src/config.mjs';
import { compilePattern, parseStamp, recorderFiles, recordingInfo } from '../src/recorder.mjs';
import { tmp } from './helpers.mjs';

test('缺省值：cpu 方案、本机存储、退回 CPU、「其他」、中性的模型目录；~ 展开为本机家目录', () => {
  const c = normalizeConfig({ timeZone: 'Asia/Shanghai' });
  assert.deepEqual([c.engine, c.storage, c.fallbackToCpu, c.unknownLabel, c.numThreads], ['cpu', 'local', true, '其他', 4]);
  assert.equal(c.modelDir, join(homedir(), '.cache/record-your-days/models/sherpa-onnx'));
  assert.equal(c.gpu.envDir, join(homedir(), '.cache/record-your-days/engine-env'));
  assert.equal(c.gpu.remote, null);
  assert.deepEqual(c.recorder, { dir: null, pattern: 'YYYYMMDD_HHmmss', extensions: ['.MP3'], timeZone: 'Asia/Shanghai', deleteAfterTranscribe: false });
  assert.ok(normalizeConfig({}).timeZone, '没配 timeZone 用本机时区');
});

test('示例配置能通过校验；gpu.remote 补全远程目录，远程路径保留 ~/ 交给远程展开', async () => {
  const ex = JSON.parse(await readFile(join(ROOT, 'dairy.config.example.json'), 'utf8'));
  assert.equal(normalizeConfig(ex).gpu.remote, null, '示例缺省用本机 GPU');
  ex.gpu.remote = { host: 'user@gpu-host', identityFile: '~/.ssh/id_ed25519' };
  const c = normalizeConfig(ex);
  assert.equal(c.gpu.remote.host, 'user@gpu-host');
  assert.equal(c.gpu.remote.identityFile, join(homedir(), '.ssh/id_ed25519'));
  assert.equal(c.gpu.remote.workDir, '~/record-your-days');
  assert.equal(c.gpu.minFreeMiB, 10000, 'GPU 方案参数原样留给 Python');
  const r = normalizeConfig({ gpu: { remote: { host: 'gpu' } } }).gpu.remote;
  assert.deepEqual(r, { host: 'gpu', port: null, identityFile: null, workDir: '~/record-your-days', envDir: '~/record-your-days/engine-env', modelRoot: '~/.cache/record-your-days/models' });
  JSON.parse(await readFile(join(ROOT, 'speakers.example.json'), 'utf8'));
});

test('校验：方案、存储、远程主机与路径、录音笔规则、标签', () => {
  const bad = [
    [{ engine: 'x' }, /engine/], [{ storage: 'cloud' }, /storage/], [{ storage: 'remote' }, /gpu\.remote/],
    [{ gpu: { remote: { host: '-oProxyCommand=x' } } }, /host/], [{ gpu: { remote: { host: 'a b' } } }, /host/],
    [{ gpu: { remote: { host: 'h', port: 0 } } }, /port/], [{ gpu: { remote: { host: 'h', workDir: 'rel' } } }, /workDir/],
    [{ gpu: { remote: { host: 'h', identityFile: 'a b' } } }, /identityFile/], [{ timeZone: 'Mars/Base' }, /timeZone/],
    [{ recorder: { pattern: 'YYYYMMDD' } }, /缺少占位符/], [{ recorder: { pattern: 'REC YYYYMMDDHHmmss' } }, /只能含/],
    [{ recorder: { extensions: ['.txt'] } }, /extensions/], [{ unknownLabel: '没 认出' }, /unknownLabel/], [{ numThreads: 0 }, /numThreads/],
    [{ fallbackToCpu: 'yes' }, /fallbackToCpu/],
  ];
  for (const [cfg, re] of bad) assert.throws(() => normalizeConfig(cfg), re, JSON.stringify(cfg));
  assert.equal(normalizeConfig({ engine: 'legacy' }).engine, 'cpu');
});

test('loadConfig：没有配置文件时提示复制示例；环境变量覆盖模型目录与线程', async () => {
  const dir = await tmp('ryd-config-');
  try {
    await assert.rejects(loadConfig(dir, {}), /dairy\.config\.example\.json/);
    await writeFile(join(dir, 'dairy.config.json'), '{ 坏');
    await assert.rejects(loadConfig(dir, {}), /不是合法的 JSON/);
    await writeFile(join(dir, 'dairy.config.json'), JSON.stringify({ modelDir: '/models' }));
    assert.equal((await loadConfig(dir, {})).modelDir, '/models');
    assert.equal((await loadConfig(dir, { DAIRY_NUM_THREADS: '8' })).numThreads, 8);
    assert.equal((await loadConfig(dir, { DAIRY_MODEL_DIR: '/other' })).modelDir, '/other');
    await assert.rejects(loadConfig(dir, { DAIRY_NUM_THREADS: '0' }), /numThreads/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('录音笔文件名规则：占位符各一次，解析出墙上时间；不是真实时刻的不认', () => {
  assert.equal(compilePattern('REC_YYYY-MM-DD_HH.mm.ss').re.source, '^REC_(\\d{4})\\-(\\d{2})\\-(\\d{2})_(\\d{2})\\.(\\d{2})\\.(\\d{2})$');
  assert.throws(() => compilePattern('YYYYMMDDHHmmssss'), /两次/);
  assert.deepEqual(parseStamp('20261103_081500'), { y: 2026, mo: 11, d: 3, h: 8, mi: 15, s: 0 });
  assert.deepEqual(parseStamp('REC_2026-11-03_08.02.53', 'REC_YYYY-MM-DD_HH.mm.ss'), { y: 2026, mo: 11, d: 3, h: 8, mi: 2, s: 53 });
  for (const x of ['20260230_080000', '20261103_250000', '20261103_0802', 'x20261103_081500']) assert.equal(parseStamp(x), null, x);
});

test('recordingInfo：文件名按 recorder.timeZone 解读，换算到 timeZone 决定归哪天与当天钟点', () => {
  const same = normalizeConfig({ timeZone: 'Asia/Shanghai' });
  assert.deepEqual(recordingInfo('20261103_081500', same), { day: '2026-11-03', clock: 8 * 3600 + 15 * 60 });
  const utc = normalizeConfig({ timeZone: 'Asia/Shanghai', recorder: { timeZone: 'UTC' } });
  assert.deepEqual(recordingInfo('20261102_200000', utc), { day: '2026-11-03', clock: 4 * 3600 });
  assert.equal(recordingInfo('foo', same), null);
});

test('recorderFiles：只列扩展名对、文件名合规且归这一天的录音；没挂载返回 null', async () => {
  const dir = await tmp('ryd-recorder-');
  try {
    for (const f of ['20261103_080000.MP3', '20261103_090000.mp3', '20261103_100000.WAV', '20261102_235959.MP3', 'note.MP3', '20261103_110000.MP3.part'])
      await writeFile(join(dir, f), 'x');
    await mkdir(join(dir, 'sub'));
    const config = normalizeConfig({ timeZone: 'Asia/Shanghai', recorder: { dir } });
    assert.deepEqual(await recorderFiles('2026-11-03', config), ['20261103_080000.MP3', '20261103_090000.mp3']);
    const wav = normalizeConfig({ timeZone: 'Asia/Shanghai', recorder: { dir, extensions: ['.MP3', '.wav'], timeZone: 'UTC' } });
    assert.deepEqual(await recorderFiles('2026-11-03', wav), ['20261102_235959.MP3', '20261103_080000.MP3', '20261103_090000.mp3', '20261103_100000.WAV']);
    assert.equal(await recorderFiles('2026-11-03', normalizeConfig({ recorder: { dir: join(dir, 'none') } })), null);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
