// 录音存放与删除：录音笔、本机、GPU 主机都用临时目录；GPU 主机用替身执行器（ssh 在临时 HOME 里执行，rsync 在本机拷贝）。
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdir, writeFile, readFile, rm, stat, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execute } from '../src/flow.mjs';
import { cleanRecorder, keepLocalSet, uploadRecorder } from '../src/storage.mjs';
import { connectRemote } from '../src/remote.mjs';
import { localExec, testConfig, tmp } from './helpers.mjs';

const DAY = '2026-11-03';
const exists = (p) => stat(p).then(() => true, () => false);
const mode = async (p) => ((await stat(p)).mode & 0o777).toString(8);

async function setup({ down = false, corruptPush = false, storage = 'remote', engine = 'gpu', deleteAfterTranscribe = true } = {}) {
  const dir = await tmp('ryd-storage-test-');
  const root = join(dir, 'root'); const home = join(dir, 'home'); const rec = join(dir, 'RECORD');
  const remoteHome = join(home, 'ryd');
  for (const d of [root, remoteHome, rec]) await mkdir(d, { recursive: true });
  const exec = localExec(home, { down, corruptPush });
  const config = testConfig({ storage, engine, recorder: { dir: rec, deleteAfterTranscribe }, gpu: { remote: { host: 'me@gpu-box', workDir: '~/ryd' } } });
  await writeFile(join(root, 'speakers.json'), JSON.stringify([{ name: '我', enroll: [{ file: 'raw/2026-11-01/enroll.MP3' }] }]));
  const put = async (base, rel, body) => { await mkdir(join(base, rel, '..'), { recursive: true }); await writeFile(join(base, rel), body); };
  return { dir, root, remoteHome, rec, config, exec, connect: (r) => connectRemote(r, exec), put, cleanup: () => rm(dir, { recursive: true, force: true }) };
}

test('storage remote，连上 GPU 主机：录音笔直传远程、补传本机暂存并删除、保留声纹与 keepLocal、冲突不动、核对后删原件', async () => {
  const t = await setup();
  try {
    t.config.keepLocal = ['raw/2026-11-02/keep.MP3'];
    await t.put(t.root, 'raw/2026-11-01/enroll.MP3', '声纹');
    await t.put(t.root, 'raw/2026-11-02/staged.MP3', '暂存');
    await t.put(t.root, 'raw/2026-11-02/keep.MP3', '保留');
    await t.put(t.root, 'raw/2026-10-30/conflict.MP3', '本机版');
    await t.put(t.remoteHome, 'raw/2026-10-30/conflict.MP3', '远程版');
    await t.put(t.rec, '20261103_080000.MP3', '上午');
    await t.put(t.rec, '20261103_090000.MP3', '下午');
    await t.put(t.rec, '20261102_070000.MP3', '别的日子');
    const logs = [];
    const out = await execute({ cmd: 'run', day: DAY, root: t.root, config: t.config, connect: t.connect, log: (s) => logs.push(s),
      runCpu: () => assert.fail('不能本机转写'),
      runRemote: async (root, remote, day, force, call, opts) => {
        assert.equal(typeof call, 'function');
        assert.equal(opts.pushAudio, false, '录音已在远程，不再推');
        assert.equal(await readFile(join(t.remoteHome, 'raw', DAY, '20261103_080000.MP3'), 'utf8'), '上午');
        assert.equal(await exists(join(root, 'raw', DAY)), false, '录音不落本机');
        await t.put(root, `transcripts/${DAY}/20261103_080000.md`, '稿');
      } });
    assert.equal(out.transcribed, 'remote');
    assert.equal(await mode(join(t.remoteHome, 'raw', DAY)), '700');
    assert.equal(await mode(join(t.remoteHome, 'raw', DAY, '20261103_090000.MP3')), '600');
    assert.equal(await readFile(join(t.remoteHome, 'raw/2026-11-02/staged.MP3'), 'utf8'), '暂存');
    assert.equal(await exists(join(t.root, 'raw/2026-11-02/staged.MP3')), false);
    assert.equal(await readFile(join(t.root, 'raw/2026-11-01/enroll.MP3'), 'utf8'), '声纹');
    assert.equal(await readFile(join(t.remoteHome, 'raw/2026-11-01/enroll.MP3'), 'utf8'), '声纹');
    assert.equal(await readFile(join(t.root, 'raw/2026-11-02/keep.MP3'), 'utf8'), '保留');
    assert.equal(await readFile(join(t.root, 'raw/2026-10-30/conflict.MP3'), 'utf8'), '本机版');
    assert.equal(await readFile(join(t.remoteHome, 'raw/2026-10-30/conflict.MP3'), 'utf8'), '远程版');
    assert.ok(logs.some((s) => s.includes('[冲突] raw/2026-10-30/conflict.MP3')));
    assert.deepEqual(out.recorder.removed, ['20261103_080000.MP3']);
    assert.deepEqual(out.recorder.kept, [{ file: '20261103_090000.MP3', reasons: [`transcripts/${DAY}/20261103_090000.md 还没生成`] }]);
    assert.deepEqual((await readdir(t.rec)).sort(), ['20261102_070000.MP3', '20261103_090000.MP3']);
    assert.ok(t.exec.calls.every((c) => ['ssh', 'rsync'].includes(c[0])), '只用 ssh 与 rsync');
  } finally { await t.cleanup(); }
});

test('storage remote，连不上：拷到本机、本机 CPU 转写、核对本机拷贝后删原件；--keep-recorder 不删；--no-fallback 报错；下次连上补传后删本机', async () => {
  const t = await setup({ down: true });
  try {
    await t.put(t.root, 'raw/2026-11-01/enroll.MP3', '声纹');
    await t.put(t.rec, '20261103_080000.MP3', '上午');
    let local = 0;
    const runCpu = async () => { local++; await t.put(t.root, `transcripts/${DAY}/20261103_080000.md`, '稿'); return 'cpu'; };
    const logs = [];
    const kept = await execute({ cmd: 'run', day: DAY, keepRecorder: true, root: t.root, config: t.config, connect: t.connect, runCpu, log: (s) => logs.push(s) });
    assert.deepEqual([kept.transcribed, kept.engine, kept.recorder], ['local', 'cpu', null]);
    assert.ok(logs.some((s) => s.includes('连不上 GPU 主机 me@gpu-box') && s.includes('CPU 方案')));
    assert.ok(await exists(join(t.rec, '20261103_080000.MP3')));
    const out = await execute({ cmd: 'run', day: DAY, root: t.root, config: t.config, connect: t.connect, runCpu, log: () => {} });
    assert.equal(local, 2);
    assert.deepEqual(out.recorder.removed, ['20261103_080000.MP3']);
    assert.equal(await readFile(join(t.root, 'raw', DAY, '20261103_080000.MP3'), 'utf8'), '上午');
    await assert.rejects(execute({ cmd: 'run', day: DAY, noFallback: true, root: t.root, config: t.config, connect: t.connect, runCpu, log: () => {} }), /连不上 GPU 主机.*不退回/);
    await assert.rejects(execute({ cmd: 'run', day: DAY, root: t.root, config: { ...t.config, fallbackToCpu: false }, connect: t.connect, runCpu, log: () => {} }), /不退回/);
    // GPU 主机恢复后，transcribe 先补传，转写结束再删本机这份。
    const up = await setup();
    try {
      const out2 = await execute({ cmd: 'transcribe', day: DAY, root: t.root, config: up.config, connect: up.connect, log: () => {},
        runCpu: () => assert.fail(), runRemote: async () => assert.ok(await exists(join(up.remoteHome, 'raw', DAY, '20261103_080000.MP3'))) });
      assert.equal(out2.transcribed, 'remote');
      assert.equal(await exists(join(t.root, 'raw', DAY, '20261103_080000.MP3')), false);
      assert.equal(await readFile(join(up.remoteHome, 'raw', DAY, '20261103_080000.MP3'), 'utf8'), '上午');
      assert.ok(await exists(join(t.root, 'raw/2026-11-01/enroll.MP3')));
    } finally { await up.cleanup(); }
  } finally { await t.cleanup(); }
});

test('storage remote，远程转写失败：临时取回当天录音本机转写，结束后删本机这份；--no-fallback 报错；CPU 方案也会临时取回', async () => {
  const t = await setup();
  try {
    await t.put(t.remoteHome, `raw/${DAY}/20261103_080000.MP3`, '上午');
    const logs = [];
    const runCpu = async () => { assert.equal(await readFile(join(t.root, 'raw', DAY, '20261103_080000.MP3'), 'utf8'), '上午'); return 'cpu'; };
    const out = await execute({ cmd: 'transcribe', day: DAY, root: t.root, config: t.config, connect: t.connect, log: (s) => logs.push(s),
      runRemote: async () => { throw new Error('不得显示的正文'); }, runCpu });
    assert.deepEqual([out.transcribed, out.engine], ['local', 'cpu']);
    assert.equal(await exists(join(t.root, 'raw', DAY, '20261103_080000.MP3')), false);
    assert.ok(logs.some((s) => s.includes('退回本机 CPU 方案')));
    assert.ok(logs.every((s) => !s.includes('不得显示的正文')));
    await assert.rejects(execute({ cmd: 'transcribe', day: DAY, noFallback: true, root: t.root, config: t.config, connect: t.connect, log: () => {},
      runRemote: async () => { throw new Error('x'); }, runCpu: () => assert.fail() }), /GPU 方案转写失败.*不退回/);
    const cpu = await execute({ cmd: 'transcribe', day: DAY, engine: 'cpu', root: t.root, config: t.config, connect: t.connect, log: () => {}, runRemote: () => assert.fail(), runCpu });
    assert.equal(cpu.transcribed, 'local');
    assert.equal(await exists(join(t.root, 'raw', DAY, '20261103_080000.MP3')), false);
  } finally { await t.cleanup(); }
});

test('storage local：录音拷到本机 raw/；远程 GPU 方案推当天录音；缺省不删录音笔原件，--delete-recorder 才删', async () => {
  const t = await setup({ storage: 'local', deleteAfterTranscribe: false });
  try {
    await t.put(t.rec, '20261103_080000.MP3', '上午');
    await t.put(t.rec, '20261102_235959.MP3', '前一天');
    const seen = [];
    const runRemote = async (root, remote, day, force, call, opts) => { seen.push(opts); await t.put(root, `transcripts/${DAY}/20261103_080000.md`, '稿'); };
    const logs = [];
    const out = await execute({ cmd: 'run', day: DAY, root: t.root, config: t.config, connect: t.connect, runRemote, runCpu: () => assert.fail(), log: (s) => logs.push(s) });
    assert.deepEqual([out.transcribed, out.recorder], ['remote', null]);
    assert.deepEqual(seen[0], { relabelOnly: false, pushAudio: true });
    assert.deepEqual(await readdir(join(t.root, 'raw', DAY)), ['20261103_080000.MP3']);
    assert.equal(await exists(join(t.remoteHome, 'raw', DAY)), false, 'storage local 不往远程存录音');
    assert.ok(logs.some((s) => s.includes('保留录音笔原件')));
    assert.ok(await exists(join(t.rec, '20261103_080000.MP3')));
    const del = await execute({ cmd: 'run', day: DAY, deleteRecorder: true, root: t.root, config: t.config, connect: t.connect, runRemote, runCpu: () => assert.fail(), log: () => {} });
    assert.deepEqual(del.recorder.removed, ['20261103_080000.MP3']);
    // CPU 方案不连 GPU 主机。
    const calls = t.exec.calls.length;
    await execute({ cmd: 'transcribe', day: DAY, engine: 'cpu', root: t.root, config: t.config, connect: t.connect, runCpu: async () => 'cpu', log: () => {} });
    assert.equal(t.exec.calls.length, calls);
    await assert.rejects(execute({ cmd: 'run', day: DAY, keepRecorder: true, deleteRecorder: true, root: t.root, config: t.config }), /不能同时/);
  } finally { await t.cleanup(); }
});

test('上传后校验不一致就报错，原件不删；录音笔未挂载跳过删除；没配录音笔目录报错；保留清单格式校验', async () => {
  const t = await setup({ corruptPush: true });
  try {
    await t.put(t.rec, '20261103_080000.MP3', '上午');
    const conn = await t.connect(t.config.gpu.remote);
    await assert.rejects(uploadRecorder(DAY, t.config, { remote: t.config.gpu.remote, call: conn.call, log: () => {} }), /校验不一致/);
    await t.put(t.root, `transcripts/${DAY}/20261103_080000.md`, '稿');
    const r = await cleanRecorder(DAY, t.root, t.config, { remote: t.config.gpu.remote, call: conn.call, log: () => {} });
    assert.deepEqual(r.removed, []);
    assert.ok(await exists(join(t.rec, '20261103_080000.MP3')));
    const logs = [];
    const none = testConfig({ recorder: { dir: join(t.dir, 'none') } });
    assert.equal(await cleanRecorder(DAY, t.root, none, { log: (s) => logs.push(s) }), null);
    assert.ok(logs[0].includes('未挂载'));
    await assert.rejects(cleanRecorder(DAY, t.root, testConfig(), { log: () => {} }), /recorder\.dir/);
    await assert.rejects(keepLocalSet(t.root, { keepLocal: ['../x'] }), /raw\/<日期>/);
  } finally { await t.cleanup(); }
});
