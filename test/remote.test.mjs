// GPU 主机：ssh / rsync 参数、远程命令、部署与远程转写的文件清单。不连网络：执行器可注入。
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { quote, homePath, remoteCommand, remoteCall, connectRemote, deployRemote, pollCommand, run, transcribeRemote, waitRemote, NO_GPU_EXIT } from '../src/remote.mjs';
import { configureInference, inferenceOptions } from '../src/inference.mjs';
import { testConfig, tmp } from './helpers.mjs';

const remote = testConfig({ gpu: { remote: { host: 'me@gpu-box', port: 2222, identityFile: '/keys/id', workDir: '~/ryd' } } }).gpu.remote;
const DAY = '2026-11-03';

test('remoteCommand：跑远程 envDir 的 Python，环境目录与模型目录用环境变量传入，flock 防并发，路径拒绝注入', () => {
  const cmd = remoteCommand(remote, DAY, true, 'abc-1');
  for (const part of ['PYTHONPATH=engine DAIRY_ENGINE_ENV_DIR=', 'ryd/engine-env', ' DAIRY_ENGINE_MODEL_ROOT=', '.cache/record-your-days/models',
    '/venv/bin/python -m dairy_engine transcribe ', '2026-11-03', ' --force > logs/transcribe.log 2>&1', 'flock -n .dairy.lock', 'chmod 700', 'logs/run-abc-1.exit'])
    assert.ok(cmd.includes(part), part);
  assert.match(cmd, /setsid nohup sh -c .* > \/dev\/null 2>&1 < \/dev\/null &$/);
  assert.ok(remoteCommand(remote, DAY, false, 'x', true).includes(' --relabel-only > logs/transcribe.log'));
  assert.throws(() => remoteCommand(remote, DAY, true, 'x', true), /不能同时/);
  assert.throws(() => remoteCommand(remote, DAY, false, 'a;b'), /runId/);
  assert.throws(() => remoteCommand(remote, '../bad', false));
  assert.equal(homePath('~/models/x'), '"$HOME"/\'models/x\'');
  assert.equal(quote("a'b"), "'a'\\''b'");
  for (const value of ['~/../x', '/x/./y', '~/x;touch /tmp/x', 'x', '/', '~/']) assert.throws(() => homePath(value));
  assert.ok(pollCommand(remote, 'abc-1').includes('rm -f logs/run-abc-1.exit'));
});

test('remoteCall：ssh 带端口、密钥、BatchMode，脚本交给远程 sh；rsync 用同一套 ssh 参数，~/ 写成相对家目录的路径', async () => {
  const calls = [];
  const call = remoteCall(remote, async (bin, args, opts) => { calls.push({ bin, args, opts }); return 'ok'; });
  await call(['ssh', "echo 'hi'"], { capture: true });
  await call(['push', '/local/engine/', '~/ryd/engine/']);
  await call(['pull', '/abs/cache/x/', '/local/cache/']);
  const [s, push, pull] = calls;
  assert.equal(s.bin, 'ssh');
  assert.deepEqual(s.args.slice(-2), ['me@gpu-box', `sh -c ${quote("echo 'hi'")}`]);
  for (const x of ['BatchMode=yes', '-p', '2222', '-i', '/keys/id']) assert.ok(s.args.includes(x), x);
  assert.equal(s.opts.capture, true);
  assert.deepEqual(push.args, ['-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 2222 -i /keys/id -o IdentitiesOnly=yes', '--', '/local/engine/', 'me@gpu-box:ryd/engine/']);
  assert.deepEqual(pull.args.slice(-2), ['me@gpu-box:/abs/cache/x/', '/local/cache/']);
  assert.throws(() => call(['bx', 'host']), /未知/);
  const plain = remoteCall(testConfig({ gpu: { remote: { host: 'gpu' } } }).gpu.remote, async (bin, args) => args);
  assert.ok(!(await plain(['ssh', 'true'])).includes('-p'), '没配端口与密钥就不带');
  const conn = await connectRemote(remote, async (bin, args) => { calls.push({ bin, args }); return ''; });
  assert.equal(conn.host, 'me@gpu-box');
  assert.equal(calls.at(-1).args.at(-1), `sh -c 'true'`);
  await assert.rejects(connectRemote(remote, async () => { throw new Error('ssh 失败'); }), /ssh 失败/);
});

test('run：能捕获输出，失败时只报退出码，不带子进程的 stdout/stderr', async () => {
  const dir = await tmp('ryd-run-');
  try {
    const bin = join(dir, 'fake');
    await writeFile(bin, '#!/bin/sh\nif [ "$1" = ok ]; then echo 结果; else echo 隐私; echo 隐私 >&2; exit 9; fi\n', { mode: 0o700 });
    assert.equal(await run(bin, ['ok'], { capture: true }), '结果');
    await assert.rejects(run(bin, ['x']), (e) => e.message.includes('退出码 9') && e.code === 9 && !e.message.includes('隐私'));
    await assert.rejects(run(join(dir, 'none'), []), /无法启动/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('轮询：容忍断线，非 0 退出码带 code 抛出，连续失败才放弃', async () => {
  const replies = [new Error('断线'), 'running', new Error('断线'), '0'];
  await waitRemote(async () => { const r = replies.shift(); if (r instanceof Error) throw r; return r; }, remote, 'abc-1', { pollMs: 0, maxFailures: 1 });
  assert.equal(replies.length, 0);
  await assert.rejects(waitRemote(async () => '3', remote, 'abc-1', { pollMs: 0 }), (e) => e.code === NO_GPU_EXIT);
  await assert.rejects(waitRemote(async () => { throw new Error('x'); }, remote, 'abc-1', { pollMs: 0, maxFailures: 2 }), /连续失败/);
  await assert.rejects(waitRemote(async () => '乱码', remote, 'abc-1', { pollMs: 0 }), /无法识别/);
});

test('deployRemote：只推 engine/ 与 dairy.config.json，在 flock 里跑 setup_gpu.sh 并传入远程的环境目录与模型目录', async () => {
  const calls = [];
  await deployRemote('/proj', remote, async (args) => calls.push(args));
  assert.deepEqual(calls.filter((a) => a[0] === 'push').map((a) => [a[1], a[2]]), [['/proj/engine/', '~/ryd/engine/'], ['/proj/dairy.config.json', '~/ryd/dairy.config.json']]);
  const setup = calls.find((a) => a[0] === 'ssh' && a[1].includes('setup_gpu.sh'));
  assert.match(setup[1], /flock -n \.dairy\.lock env DAIRY_ENGINE_ENV_DIR=.* DAIRY_ENGINE_MODEL_ROOT=.* bash engine\/setup_gpu\.sh "\$HOME"\/'ryd\/engine-env' > logs\/engine-install\.log 2>&1/);
});

async function project() {
  const dir = await tmp('ryd-remote-test-');
  await mkdir(join(dir, 'raw/2026-11-01'), { recursive: true });
  await mkdir(join(dir, `raw/${DAY}`), { recursive: true });
  await writeFile(join(dir, `raw/${DAY}/20261103_080000.MP3`), '当天录音');
  await mkdir(join(dir, `transcripts/${DAY}`), { recursive: true });
  await writeFile(join(dir, `transcripts/${DAY}/test.md`), '已有结果');
  await writeFile(join(dir, 'raw/2026-11-01/enroll.MP3'), '假的音频');
  await mkdir(join(dir, 'voiceprints/clips'), { recursive: true });
  await writeFile(join(dir, 'voiceprints/clips/20261103_140000_1000-3000.wav'), '假的 clip');
  await writeFile(join(dir, 'speakers.json'), JSON.stringify([{ enroll: [{ file: 'raw/2026-11-01/enroll.MP3' },
    { clip: 'voiceprints/clips/20261103_140000_1000-3000.wav', source: 'raw/2026-11-03/20261103_140000.MP3', start: 1, end: 3 }] }]));
  return dir;
}

test('transcribeRemote：只推程序、配置与声纹引用；storage 为 local 时推当天录音、结束后删远程这份；拉取全部成功才更新本地', async () => {
  const dir = await project();
  try {
    const calls = [];
    const fake = async (args) => {
      calls.push(args);
      if (args[0] === 'pull') await writeFile(join(args[2], 'new.json'), '假的产物');
      if (args[0] === 'ssh' && args[1].includes('.exit; then')) return '0';
    };
    await transcribeRemote(dir, remote, DAY, true, fake, { pollMs: 0 });
    const pushed = calls.filter((a) => a[0] === 'push').map((a) => a[1].slice(dir.length + 1));
    assert.deepEqual(pushed, ['engine/', 'dairy.config.json', 'speakers.json', 'voiceprints/clips/', 'raw/2026-11-01/enroll.MP3']);
    assert.equal(await readFile(join(dir, `cache/${DAY}/new.json`), 'utf8'), '假的产物');
    calls.length = 0;
    await transcribeRemote(dir, remote, DAY, false, fake, { pollMs: 0, pushAudio: true });
    assert.ok(calls.some((a) => a[0] === 'push' && a[1] === join(dir, 'raw', DAY) + '/' && a[2] === `~/ryd/raw/${DAY}/`), '推当天录音');
    const runAt = calls.findIndex((a) => a[0] === 'ssh' && a[1].includes('setsid nohup'));
    const rmAt = calls.findIndex((a) => a[0] === 'ssh' && a[1] === `rm -rf "$HOME"/'ryd/raw/${DAY}'`);
    assert.ok(runAt > 0 && rmAt > runAt, '转写结束后删远程的录音');
    calls.length = 0;
    await transcribeRemote(dir, remote, DAY, false, fake, { pollMs: 0, pushAudio: true, relabelOnly: true });
    assert.ok(!calls.some((a) => a[0] === 'push' && a[1].includes(`raw/${DAY}`)), '只重算认人不推录音');
    // 远程失败也删远程的录音；拉取失败时本地结果不动。
    calls.length = 0;
    await assert.rejects(transcribeRemote(dir, remote, DAY, false, async (args) => { calls.push(args); if (args[0] === 'ssh' && args[1].includes('.exit; then')) return '1'; }, { pollMs: 0, pushAudio: true }), /退出码 1/);
    assert.ok(calls.some((a) => a[0] === 'ssh' && a[1].startsWith('rm -rf')));
    await assert.rejects(transcribeRemote(dir, remote, DAY, false, async (args) => {
      if (args[0] === 'ssh' && args[1].includes('.exit; then')) return '0';
      if (args[0] === 'pull') {
        if (args[1].includes('/cache/')) throw new Error('拉取失败');
        await writeFile(join(args[2], 'test.md'), '不能覆盖');
      }
    }, { pollMs: 0 }));
    assert.equal(await readFile(join(dir, `transcripts/${DAY}/test.md`), 'utf8'), '已有结果');
    await writeFile(join(dir, 'speakers.json'), JSON.stringify([{ enroll: [{ file: '../secret' }] }]));
    await assert.rejects(transcribeRemote(dir, remote, DAY, false, () => assert.fail()), /声纹引用/);
    await writeFile(join(dir, 'speakers.json'), JSON.stringify([{ enroll: [{ clip: 'voiceprints/clips/../../x.wav' }] }]));
    await assert.rejects(transcribeRemote(dir, remote, DAY, false, () => assert.fail()), /声纹 clip/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('线程边界校验', () => {
  assert.throws(() => configureInference(37));
  configureInference(2); assert.equal(inferenceOptions().numThreads, 2); configureInference(4);
});
