// 转写方案路由：--engine / 配置缺省、本机 GPU 与远程 GPU、连不上与失败时退回本机 CPU 方案。Python 侧单测也从这里跑。
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parseArgs } from '../src/cli.mjs';
import { execute, resolveEngine } from '../src/flow.mjs';
import { transcribeGpuLocal } from '../src/gpu.mjs';
import { NO_GPU_EXIT } from '../src/remote.mjs';
import { testConfig, tmp } from './helpers.mjs';

const DAY = '2026-11-03';
const REMOTE = { host: 'me@gpu-box', workDir: '~/ryd' };

test('parseArgs：--engine 两种写法与 legacy 别名，开关组合，models / setup-gpu 不带参数', () => {
  assert.equal(parseArgs(['transcribe', DAY, '--engine', 'legacy']).engine, 'cpu');
  assert.equal(parseArgs(['run', '--engine=gpu', '--force', '--no-fallback']).engine, 'gpu');
  assert.equal(parseArgs(['run', '--no-fallback']).noFallback, true);
  assert.equal(parseArgs(['transcribe']).engine, undefined);
  assert.equal(parseArgs(['run', '--delete-recorder']).deleteRecorder, true);
  assert.equal(parseArgs(['models']).cmd, 'models');
  assert.equal(parseArgs(['setup-gpu']).cmd, 'setup-gpu');
  for (const bad of [['transcribe', '--engine', 'x'], ['transcribe', '--engine'], ['import', '--engine', 'gpu'], ['import', '--no-fallback'],
    ['transcribe', '--keep-recorder'], ['run', '--keep-recorder', '--delete-recorder'], ['models', DAY], ['setup-gpu', '--force'], ['nope'],
    ['run', DAY, '--relabel-only'], ['transcribe', DAY, '--relabel-only', '--force'], ['transcribe', '--local']])
    assert.throws(() => parseArgs(bad), /用法/, bad.join(' '));
  assert.equal(parseArgs(['transcribe', DAY, '--relabel-only']).relabelOnly, true);
});

test('resolveEngine：缺省 cpu，配置可改缺省，--engine 优先，legacy 是 cpu 的旧名', () => {
  assert.equal(resolveEngine({ config: testConfig() }), 'cpu');
  assert.equal(resolveEngine({ config: testConfig({ engine: 'gpu' }) }), 'gpu');
  assert.equal(resolveEngine({ config: testConfig({ engine: 'legacy' }) }), 'cpu');
  assert.equal(resolveEngine({ engine: 'cpu', config: testConfig({ engine: 'gpu' }) }), 'cpu');
  assert.throws(() => resolveEngine({ engine: 'x', config: testConfig() }), /cpu 或 gpu/);
});

async function fixture() {
  const root = await tmp('ryd-engine-route-');
  await writeFile(join(root, 'speakers.json'), '[]');
  await mkdir(join(root, 'raw', DAY), { recursive: true });
  await writeFile(join(root, 'raw', DAY, '20261103_080000.MP3'), '音频');
  return root;
}

const up = async (remote) => ({ host: remote.host, call: async () => '' });
const down = async () => { throw new Error('连不上'); };

test('路由：本机 GPU（没配 gpu.remote）直接跑 runGpu；失败退回 CPU 并提示，没有空闲 GPU 单独说明；不许退回时报错', async () => {
  const root = await fixture();
  try {
    const config = testConfig({ engine: 'gpu' });
    const seen = [];
    const logs = [];
    const base = { cmd: 'transcribe', day: DAY, root, config, log: (s) => logs.push(s), connect: () => assert.fail('不连远程'),
      runCpu: async () => { seen.push('cpu'); return 'cpu'; }, runGpu: async (o) => seen.push(['gpu', o]), runRemote: () => assert.fail() };
    assert.deepEqual(await execute(base), { engine: 'gpu', transcribed: 'local', recorder: null });
    assert.deepEqual(seen.pop(), ['gpu', { relabelOnly: false }]);
    const noGpu = Object.assign(new Error('隐私正文'), { code: NO_GPU_EXIT });
    const fell = await execute({ ...base, runGpu: async () => { throw noGpu; } });
    assert.deepEqual([fell.transcribed, fell.engine, seen.pop()], ['local', 'cpu', 'cpu']);
    assert.ok(logs.some((s) => s.includes('本机没有空闲 GPU') && s.includes('退回本机 CPU 方案')));
    assert.ok(logs.every((s) => !s.includes('隐私正文')));
    await assert.rejects(execute({ ...base, noFallback: true, runGpu: async () => { throw noGpu; } }), /本机没有空闲 GPU.*不退回/);
    await assert.rejects(execute({ ...base, config: { ...config, fallbackToCpu: false }, runGpu: async () => { throw new Error('x'); } }), /不退回/);
    assert.equal((await execute({ ...base, engine: 'cpu' })).engine, 'cpu');
    assert.equal(seen.pop(), 'cpu');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('路由：远程 GPU 连得上就在远程跑；连不上、远程失败都退回本机 CPU；--no-fallback 不退回', async () => {
  const root = await fixture();
  try {
    const config = testConfig({ engine: 'gpu', gpu: { remote: REMOTE } });
    const seen = [];
    const logs = [];
    const base = { cmd: 'transcribe', day: DAY, root, config, connect: up, log: (s) => logs.push(s), runGpu: () => assert.fail('有 gpu.remote 时不在本机跑 GPU'),
      runCpu: async () => { seen.push('cpu'); return 'cpu'; }, runRemote: async (...a) => seen.push(['remote', a[5]]) };
    assert.deepEqual(await execute(base), { engine: 'gpu', transcribed: 'remote', recorder: null });
    assert.deepEqual(seen.pop(), ['remote', { relabelOnly: false, pushAudio: true }]);
    const off = await execute({ ...base, connect: down, runRemote: () => assert.fail('连不上不能跑远程') });
    assert.deepEqual([off.transcribed, off.engine, seen.pop()], ['local', 'cpu', 'cpu']);
    assert.ok(logs.some((s) => s.includes('连不上 GPU 主机 me@gpu-box')));
    const noGpu = Object.assign(new Error('隐私正文'), { code: NO_GPU_EXIT });
    const failed = await execute({ ...base, runRemote: async () => { throw noGpu; } });
    assert.deepEqual([failed.transcribed, seen.pop()], ['local', 'cpu']);
    assert.ok(logs.some((s) => s.includes('GPU 主机 me@gpu-box 上没有空闲 GPU') && s.includes('退回本机 CPU 方案')));
    await assert.rejects(execute({ ...base, noFallback: true, runRemote: async () => { throw noGpu; } }), /没有空闲 GPU.*不退回/);
    await assert.rejects(execute({ ...base, noFallback: true, connect: down }), /连不上 GPU 主机/);
    await rm(join(root, 'raw', DAY), { recursive: true });
    await assert.rejects(execute({ ...base, connect: down }), /没有音频/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('路由：只重算认人时远程传 relabelOnly、不推录音；本机不要求有录音，用 runCpu 返回的方案；远程或本机 GPU 失败改在本机重算', async () => {
  const root = await fixture();
  try {
    const seen = [];
    const logs = [];
    const remoteConfig = testConfig({ engine: 'gpu', gpu: { remote: REMOTE } });
    const base = { cmd: 'transcribe', day: DAY, root, relabelOnly: true, config: remoteConfig, connect: up, log: (s) => logs.push(s),
      runCpu: async () => { seen.push('local'); return 'gpu'; }, runRemote: async (...a) => seen.push(a[5]), runGpu: async (o) => seen.push(['gpu', o]) };
    assert.deepEqual(await execute(base), { engine: 'gpu', transcribed: 'remote', recorder: null });
    assert.deepEqual(seen.pop(), { relabelOnly: true, pushAudio: true });
    const fell = await execute({ ...base, runRemote: async () => { throw new Error('x'); } });
    assert.deepEqual([fell.transcribed, fell.engine, seen.pop()], ['local', 'gpu', 'local']);
    assert.ok(logs.some((s) => s.includes('改在本机按缓存重算认人')));
    const localGpu = await execute({ ...base, config: testConfig({ engine: 'gpu' }) });
    assert.deepEqual([localGpu.transcribed, seen.pop()], ['local', ['gpu', { relabelOnly: true }]]);
    await rm(join(root, 'raw', DAY), { recursive: true });
    const cpu = await execute({ ...base, engine: 'cpu' });
    assert.deepEqual([cpu.transcribed, cpu.engine, seen.pop()], ['local', 'gpu', 'local']);
    await assert.rejects(execute({ ...base, force: true }), /relabel-only/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('transcribeGpuLocal：用 gpu.envDir 的 Python 跑 dairy_engine，传 PYTHONPATH 与环境、模型目录；没有环境时提示 setup-gpu；退出码带出', async () => {
  const root = await tmp('ryd-gpu-local-');
  try {
    const config = testConfig({ gpu: { envDir: join(root, 'env'), modelRoot: join(root, 'models') } });
    await assert.rejects(transcribeGpuLocal(root, config, DAY), /npm run setup-gpu/);
    await mkdir(join(root, 'env/venv/bin'), { recursive: true });
    await writeFile(join(root, 'env/venv/bin/python'), '');
    const calls = [];
    const fakeSpawn = (code) => (bin, args, opts) => {
      calls.push({ bin, args, opts });
      const child = new EventEmitter();
      setImmediate(() => child.emit('close', code));
      return child;
    };
    await transcribeGpuLocal(root, config, DAY, { force: true, spawnFn: fakeSpawn(0) });
    const [c] = calls;
    assert.equal(c.bin, join(root, 'env/venv/bin/python'));
    assert.deepEqual(c.args, ['-m', 'dairy_engine', 'transcribe', DAY, '--force', '--root', root]);
    assert.equal(c.opts.env.PYTHONPATH, join(root, 'engine'));
    assert.equal(c.opts.env.DAIRY_ENGINE_MODEL_ROOT, join(root, 'models'));
    assert.equal(c.opts.env.DAIRY_ENGINE_ENV_DIR, join(root, 'env'));
    await transcribeGpuLocal(root, config, DAY, { relabelOnly: true, spawnFn: fakeSpawn(0) });
    assert.ok(calls[1].args.includes('--relabel-only'));
    await assert.rejects(transcribeGpuLocal(root, config, DAY, { spawnFn: fakeSpawn(3) }), (e) => e.code === NO_GPU_EXIT);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('Python 侧单测（engine/tests，标准库 unittest）', (t) => {
  const py = spawnSync('python3', ['--version']);
  if (py.status !== 0) { t.skip('没有 python3'); return; }
  const cwd = fileURLToPath(new URL('../engine/', import.meta.url));
  const r = spawnSync('python3', ['-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], { cwd, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
});
