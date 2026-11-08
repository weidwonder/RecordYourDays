// 本机只重算认人（CPU 方案，或 GPU 方案不可用时）：不读录音，只用 cache/<日期>/ 里的缓存。
// 有 GPU 方案缓存（gpu-*.embed.json）且没指定 --engine cpu 时，用 engine/ 的 Python 代码重算（只用标准库，本机 python3 即可）；
// 否则按 CPU 方案缓存重算。GPU 方案的声纹优先用 GPU 方案算好的 cache/gpu-voiceprints-<hash>.json（与 GPU 方案结果一致），
// 没有就用本机 CAM++ 算的 cache/voiceprints-<hash>.json（同一模型，相似度有极小差异）。
import { spawn } from 'node:child_process';
import { readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { relabelDay } from './pipeline.mjs';
import { loadSpeakers } from './speakers.mjs';

const exists = (p) => stat(p).then(() => true, () => false);

export async function hasGpuCache(root, day) {
  return (await readdir(join(root, 'cache', day)).catch(() => [])).some((f) => /^gpu-.+\.embed\.json$/.test(f));
}

function runPython(args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn('python3', args, { cwd, env: { ...process.env, PYTHONPATH: join(cwd, 'engine') }, stdio: ['ignore', 'inherit', 'inherit'] });
    child.on('error', () => reject(new Error('本机没有 python3，不能重算 GPU 方案的认人')));
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`本机重算 GPU 方案认人失败（退出码 ${code}）`))));
  });
}

/**
 * engine：用户显式指定的方案（可为 undefined）。loadPeople：() => CPU 方案声纹（需要本机模型）。
 * 返回实际用的方案 'gpu' | 'cpu'。
 */
export async function relabelLocal(day, root, { engine, loadPeople, log = console.log }) {
  if (engine !== 'cpu' && (await hasGpuCache(root, day))) {
    const { hash } = await loadSpeakers(root);
    const args = ['-m', 'dairy_engine', 'transcribe', day, '--relabel-only', '--root', root];
    if (!(await exists(join(root, 'cache', `gpu-voiceprints-${hash}.json`)))) {
      await loadPeople();
      args.push('--voiceprints', join(root, 'cache', `voiceprints-${hash}.json`));
      log('本机没有 GPU 方案算的这份声纹，改用本机 CAM++ 算的声纹（相似度有极小差异）');
    }
    await runPython(args, root);
    return 'gpu';
  }
  if (engine === 'gpu') throw new Error(`cache/${day}/ 下没有 GPU 方案缓存，不能只重算认人`);
  await relabelDay(day, root, { people: await loadPeople(), log });
  return 'cpu';
}
