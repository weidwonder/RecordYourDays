// 本机 GPU 方案：在有 NVIDIA 显卡的 Linux 上直接用 engine/ 的 Python 环境转写，不经 ssh。
// 环境由 engine/setup_gpu.sh 建在 gpu.envDir，模型下载到 gpu.modelRoot（npm run setup-gpu）。
import { spawn } from 'node:child_process';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { resolveDay } from './dates.mjs';

const exists = (p) => stat(p).then(() => true, () => false);

/** 传给 Python 的环境变量：环境目录与模型目录以 dairy.config.json 经 Node 解析后的为准。 */
export const engineEnv = (root, config) => ({ ...process.env, PYTHONPATH: join(root, 'engine'),
  DAIRY_ENGINE_ENV_DIR: config.gpu.envDir, DAIRY_ENGINE_MODEL_ROOT: config.gpu.modelRoot });

function runInherit(bin, args, { cwd, env, spawnFn }) {
  return new Promise((resolve, reject) => {
    const child = spawnFn(bin, args, { cwd, env, stdio: ['ignore', 'inherit', 'inherit'] });
    child.on('error', () => reject(new Error(`${bin} 无法启动`)));
    child.on('close', (code) => (code === 0 ? resolve() : reject(Object.assign(new Error(`${bin} 退出码 ${code}`), { code }))));
  });
}

/** 本机 GPU 方案转写（或只重算认人）一天；退出码非 0 时抛错（带 code，3 为没有空闲 GPU）。 */
export async function transcribeGpuLocal(root, config, day, { force = false, relabelOnly = false, spawnFn = spawn } = {}) {
  resolveDay(day);
  const python = join(config.gpu.envDir, 'venv/bin/python');
  if (!(await exists(python))) throw new Error(`本机没有 GPU 方案的 Python 环境（${config.gpu.envDir}），先跑 npm run setup-gpu`);
  const args = ['-m', 'dairy_engine', 'transcribe', day, ...(force ? ['--force'] : []), ...(relabelOnly ? ['--relabel-only'] : []), '--root', root];
  await runInherit(python, args, { cwd: root, env: engineEnv(root, config), spawnFn });
}

/** 在本机建 GPU 方案的 Python 环境并下载缺的模型（engine/setup_gpu.sh，只支持 Linux）。 */
export async function setupGpuLocal(root, config, { spawnFn = spawn } = {}) {
  await runInherit('bash', [join(root, 'engine/setup_gpu.sh'), config.gpu.envDir], { cwd: root, env: engineEnv(root, config), spawnFn });
}
