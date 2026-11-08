// 项目根目录与 dairy.config.json。各字段的含义与缺省值见 dairy.config.example.json。
import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkTimeZone, systemTimeZone } from './dates.mjs';
import { DEFAULT_UNKNOWN_LABEL, checkLabel } from './label.mjs';
import { AUDIO_EXTS, DEFAULT_PATTERN, compilePattern } from './recorder.mjs';
import { homePath } from './shell.mjs';

export const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
export const CONFIG_FILE = 'dairy.config.json';
export const ENGINES = ['cpu', 'gpu'];
const STORAGES = ['local', 'remote'];
const CACHE = '~/.cache/record-your-days';
export const DEFAULTS = {
  modelDir: `${CACHE}/models/sherpa-onnx`,
  gpuEnvDir: `${CACHE}/engine-env`,
  gpuModelRoot: `${CACHE}/models`,
  remoteWorkDir: '~/record-your-days',
};

/** 本机路径的 ~ 展开为本机家目录。 */
export const expandHome = (p) => (p === '~' ? homedir() : p.startsWith('~/') ? join(homedir(), p.slice(2)) : p);

/** cpu | gpu；legacy 是 cpu 的旧名。 */
export function normalizeEngine(engine, what = '--engine') {
  const e = engine === 'legacy' ? 'cpu' : engine;
  if (!ENGINES.includes(e)) throw new Error(`${what} 只能是 cpu 或 gpu`);
  return e;
}

const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const HOST_RE = /^[A-Za-z0-9_.@[\]:%-]+$/;

/** gpu.remote：ssh 目标与远程目录。没配返回 null（GPU 方案在本机跑）。 */
function normalizeRemote(r) {
  if (r === undefined || r === null) return null;
  if (!isObj(r)) throw new Error('gpu.remote 应为对象');
  if (typeof r.host !== 'string' || !HOST_RE.test(r.host) || r.host.startsWith('-'))
    throw new Error('gpu.remote.host 应为 ssh 目标，例如 user@gpu-host 或 ~/.ssh/config 里的别名');
  if (r.port !== undefined && !(Number.isInteger(r.port) && r.port > 0 && r.port < 65536)) throw new Error('gpu.remote.port 应为 1–65535 的整数');
  if (r.identityFile !== undefined && (typeof r.identityFile !== 'string' || !r.identityFile || /[\s'"\\]/.test(r.identityFile)))
    throw new Error('gpu.remote.identityFile 应为不含空白和引号的路径');
  const workDir = r.workDir ?? DEFAULTS.remoteWorkDir;
  const out = { host: r.host, port: r.port ?? null, identityFile: r.identityFile ? expandHome(r.identityFile) : null,
    workDir, envDir: r.envDir ?? `${workDir}/engine-env`, modelRoot: r.modelRoot ?? DEFAULTS.gpuModelRoot };
  for (const k of ['workDir', 'envDir', 'modelRoot']) {
    try { homePath(out[k]); } catch (e) { throw new Error(`gpu.remote.${k}：${e.message}`); }
  }
  return out;
}

function normalizeRecorder(r = {}, timeZone) {
  if (!isObj(r)) throw new Error('recorder 应为对象');
  const pattern = r.pattern ?? DEFAULT_PATTERN;
  compilePattern(pattern);
  const extensions = r.extensions ?? ['.MP3'];
  if (!Array.isArray(extensions) || extensions.length === 0 || extensions.some((e) => typeof e !== 'string' || !AUDIO_EXTS.includes(e.toLowerCase())))
    throw new Error(`recorder.extensions 应为非空数组，元素取自 ${AUDIO_EXTS.join(' ')}（不分大小写）`);
  if (r.dir !== undefined && r.dir !== null && (typeof r.dir !== 'string' || !r.dir)) throw new Error('recorder.dir 应为录音笔上存录音的目录');
  if (r.deleteAfterTranscribe !== undefined && typeof r.deleteAfterTranscribe !== 'boolean') throw new Error('recorder.deleteAfterTranscribe 应为 true 或 false');
  return { dir: r.dir ? expandHome(r.dir) : null, pattern, extensions,
    timeZone: checkTimeZone(r.timeZone ?? timeZone, 'recorder.timeZone'), deleteAfterTranscribe: r.deleteAfterTranscribe ?? false };
}

/** 校验并补全缺省值。env：DAIRY_MODEL_DIR、DAIRY_NUM_THREADS 覆盖 modelDir、numThreads。 */
export function normalizeConfig(cfg, env = {}) {
  if (!isObj(cfg)) throw new Error(`${CONFIG_FILE} 应为 JSON 对象`);
  const timeZone = checkTimeZone(cfg.timeZone ?? systemTimeZone());
  const modelDir = env.DAIRY_MODEL_DIR || cfg.modelDir || DEFAULTS.modelDir;
  const numThreads = env.DAIRY_NUM_THREADS === undefined ? (cfg.numThreads ?? 4) : Number(env.DAIRY_NUM_THREADS);
  if (typeof modelDir !== 'string') throw new Error('modelDir 应为目录路径');
  if (!Number.isInteger(numThreads) || numThreads < 1 || numThreads > 36) throw new Error('numThreads 必须为 1–36 的整数');
  const keepLocal = cfg.keepLocal ?? [];
  if (!Array.isArray(keepLocal)) throw new Error('keepLocal 必须是数组');
  const storage = cfg.storage ?? 'local';
  if (!STORAGES.includes(storage)) throw new Error('storage 只能是 local 或 remote');
  const fallbackToCpu = cfg.fallbackToCpu ?? true;
  if (typeof fallbackToCpu !== 'boolean') throw new Error('fallbackToCpu 应为 true 或 false');
  const gpu = cfg.gpu ?? {};
  if (!isObj(gpu)) throw new Error('gpu 应为对象');
  const remote = normalizeRemote(gpu.remote);
  if (storage === 'remote' && !remote) throw new Error('storage 为 remote 时必须配 gpu.remote（录音存到 GPU 主机上）');
  return {
    timeZone,
    engine: normalizeEngine(cfg.engine ?? 'cpu', 'engine'),
    fallbackToCpu,
    unknownLabel: checkLabel(cfg.unknownLabel ?? DEFAULT_UNKNOWN_LABEL, 'unknownLabel '),
    modelDir: expandHome(modelDir),
    numThreads,
    storage,
    keepLocal,
    recorder: normalizeRecorder(cfg.recorder, timeZone),
    gpu: { ...gpu, envDir: expandHome(gpu.envDir ?? DEFAULTS.gpuEnvDir), modelRoot: expandHome(gpu.modelRoot ?? DEFAULTS.gpuModelRoot), remote },
  };
}

export async function loadConfig(root = ROOT, env = process.env) {
  let text;
  try { text = await readFile(join(root, CONFIG_FILE), 'utf8'); }
  catch (e) {
    if (e.code === 'ENOENT') throw new Error(`找不到 ${CONFIG_FILE}：先复制示例 cp dairy.config.example.json ${CONFIG_FILE}，再按需修改`);
    throw e;
  }
  let cfg;
  try { cfg = JSON.parse(text); } catch (e) { throw new Error(`${CONFIG_FILE} 不是合法的 JSON：${e.message}`); }
  return normalizeConfig(cfg, env);
}
