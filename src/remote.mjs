// GPU 主机（dairy.config.json 的 gpu.remote）：用普通的 ssh 执行命令、rsync 传文件。
// 远程只收程序必需文件、配置与声纹引用；storage 为 remote 时录音由 storage.mjs 直接存到 GPU 主机。
// 子进程输出不转发，避免错误回显隐私；远程转写的日志只留在远程 logs/ 下。
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdir, realpath, mkdtemp, cp, rm } from 'node:fs/promises';
import { join, resolve, sep } from 'node:path';
import { pushClips } from './clips.mjs';
import { resolveDay } from './dates.mjs';
import { homePath, quote } from './shell.mjs';
import { CLIP_RE, RAW_RE, readSpeakersText, speakersHash } from './speakers.mjs';

export { homePath, quote } from './shell.mjs';

/** 跑一个本机程序；capture 时返回 stdout（去掉首尾空白），失败时抛错（带退出码 code），不带子进程输出。 */
export function run(bin, args, { capture = false, env = process.env } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(bin, args, { env, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    child.stdout.on('data', (s) => { if (capture && output.length < 1 << 20) output += s; });
    child.stderr.resume();
    child.on('error', () => reject(new Error(`${bin} 无法启动（没有安装？）`)));
    child.on('close', (code) => code === 0 ? resolvePromise(output.trim())
      : reject(Object.assign(new Error(`${bin} 失败（退出码 ${code}，详细输出已隐藏）`), { code })));
  });
}

/** ssh 的公共参数：不交互、连接超时，可选端口与密钥。 */
export function sshOptions(remote) {
  return ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
    ...(remote.port ? ['-p', String(remote.port)] : []), ...(remote.identityFile ? ['-i', remote.identityFile, '-o', 'IdentitiesOnly=yes'] : [])];
}

/** rsync 的远程路径：~/x 写成相对路径 x（rsync 的相对路径从远程家目录起算）。 */
const rsyncPath = (p) => (p.startsWith('~/') ? p.slice(2) : p);

/**
 * 远程调用：call(['ssh', 脚本]) 在远程用 sh 执行脚本；call(['push', 本机路径, 远程路径])、call(['pull', 远程路径, 本机路径]) 用 rsync -a 传。
 * 路径以 / 结尾时与 rsync 同义（传目录内容）。exec 可注入，测试用。
 */
export function remoteCall(remote, exec = run) {
  const ssh = sshOptions(remote);
  const e = ['ssh', ...ssh].join(' ');
  return (args, options = {}) => {
    const [verb, a, b] = args;
    if (verb === 'ssh') return exec('ssh', [...ssh, remote.host, `sh -c ${quote(a)}`], options);
    if (verb === 'push') return exec('rsync', ['-a', '-e', e, '--', a, `${remote.host}:${rsyncPath(b)}`], options);
    if (verb === 'pull') return exec('rsync', ['-a', '-e', e, '--', `${remote.host}:${rsyncPath(a)}`, b], options);
    throw new Error(`未知的远程调用：${verb}`);
  };
}

/** 试连 GPU 主机（ssh 执行 true）；返回 { host, call }。 */
export async function connectRemote(remote, exec = run) {
  const call = remoteCall(remote, exec);
  await call(['ssh', 'true']);
  return { host: remote.host, call };
}

// GPU 方案没有空闲 GPU 时的退出码（engine/dairy_engine/__main__.py）。
export const NO_GPU_EXIT = 3;
const RUN_ID_RE = /^[a-z0-9-]+$/;

/** 远程 Python 用的环境变量：环境目录与模型目录以 gpu.remote 为准，覆盖 dairy.config.json 的 gpu.envDir / gpu.modelRoot。 */
const engineEnv = (remote) => `DAIRY_ENGINE_ENV_DIR=${homePath(remote.envDir)} DAIRY_ENGINE_MODEL_ROOT=${homePath(remote.modelRoot)}`;

/**
 * 在 GPU 主机上启动当天转写：用 setsid nohup 脱离 ssh 会话在后台跑，flock 防止同时跑两份，
 * 结束后把退出码写到 logs/run-<runId>.exit。本机随后用 pollCommand 轮询，网络中途断开不会打断远程的转写。
 * relabelOnly：只用缓存重算认人与渲染（缓存不全就失败，不转写）。
 */
export function remoteCommand(remote, day, force, runId = 'x', relabelOnly = false) {
  resolveDay(day);
  if (!RUN_ID_RE.test(runId)) throw new Error('runId 无效');
  if (force && relabelOnly) throw new Error('--force 与 --relabel-only 不能同时使用');
  const flag = force ? ' --force' : relabelOnly ? ' --relabel-only' : '';
  const py = `PYTHONPATH=engine ${engineEnv(remote)} ${homePath(remote.envDir)}/venv/bin/python -m dairy_engine transcribe ${quote(day)}${flag}`;
  const done = `logs/run-${runId}.exit`;
  const job = `flock -n .dairy.lock sh -c ${quote(`${py} > logs/transcribe.log 2>&1`)}; code=$?; find raw transcripts cache -type d -exec chmod 700 {} +; echo $code > ${done}.part; mv ${done}.part ${done}`;
  return `set -eu; umask 077; cd ${homePath(remote.workDir)}; mkdir -p logs raw transcripts cache; chmod 700 raw transcripts cache logs; find raw transcripts cache -type d -exec chmod 700 {} +; setsid nohup sh -c ${quote(job)} > /dev/null 2>&1 < /dev/null &`;
}

/** 查询 remoteCommand 启动的转写：还在跑输出 running，结束输出退出码（读后删掉标记）。 */
export function pollCommand(remote, runId) {
  if (!RUN_ID_RE.test(runId)) throw new Error('runId 无效');
  const done = `logs/run-${runId}.exit`;
  return `cd ${homePath(remote.workDir)}; if test -f ${done}; then cat ${done}; rm -f ${done}; else echo running; fi`;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 轮询到远程转写结束；退出码非 0 时抛错（带 code）。连续 maxFailures 次连不上才放弃。 */
export async function waitRemote(call, remote, runId, { pollMs = 20000, maxFailures = 30 } = {}) {
  let failures = 0;
  for (;;) {
    await sleep(pollMs);
    let out;
    try { out = String(await call(['ssh', pollCommand(remote, runId)], { capture: true }) ?? '').trim(); failures = 0; }
    catch {
      failures += 1;
      if (failures > maxFailures) throw new Error('轮询远程转写状态连续失败，转写可能仍在 GPU 主机上进行');
      continue;
    }
    if (out === 'running') continue;
    const code = Number(out);
    if (!/^\d+$/.test(out)) throw new Error('GPU 主机返回的转写状态无法识别');
    if (code !== 0) throw Object.assign(new Error(`GPU 主机上转写失败（退出码 ${code}，详情见远程 ${remote.workDir}/logs/transcribe.log）`), { code });
    return;
  }
}

/**
 * 同步程序到 GPU 主机并建好 Python 环境、下载缺的模型（engine/setup_gpu.sh：锁文件没变就跳过建环境，模型已有就跳过）。
 * 只同步 engine/ 与 dairy.config.json，不同步整个仓库，防止 raw/、transcripts/ 等隐私数据随代码外传。
 */
export async function deployRemote(root, remote, call) {
  await call(['ssh', `set -eu; umask 077; mkdir -p ${homePath(remote.workDir)}; cd ${homePath(remote.workDir)}; mkdir -p logs raw transcripts cache; chmod 700 logs raw transcripts cache`]);
  for (const file of ['engine/', 'dairy.config.json']) await call(['push', join(root, file), `${remote.workDir}/${file}`]);
  await call(['ssh', `set -eu; umask 077; cd ${homePath(remote.workDir)}; flock -n .dairy.lock env ${engineEnv(remote)} bash engine/setup_gpu.sh ${homePath(remote.envDir)} > logs/engine-install.log 2>&1`]);
}

/** speakers.json 引用的旧写法录音（raw/<日期>/<文件名>）；clip 写法的片段整目录同步，见 pushClips。 */
async function enrollmentFiles(root) {
  const text = await readSpeakersText(root);
  const speakers = JSON.parse(text);
  if (!Array.isArray(speakers)) throw new Error('speakers.json 应为数组');
  const entries = speakers.flatMap((s) => s.enroll ?? []);
  for (const e of entries) if (e.clip !== undefined && !CLIP_RE.test(e.clip)) throw new Error('声纹 clip 必须为 voiceprints/clips/<文件名>.wav');
  const files = [...new Set(entries.filter((e) => e.clip === undefined).map((e) => e.file))];
  const rawRoot = files.length ? await realpath(join(root, 'raw')) : null;
  for (const file of files) {
    if (typeof file !== 'string' || !RAW_RE.test(file)) throw new Error('声纹引用必须为 raw/<日期>/<文件名>');
    if (!(await realpath(resolve(root, file))).startsWith(rawRoot + sep)) throw new Error('声纹引用不能越过 raw 目录');
  }
  return { files, hash: speakersHash(text) };
}

/**
 * 在 GPU 主机上转写一天。options：relabelOnly（只重算认人）；pushAudio（storage 为 local 时为真：
 * 先把本机 raw/<日期>/ 传过去，结束后删掉远程这份）；pollMs / maxFailures（轮询参数，见 waitRemote）。
 * 跑完后顺带拉回这份 speakers.json 的声纹缓存 cache/gpu-voiceprints-<hash>.json，本机登记工具据此与远程的认人结果一致。
 */
export async function transcribeRemote(root, remote, day, force, call, { relabelOnly = false, pushAudio = false, ...poll } = {}) {
  resolveDay(day);
  const { files, hash } = await enrollmentFiles(root);
  await deployRemote(root, remote, call);
  await call(['push', join(root, 'speakers.json'), `${remote.workDir}/speakers.json`]);
  await pushClips(root, { remote, call });
  for (const file of files) {
    const parent = file.slice(0, file.lastIndexOf('/'));
    await call(['ssh', `set -eu; umask 077; mkdir -p ${homePath(`${remote.workDir}/${parent}`)}; chmod 700 ${homePath(`${remote.workDir}/${parent}`)}`]);
    await call(['push', join(root, file), `${remote.workDir}/${file}`]);
  }
  const audio = pushAudio && !relabelOnly;
  const rawDay = `${remote.workDir}/raw/${day}`;
  try {
    if (audio) {
      await call(['ssh', `set -eu; umask 077; mkdir -p ${homePath(rawDay)}; chmod 700 ${homePath(`${remote.workDir}/raw`)} ${homePath(rawDay)}`]);
      await call(['push', `${join(root, 'raw', day)}/`, `${rawDay}/`]);
    }
    const runId = `${Date.now().toString(36)}-${randomBytes(4).toString('hex')}`;
    await call(['ssh', remoteCommand(remote, day, force, runId, relabelOnly)]);
    await waitRemote(call, remote, runId, poll);
  } finally {
    if (audio) await call(['ssh', `rm -rf ${homePath(rawDay)}`]).catch(() => {});
  }
  // 全部拉取成功才替换本地结果；失败时保留本地完整缓存供退回使用。
  await mkdir(join(root, 'cache'), { recursive: true, mode: 0o700 });
  const stage = await mkdtemp(join(root, 'cache', '.remote-result-'));
  try {
    for (const dir of ['transcripts', 'cache']) {
      await mkdir(join(stage, dir), { recursive: true, mode: 0o700 });
      await call(['pull', `${remote.workDir}/${dir}/${day}/`, `${join(stage, dir)}/`]);
    }
    for (const dir of ['transcripts', 'cache']) {
      await mkdir(join(root, dir, day), { recursive: true, mode: 0o700 });
      await cp(join(stage, dir), join(root, dir, day), { recursive: true });
    }
  } finally { await rm(stage, { recursive: true, force: true }); }
  await call(['pull', `${remote.workDir}/cache/gpu-voiceprints-${hash}.json`, join(root, 'cache') + '/']).catch(() => {});
}
