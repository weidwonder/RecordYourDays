// 声纹 clip：把录音里选中的片段裁成 16 kHz 单声道 float32 wav，存在 voiceprints/clips/（隐私数据，不进 git），
// 用远程 GPU 方案时在 GPU 主机的 <workDir>/voiceprints/clips/ 同步一份。storage 为 remote 时录音只存 GPU 主机，
// 所以连得上时在 GPU 主机上裁、再拉回本机；其余情况用本机 raw/ 下的录音在本机裁。
// 裁剪参数与 engine 的 decode_range 一致，clip 与原录音那一段逐点相同。
import { spawn } from 'node:child_process';
import { mkdir, readdir, rename, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { homePath, quote } from './shell.mjs';
import { CLIP_DIR, CLIP_RE, RAW_RE } from './speakers.mjs';

const exists = (p) => stat(p).then(() => true, () => false);
// GPU 主机上缺来源录音时脚本的退出码。
const MISSING_SOURCE_EXIT = 4;
// float32 wav：约 44–100 字节头 + 每秒 16000 × 4 字节。
const wavSeconds = (size) => (size - 44) / 64000;

/** ffmpeg 参数（不含 ffmpeg 本身），与 engine/dairy_engine/stages.py 的 decode_range 同样的 -ss/-t 与重采样。 */
export function clipArgs(src, start, end, dst) {
  return ['-nostdin', '-v', 'error', '-y', '-ss', start.toFixed(3), '-t', (end - start).toFixed(3), '-i', src,
    '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_f32le', '-f', 'wav', dst];
}

function check(items) {
  for (const it of items) {
    if (!RAW_RE.test(it.source ?? '')) throw new Error(`来源录音必须写成 raw/<日期>/<文件名>：${it.source}`);
    if (!CLIP_RE.test(it.clip ?? '')) throw new Error(`clip 路径无效：${it.clip}`);
    if (!(it.start >= 0 && it.end > it.start)) throw new Error(`片段起止无效：${it.source} ${it.start}-${it.end}`);
  }
}

/** 在 GPU 主机上裁 clip 的远程脚本：来源缺失时以 MISSING_SOURCE_EXIT 退出。 */
export function remoteClipScript(remote, items) {
  check(items);
  const lines = items.map((it) => {
    const part = `${it.clip}.part`;
    return `test -f ${quote(it.source)} || exit ${MISSING_SOURCE_EXIT}; ffmpeg ${clipArgs(it.source, it.start, it.end, part).map(quote).join(' ')} < /dev/null; mv ${quote(part)} ${quote(it.clip)}`;
  });
  return `set -eu; umask 077; cd ${homePath(remote.workDir)}; mkdir -p ${CLIP_DIR}; chmod 700 voiceprints ${CLIP_DIR}; ${lines.join('; ')}`;
}

function runFfmpeg(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('ffmpeg', args, { stdio: ['ignore', 'ignore', 'ignore'] });
    child.on('error', () => reject(new Error('本机没有 ffmpeg，裁不了 clip；请先安装 ffmpeg')));
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg 裁 clip 失败（退出码 ${code}）`))));
  });
}

/** 本机 clip 目录（700）。 */
export async function ensureClipDir(root) {
  await mkdir(join(root, CLIP_DIR), { recursive: true, mode: 0o700 });
}

/** 把本机全部 clip 推到 GPU 主机（rsync 增量，不删远程的）。 */
export async function pushClips(root, { remote, call }) {
  if ((await readdir(join(root, CLIP_DIR)).catch(() => [])).length === 0) return;
  await call(['ssh', `set -eu; umask 077; cd ${homePath(remote.workDir)}; mkdir -p ${CLIP_DIR}; chmod 700 voiceprints ${CLIP_DIR}`]);
  await call(['push', `${join(root, CLIP_DIR)}/`, `${remote.workDir}/${CLIP_DIR}/`]);
}

/** 把 GPU 主机上的 clip 拉回本机（只增不删）。 */
export async function pullClips(root, { remote, call }) {
  await ensureClipDir(root);
  await call(['ssh', `set -eu; umask 077; cd ${homePath(remote.workDir)}; mkdir -p ${CLIP_DIR}; chmod 700 voiceprints ${CLIP_DIR}`]);
  await call(['pull', `${remote.workDir}/${CLIP_DIR}/`, `${join(root, CLIP_DIR)}/`]);
}

/**
 * 裁出本机还没有的 clip。items: [{ clip, source, start, end }]。conn 为 GPU 主机连接（{ call }）或 null。
 * 返回 { cut: 新裁的个数, where: 'GPU 主机' | '本机' | null }。裁完核对每个 clip 的时长。
 */
export async function cutClips(root, items, { conn, remote }) {
  check(items);
  const todo = [];
  for (const it of items) if (!(await exists(join(root, it.clip))) && !todo.some((x) => x.clip === it.clip)) todo.push(it);
  if (todo.length === 0) return { cut: 0, where: null };
  await ensureClipDir(root);
  let where;
  if (conn) {
    try { await conn.call(['ssh', remoteClipScript(remote, todo)]); }
    catch (e) {
      if (e?.code === MISSING_SOURCE_EXIT) throw new Error('GPU 主机上没有来源录音（raw/<日期>/ 下找不到），裁不了 clip');
      throw new Error(`在 GPU 主机上裁 clip 失败（${e.message}）`);
    }
    await pullClips(root, { remote, call: conn.call });
    where = 'GPU 主机';
  } else {
    for (const it of todo) {
      if (!(await exists(join(root, it.source)))) throw new Error(`本机没有 ${it.source}，裁不了 clip`);
      const part = join(root, `${it.clip}.part`);
      await runFfmpeg(clipArgs(join(root, it.source), it.start, it.end, part));
      await rename(part, join(root, it.clip));
    }
    where = '本机';
  }
  for (const it of todo) {
    const st = await stat(join(root, it.clip)).catch(() => null);
    if (!st) throw new Error(`clip 没拿到：${it.clip}`);
    if (Math.abs(wavSeconds(st.size) - (it.end - it.start)) > 0.05) throw new Error(`clip 时长与片段不符：${it.clip}`);
  }
  return { cut: todo.length, where };
}
