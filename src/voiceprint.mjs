// speakers.json → 每人一个声纹向量。声纹从注册片段现算，按 speakers.json 内容哈希缓存；
// 每个片段的 embedding 另按片段缓存（cache/enroll-embs.json），登记工具的相似度检查也用它。
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeToMono16k, SAMPLE_RATE } from './decode.mjs';
import { embedRange, meanEmb } from './embedding.mjs';
import { enrollAudio, loadSpeakers } from './speakers.mjs';

export { loadSpeakers } from './speakers.mjs';

async function readJson(p) {
  return readFile(p, 'utf8').then(JSON.parse, () => null);
}

async function writeJson(p, v) {
  await mkdir(join(p, '..'), { recursive: true });
  await writeFile(`${p}.part`, JSON.stringify(v));
  await rename(`${p}.part`, p);
}

/** 片段缓存键：clip 用文件内容的 sha256；旧写法（几小时的录音）用路径、大小、修改时间和区间。 */
async function enrollKey(root, e) {
  const a = enrollAudio(e);
  const p = join(root, a.file);
  const st = await stat(p).catch(() => null);
  if (!st) throw new Error(`注册片段的音频不在本机：${a.file}${e.clip ? '（配了 gpu.remote 时可用 npm run voice -- sync 从 GPU 主机取回）' : ''}`);
  if (e.clip !== undefined) return `${a.file}#${createHash('sha256').update(await readFile(p)).digest('hex').slice(0, 16)}`;
  return `${a.file}#${st.size}-${Math.floor(st.mtimeMs)}#${a.start}-${a.end}`;
}

/**
 * 每个注册片段的 embedding（L2 归一化），返回 Map<人名, [{ entry, vec }]>，顺序同 speakers.json。
 * 按文件分组，每个文件只解码一次，解完即释放（长录音解码后约 600 MB）。
 */
export async function embedEnrollments(root, speakers, paths) {
  const cachePath = join(root, 'cache', 'enroll-embs.json');
  const cache = (await readJson(cachePath)) ?? {};
  const keys = new Map();
  const byFile = new Map();
  for (const s of speakers)
    for (const e of s.enroll) {
      const k = await enrollKey(root, e);
      keys.set(e, k);
      if (cache[k]) continue;
      const { file } = enrollAudio(e);
      if (!byFile.has(file)) byFile.set(file, []);
      byFile.get(file).push({ name: s.name, e, k });
    }
  for (const [file, segs] of byFile) {
    const samples = await decodeToMono16k(join(root, file));
    for (const g of segs) {
      const a = enrollAudio(g.e);
      const v = await embedRange(samples, a.start, a.end ?? samples.length / SAMPLE_RATE, paths);
      if (!v) throw new Error(`注册片段太短，算不出 embedding：${g.name} ${file} ${a.start}-${a.end ?? '末尾'}`);
      cache[g.k] = v;
    }
  }
  if (byFile.size) await writeJson(cachePath, cache);
  return new Map(speakers.map((s) => [s.name, s.enroll.map((e) => ({ entry: e, vec: cache[keys.get(e)] }))]));
}

const person = (s, vec) => ({ name: s.name, label: s.display, role: s.role, threshold: s.threshold, vec });

/**
 * 返回 [{ name, label, role, threshold, vec }]，顺序同 speakers.json；label 是转写稿里显示的标签（display）。
 * 声纹 = 各注册片段 embedding（已 L2 归一化）等权平均后再归一化。
 */
export async function loadVoiceprints(root, paths, log = () => {}) {
  const { speakers, hash } = await loadSpeakers(root);
  const cachePath = join(root, 'cache', `voiceprints-${hash}.json`);
  let vecs = await readJson(cachePath);
  if (!vecs) {
    const t0 = Date.now();
    const embs = await embedEnrollments(root, speakers, paths);
    vecs = Object.fromEntries([...embs].map(([name, xs]) => [name, meanEmb(xs.map((x) => x.vec))]));
    await writeJson(cachePath, vecs);
    log(`声纹已按 speakers.json 重算（${((Date.now() - t0) / 1000).toFixed(0)}s）→ cache/voiceprints-${hash}.json`);
  }
  return speakers.map((s) => person(s, vecs[s.name]));
}

/** 本机已有 GPU 方案算的同一 speakers.json 的声纹缓存时用它，认人结果与 GPU 方案一致；否则返回 null。 */
export async function loadGpuVoiceprints(root) {
  const { speakers, hash } = await loadSpeakers(root);
  const vecs = await readJson(join(root, 'cache', `gpu-voiceprints-${hash}.json`));
  return vecs && speakers.every((s) => vecs[s.name]) ? speakers.map((s) => person(s, vecs[s.name])) : null;
}
