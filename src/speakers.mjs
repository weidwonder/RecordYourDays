// speakers.json 的读写与校验。每人：name、role（family 家人 / known 已登记的其他人）、display（转写稿里的标签）、threshold、enroll。
// enroll 两种写法：新写法 { clip, source, start, end } 引用 voiceprints/clips/ 下裁好的 16 kHz 单声道 wav，
// source/start/end 只记来源；旧写法 { file, start, end } 直接引用录音里的一段。两种可以混用。
import { createHash } from 'node:crypto';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { basename, extname, join } from 'node:path';

export const ROLES = ['family', 'known'];
export const CLIP_DIR = 'voiceprints/clips';
export const CLIP_RE = /^voiceprints\/clips\/[a-zA-Z0-9_.-]+\.wav$/;
export const RAW_RE = /^raw\/\d{4}-\d{2}-\d{2}\/[a-zA-Z0-9_.-]+$/;
// 新登记的人缺省阈值（偏保守）。
export const DEFAULT_THRESHOLD = 0.4;

/** 校验并补全缺省值：role 缺省 family，display 缺省为名字。返回新数组，不改入参。 */
export function normalizeSpeakers(speakers) {
  if (!Array.isArray(speakers) || speakers.length === 0) throw new Error('speakers.json 应为非空数组');
  const names = new Set();
  return speakers.map((s) => {
    if (typeof s.name !== 'string' || !s.name || names.has(s.name)) throw new Error(`speakers.json：名字为空或重复：${s.name}`);
    names.add(s.name);
    const role = s.role ?? 'family';
    if (!ROLES.includes(role)) throw new Error(`speakers.json：${s.name} 的 role 只能是 family 或 known`);
    const display = s.display ?? s.name;
    if (typeof display !== 'string' || !display || /\s/.test(display)) throw new Error(`speakers.json：${s.name} 的 display 不能为空或含空白`);
    if (typeof s.threshold !== 'number') throw new Error(`speakers.json：${s.name} 缺少数值 threshold`);
    if (!Array.isArray(s.enroll) || s.enroll.length === 0) throw new Error(`speakers.json：${s.name} 没有注册片段`);
    for (const e of s.enroll) {
      const bad = () => new Error(`speakers.json：${s.name} 的注册片段无效：${JSON.stringify(e)}`);
      if (e.clip !== undefined) {
        if (typeof e.clip !== 'string' || !CLIP_RE.test(e.clip)) throw bad();
      } else if (typeof e.file !== 'string' || !(e.start < e.end)) throw bad();
    }
    return { ...s, name: s.name, role, display, threshold: s.threshold, enroll: s.enroll };
  });
}

export function parseSpeakers(text) {
  return normalizeSpeakers(JSON.parse(text));
}

/** speakers.json 的内容哈希（声纹缓存键），与 engine/dairy_engine/voiceprint.py 一致。 */
export const speakersHash = (text) => createHash('sha256').update(text).digest('hex').slice(0, 16);

/** speakers.json 原文；没有这个文件时提示从示例复制。 */
export async function readSpeakersText(root) {
  try { return await readFile(join(root, 'speakers.json'), 'utf8'); }
  catch (e) {
    if (e.code === 'ENOENT') throw new Error('找不到 speakers.json：先登记你自己的声纹，例如 npm run voice -- add 我 <录音名> <开始> <结束> --whole --role family（格式见 speakers.example.json）');
    throw e;
  }
}

/** 读并校验 speakers.json，返回 { speakers, hash, text }。 */
export async function loadSpeakers(root) {
  const text = await readSpeakersText(root);
  return { speakers: parseSpeakers(text), hash: speakersHash(text), text };
}

/** 片段的取音位置：clip 取整个文件（end 为 null），旧写法取录音里的一段。 */
export const enrollAudio = (e) => (e.clip !== undefined ? { file: e.clip, start: 0, end: null } : { file: e.file, start: e.start, end: e.end });

/** 片段的来源录音（相对路径）与来源日期。 */
export function enrollSource(e) {
  const file = e.clip !== undefined ? e.source : e.file;
  return { file, day: /raw\/(\d{4}-\d{2}-\d{2})\//.exec(file ?? '')?.[1] ?? null };
}

/** 片段时长（秒）：两种写法都记着来源的 start/end。 */
export const enrollSeconds = (e) => (typeof e.start === 'number' && typeof e.end === 'number' ? e.end - e.start : 0);

const r3 = (x) => Math.round(x * 1000) / 1000;

/** 来源录音的一段 → clip 相对路径：voiceprints/clips/<录音名>_<起点毫秒>-<终点毫秒>.wav。只用 ASCII，远程路径校验能过。 */
export function clipPath(source, start, end) {
  const base = basename(source, extname(source));
  return `${CLIP_DIR}/${base}_${Math.round(start * 1000)}-${Math.round(end * 1000)}.wav`;
}

/** 新写法的片段。 */
export const clipEntry = (source, start, end) => ({ clip: clipPath(source, r3(start), r3(end)), source, start: r3(start), end: r3(end) });

const inline = (o) => `{${Object.entries(o).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(', ')}}`;

/** 写出格式：每人一段，enroll 每个片段一行，便于看 diff。 */
export function formatSpeakers(speakers) {
  const people = normalizeSpeakers(speakers).map(({ name, role, display, threshold, enroll, ...rest }) => {
    const head = Object.entries({ name, role, display, threshold, ...rest }).map(([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v)},`);
    return `  {\n${head.join('\n')}\n    "enroll": [\n${enroll.map((e) => `      ${inline(e)}`).join(',\n')}\n    ]\n  }`;
  });
  return `[\n${people.join(',\n')}\n]\n`;
}

export async function writeSpeakers(root, speakers) {
  const p = join(root, 'speakers.json');
  const text = formatSpeakers(speakers);
  await writeFile(`${p}.part`, text);
  await rename(`${p}.part`, p);
  return { text, hash: speakersHash(text) };
}

/** 本机必须常驻的录音：旧写法片段引用的录音。clip 写法的来源录音不需要留在本机。 */
export const rawReferences = (speakers) => speakers.flatMap((s) => (s.enroll ?? []).filter((e) => e.clip === undefined && e.file !== undefined).map((e) => e.file));

/** 所有 clip 引用（去重）。 */
export const clipReferences = (speakers) => [...new Set(speakers.flatMap((s) => (s.enroll ?? []).filter((e) => e.clip !== undefined).map((e) => e.clip)))];
