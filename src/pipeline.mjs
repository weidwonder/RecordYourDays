// 一天的录音 → 转写稿。ASR、分离、embedding 结果缓存在 cache/<日期>/，标注与渲染每次现算。
import { mkdir, readdir, readFile, rename, writeFile } from 'node:fs/promises';
import { basename, extname, join } from 'node:path';
import { transcribe } from './asr.mjs';
import { decodeToMono16k } from './decode.mjs';
import { diarize } from './diarize.mjs';
import { embedRange } from './embedding.mjs';
import { labelUtterances } from './label.mjs';
import { renderTranscript } from './render.mjs';

const AUDIO_RE = /\.(mp3|m4a|wav|aac|flac|ogg|opus)$/i;
// 短于此的分离 span 不算 embedding，太短的片段声纹不可靠。
const MIN_SPAN_EMB_SEC = 0.3;

async function readJson(p) {
  return readFile(p, 'utf8').then(JSON.parse, () => null);
}

async function writeJson(p, v) {
  await writeFile(`${p}.part`, JSON.stringify(v));
  await rename(`${p}.part`, p);
}

const secs = (t0) => `${((Date.now() - t0) / 1000).toFixed(0)}s`;

/**
 * 转写一个音频文件并写出 outPath。people 为 voiceprint.mjs 的 loadVoiceprints 结果。
 * 缓存缺失或 force 时才解码并重算；ASR 或分离重算过，embedding 必须跟着重算。
 * 返回 [{ start, end, text, label }]。
 */
export async function transcribeFile(audioPath, { cacheDir, outPath, paths, people, force = false, log = () => {} }) {
  const base = basename(audioPath, extname(audioPath));
  const asrPath = join(cacheDir, `${base}.asr.json`);
  const diarPath = join(cacheDir, `${base}.diar.json`);
  const embPath = join(cacheDir, `${base}.emb.json`);
  await mkdir(cacheDir, { recursive: true });

  let samples;
  const getSamples = async () => {
    if (!samples) {
      const t0 = Date.now();
      samples = await decodeToMono16k(audioPath);
      log(`  解码 ${secs(t0)}（音频 ${(samples.length / 16000 / 60).toFixed(1)} min）`);
    }
    return samples;
  };

  let utts = force ? null : await readJson(asrPath);
  let spans = force ? null : await readJson(diarPath);
  let embs = force ? null : await readJson(embPath);
  if (!utts) {
    const s = await getSamples();
    const t0 = Date.now();
    utts = await transcribe(s, paths);
    await writeJson(asrPath, utts);
    embs = null;
    log(`  VAD+ASR ${secs(t0)}，${utts.length} 句`);
  }
  if (!spans) {
    const s = await getSamples();
    const t0 = Date.now();
    spans = await diarize(s, paths);
    await writeJson(diarPath, spans);
    embs = null;
    log(`  分离 ${secs(t0)}，${spans.length} 段 ${new Set(spans.map((x) => x.speaker)).size} 簇`);
  }
  if (!embs) {
    const s = await getSamples();
    const t0 = Date.now();
    embs = { utts: [], spans: [] };
    for (const u of utts) embs.utts.push(await embedRange(s, u.start, u.end, paths));
    for (const p of spans) embs.spans.push(p.end - p.start >= MIN_SPAN_EMB_SEC ? await embedRange(s, p.start, p.end, paths) : null);
    await writeJson(embPath, embs);
    log(`  embedding ${secs(t0)}`);
  }
  if (!samples) log('  ASR、分离、embedding 均用缓存');

  const labels = labelUtterances(utts, embs.utts, spans, embs.spans, people);
  const labeled = utts.map((u, i) => ({ ...u, label: labels[i] }));
  await writeFile(outPath, renderTranscript(labeled));
  return labeled;
}

/** 每个标签的总时长（秒），按逐句（合并前）计。 */
export function labelDurations(labeled) {
  const out = {};
  for (const u of labeled) out[u.label] = (out[u.label] ?? 0) + (u.end - u.start);
  return out;
}

export function formatDurations(d) {
  return Object.entries(d)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${v >= 60 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(0)} s`}`)
    .join('，');
}

/** 转写 raw/<day>/ 下全部音频到 transcripts/<day>/。 */
export async function transcribeDay(day, root, { paths, people, force = false, log = console.log }) {
  const rawDir = join(root, 'raw', day);
  const files = (await readdir(rawDir).catch(() => [])).filter((f) => AUDIO_RE.test(f)).sort();
  if (files.length === 0) throw new Error(`raw/${day}/ 下没有音频`);
  const outDir = join(root, 'transcripts', day);
  await mkdir(outDir, { recursive: true });
  const total = {};
  const tDay = Date.now();
  for (const [i, f] of files.entries()) {
    const t0 = Date.now();
    const out = join(outDir, `${basename(f, extname(f))}.md`);
    log(`[${i + 1}/${files.length}] ${f}`);
    const labeled = await transcribeFile(join(rawDir, f), { cacheDir: join(root, 'cache', day), outPath: out, paths, people, force, log });
    const d = labelDurations(labeled);
    for (const [k, v] of Object.entries(d)) total[k] = (total[k] ?? 0) + v;
    log(`  完成 ${secs(t0)} → transcripts/${day}/${basename(out)}：${formatDurations(d) || '无语音'}`);
  }
  log(`${day} 共 ${files.length} 个文件，${secs(tDay)}：${formatDurations(total) || '无语音'}`);
}

/**
 * CPU 方案只重算认人：按 cache/<day>/ 里成套的 .asr/.diar/.emb.json 重新标注并写转写稿，不读录音。
 * 返回处理的文件数；缓存不全时报错。
 */
export async function relabelDay(day, root, { people, log = console.log }) {
  const cacheDir = join(root, 'cache', day);
  const bases = (await readdir(cacheDir).catch(() => [])).filter((f) => f.endsWith('.asr.json') && !f.startsWith('gpu-')).map((f) => f.slice(0, -'.asr.json'.length)).sort();
  if (bases.length === 0) throw new Error(`cache/${day}/ 下没有 CPU 方案缓存，不能只重算认人`);
  const outDir = join(root, 'transcripts', day);
  await mkdir(outDir, { recursive: true });
  const total = {};
  for (const base of bases) {
    const [utts, spans, embs] = await Promise.all(['asr', 'diar', 'emb'].map((k) => readJson(join(cacheDir, `${base}.${k}.json`))));
    if (!utts || !spans || !embs) throw new Error(`${base} 的 CPU 方案缓存不全，不能只重算认人`);
    const labels = labelUtterances(utts, embs.utts, spans, embs.spans, people);
    const labeled = utts.map((u, i) => ({ ...u, label: labels[i] }));
    await writeFile(join(outDir, `${base}.md`), renderTranscript(labeled));
    const d = labelDurations(labeled);
    for (const [k, v] of Object.entries(d)) total[k] = (total[k] ?? 0) + v;
    log(`  ${base}.md：${formatDurations(d) || '无语音'}`);
  }
  log(`${day} 只重算认人（CPU 方案）${bases.length} 个文件：${formatDurations(total) || '无语音'}`);
  return bases.length;
}
