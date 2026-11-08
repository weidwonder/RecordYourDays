// CPU 方案的模型：文件路径、齐全检查与下载（npm run models，从 sherpa-onnx 官方 GitHub Releases 下载）。
import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdir, mkdtemp, rename, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const RELEASES = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';

// 目录布局 <modelDir>/<id>/<version>/<file>。
// asr 压缩包里的 bpe.model 只有热词要用，这里不用热词，故不列。
// url 是压缩包（.tar.bz2，解开后是 archiveDir/<file>）或单个文件。
export const COMPONENTS = {
  asr: {
    version: 'sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03',
    url: `${RELEASES}/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2`,
    archiveDir: 'sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03',
    files: {
      encoder: 'encoder-epoch-99-avg-1.int8.onnx',
      decoder: 'decoder-epoch-99-avg-1.onnx',
      joiner: 'joiner-epoch-99-avg-1.int8.onnx',
      tokens: 'tokens.txt',
    },
  },
  vad: { version: 'silero_vad', url: `${RELEASES}/asr-models/silero_vad.onnx`, files: { model: 'silero_vad.onnx' } },
  'speaker-seg': {
    version: 'sherpa-onnx-pyannote-segmentation-3-0',
    url: `${RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`,
    archiveDir: 'sherpa-onnx-pyannote-segmentation-3-0',
    files: { model: 'model.int8.onnx' },
  },
  'speaker-embed': {
    version: '3dspeaker_campplus_sv_zh_en_16k-common_advanced',
    url: `${RELEASES}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`,
    files: { model: '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx' },
  },
};

/** { asr: { encoder, decoder, joiner, tokens }, vad: { model }, 'speaker-seg': { model }, 'speaker-embed': { model } }，均为绝对路径。 */
export function modelPaths(modelDir) {
  const out = {};
  for (const [id, c] of Object.entries(COMPONENTS)) {
    out[id] = {};
    for (const [k, f] of Object.entries(c.files)) out[id][k] = join(modelDir, id, c.version, f);
  }
  return out;
}

const present = async (p) => { const st = await stat(p).catch(() => null); return !!st?.isFile() && st.size > 0; };

/** 缺文件（或文件为空）时抛错，列出全部缺失路径；齐全时返回 modelPaths。 */
export async function requireModels(modelDir) {
  const paths = modelPaths(modelDir);
  const missing = [];
  for (const group of Object.values(paths)) for (const p of Object.values(group)) if (!(await present(p))) missing.push(p);
  if (missing.length) throw new Error(`模型文件缺失，先跑 npm run models 下载（modelDir：${modelDir}）：\n  ${missing.join('\n  ')}`);
  return paths;
}

/** 下载到 dest；服务器给了 Content-Length 时核对字节数，不一致报错。 */
async function download(url, dest, fetchFn) {
  const res = await fetchFn(url, { redirect: 'follow' });
  if (!res.ok || !res.body) throw new Error(`下载失败（HTTP ${res.status}）：${url}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
  const want = Number(res.headers.get('content-length'));
  const got = (await stat(dest)).size;
  if (want && got !== want) throw new Error(`下载不完整（${got}/${want} 字节）：${url}`);
  return got;
}

function untar(file, dir) {
  return new Promise((resolve, reject) => {
    const child = spawn('tar', ['-xjf', file, '-C', dir], { stdio: ['ignore', 'ignore', 'inherit'] });
    child.on('error', () => reject(new Error('本机没有 tar，解不开模型压缩包')));
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`解压失败（退出码 ${code}）：${file}`))));
  });
}

/**
 * 下载缺的模型到 modelDir：已存在且非空的组件跳过；先下到临时目录，齐了再移到位，中途失败不留半截文件。
 * 返回 { downloaded: [组件], skipped: [组件] }。
 */
export async function downloadModels(modelDir, { fetchFn = fetch, log = console.log } = {}) {
  const paths = modelPaths(modelDir);
  const out = { downloaded: [], skipped: [] };
  await mkdir(modelDir, { recursive: true });
  for (const [id, c] of Object.entries(COMPONENTS)) {
    const targets = Object.values(paths[id]);
    if ((await Promise.all(targets.map(present))).every(Boolean)) { out.skipped.push(id); log(`[跳过] ${id} 已存在`); continue; }
    log(`[下载] ${id} ← ${c.url}`);
    const tmp = await mkdtemp(join(modelDir, `.download-${id}-`));
    try {
      const file = join(tmp, c.url.slice(c.url.lastIndexOf('/') + 1));
      const bytes = await download(c.url, file, fetchFn);
      if (c.archiveDir) await untar(file, tmp);
      const dir = join(modelDir, id, c.version);
      await mkdir(dir, { recursive: true });
      for (const f of Object.values(c.files)) {
        const src = c.archiveDir ? join(tmp, c.archiveDir, f) : file;
        if (!(await present(src))) throw new Error(`${id} 的下载内容里没有 ${f}`);
        await rename(src, join(dir, f));
      }
      out.downloaded.push(id);
      log(`  完成 ${(bytes / 2 ** 20).toFixed(1)} MB → ${dir}`);
    } finally { await rm(tmp, { recursive: true, force: true }); }
  }
  return out;
}
