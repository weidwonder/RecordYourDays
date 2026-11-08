// 录音存放。storage 为 local 时录音存本机 raw/<日期>/（见 importer.mjs）；为 remote 时 GPU 主机是录音的长期存放处，
// 本机只留旧写法声纹引用的录音、keepLocal 列出的文件，以及连不上 GPU 主机时的暂存。
// 声纹 clip（voiceprints/clips/）不归这里管，见 clips.mjs。
// 所有删除都以删除前当场算出的 sha256 为准。
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readdir, rmdir, stat, unlink } from 'node:fs/promises';
import { basename, extname, join } from 'node:path';
import { resolveDay } from './dates.mjs';
import { recorderFiles } from './recorder.mjs';
import { homePath, quote } from './shell.mjs';
import { rawReferences, readSpeakersText } from './speakers.mjs';

export { recorderFiles } from './recorder.mjs';
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const AUDIO_RE = /^[a-zA-Z0-9_.-]+\.(mp3|m4a|wav|aac|flac|ogg|opus)$/i;
const REL_RE = /^raw\/\d{4}-\d{2}-\d{2}\/[a-zA-Z0-9_.-]+$/;

export async function sha256File(p) {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(p)) h.update(chunk);
  return h.digest('hex');
}

const exists = (p) => stat(p).then(() => true, () => false);

/**
 * 本机必须常驻的录音（相对项目根）：speakers.json 里旧写法片段引用的录音，加 config.keepLocal。读不出来就报错，宁可不删。
 * clip 写法的片段已裁成 voiceprints/clips/ 下的小文件，来源录音不必留在本机。
 */
export async function keepLocalSet(root, config) {
  const speakers = JSON.parse(await readSpeakersText(root));
  if (!Array.isArray(speakers)) throw new Error('speakers.json 应为数组');
  const files = [...rawReferences(speakers), ...(config.keepLocal ?? [])];
  for (const f of files) if (typeof f !== 'string' || !REL_RE.test(f)) throw new Error('本机保留的录音必须写成 raw/<日期>/<文件名>');
  return new Set(files);
}

/** GPU 主机上 workDir 下若干相对路径的 sha256；不存在的文件不出现在结果里。 */
export async function remoteHashes(call, remote, rels) {
  const out = new Map();
  if (rels.length === 0) return out;
  for (const r of rels) if (!REL_RE.test(r)) throw new Error('远程录音路径必须为 raw/<日期>/<文件名>');
  const text = await call(['ssh', `cd ${homePath(remote.workDir)} && { sha256sum -- ${rels.map(quote).join(' ')} 2>/dev/null || true; }`], { capture: true });
  for (const line of String(text ?? '').split('\n')) {
    const m = /^([0-9a-f]{64}) [ *]?(\S+)$/.exec(line.trim());
    if (m && rels.includes(m[2])) out.set(m[2], m[1]);
  }
  return out;
}

async function ensureRemoteDays(call, remote, days) {
  const dirs = [...new Set(days)].map((d) => `raw/${resolveDay(d)}`);
  await call(['ssh', `set -eu; umask 077; cd ${homePath(remote.workDir)}; mkdir -p ${dirs.map(quote).join(' ')}; chmod 700 raw ${dirs.map(quote).join(' ')}`]);
}

const chmodRemote = (call, remote, rels) => rels.length === 0 ? null
  : call(['ssh', `cd ${homePath(remote.workDir)} && chmod 600 ${rels.map(quote).join(' ')}`]);

/** 录音笔 → GPU 主机的 raw/<日期>/，传完逐个核对 sha256。返回 { uploaded, skipped }。 */
export async function uploadRecorder(day, config, { remote, call, log = console.log }) {
  const recordDir = config.recorder.dir;
  const files = await recorderFiles(day, config);
  if (files === null) throw new Error(`找不到 ${recordDir}：录音笔没有挂载`);
  const uploaded = [];
  const skipped = [];
  if (files.length === 0) { log(`录音笔上没有 ${day} 的录音`); return { uploaded, skipped }; }
  await ensureRemoteDays(call, remote, [day]);
  const rel = (f) => `raw/${day}/${f}`;
  const before = await remoteHashes(call, remote, files.map(rel));
  const want = new Map();
  for (const f of files) {
    const sha = await sha256File(join(recordDir, f));
    want.set(f, sha);
    if (before.get(rel(f)) === sha) { skipped.push(f); log(`[跳过] ${f} GPU 主机上已有且内容相同`); continue; }
    await call(['push', join(recordDir, f), `${remote.workDir}/${rel(f)}`]);
    uploaded.push(f);
  }
  await chmodRemote(call, remote, files.map(rel));
  const after = await remoteHashes(call, remote, uploaded.map(rel));
  for (const f of uploaded) {
    if (after.get(rel(f)) !== want.get(f)) throw new Error(`传到 GPU 主机后校验不一致：${f}`);
    log(before.has(rel(f)) ? `[覆盖] ${f} GPU 主机上内容不同，已按录音笔重传` : `[上传] ${f} → GPU 主机`);
  }
  return { uploaded, skipped };
}

async function localRecordings(root) {
  const days = (await readdir(join(root, 'raw')).catch(() => [])).filter((d) => DAY_RE.test(d)).sort();
  const out = [];
  for (const day of days) {
    for (const f of (await readdir(join(root, 'raw', day)).catch(() => [])).filter((x) => AUDIO_RE.test(x)).sort())
      if ((await stat(join(root, 'raw', day, f))).isFile()) out.push({ day, f, rel: `raw/${day}/${f}` });
  }
  return out;
}

/**
 * 本机 raw/ 下的录音补传 GPU 主机并核对 sha256。远程没有就传，内容相同算已核对，内容不同两边都不动并报告。
 * remove 为真时删掉已核对且不在 keep 里的本机文件。返回 { uploaded, verified, removed, conflicts }（相对路径）。
 */
export async function syncLocalRaw(root, { remote, call, keep = new Set(), remove = false, log = console.log }) {
  const items = await localRecordings(root);
  const result = { uploaded: [], verified: [], removed: [], conflicts: [] };
  if (items.length === 0) return result;
  await ensureRemoteDays(call, remote, items.map((x) => x.day));
  const before = await remoteHashes(call, remote, items.map((x) => x.rel));
  const local = new Map();
  for (const x of items) {
    const sha = await sha256File(join(root, x.rel));
    local.set(x.rel, sha);
    const there = before.get(x.rel);
    if (there === undefined) { await call(['push', join(root, x.rel), `${remote.workDir}/${x.rel}`]); result.uploaded.push(x.rel); }
    else if (there !== sha) { result.conflicts.push(x.rel); log(`[冲突] ${x.rel}：本机与 GPU 主机上的内容不同，两边都保留，请人工核对`); }
  }
  const checked = items.map((x) => x.rel).filter((r) => !result.conflicts.includes(r));
  await chmodRemote(call, remote, checked);
  const after = await remoteHashes(call, remote, result.uploaded);
  for (const r of checked) {
    if ((after.get(r) ?? before.get(r)) === local.get(r)) result.verified.push(r);
    else log(`[未核对] ${r}：传到 GPU 主机后校验不一致，本机保留`);
  }
  for (const r of result.uploaded) if (result.verified.includes(r)) log(`[补传] ${r} → GPU 主机`);
  if (remove) {
    for (const r of result.verified) {
      if (keep.has(r)) continue;
      await unlink(join(root, r));
      result.removed.push(r);
      log(`[删本机] ${r}（GPU 主机上已核对一致）`);
    }
    for (const day of new Set(result.removed.map((r) => r.split('/')[1]))) await rmdir(join(root, 'raw', day)).catch(() => {});
  }
  return result;
}

export async function hasLocalAudio(root, day) {
  return (await readdir(join(root, 'raw', day)).catch(() => [])).some((f) => AUDIO_RE.test(f));
}

/** 需要在本机转写而录音在 GPU 主机上时，把当天录音临时取回本机；之后由 syncLocalRaw 核对删除。 */
export async function pullDay(root, { remote, call, day }) {
  resolveDay(day);
  await mkdir(join(root, 'raw', day), { recursive: true });
  await call(['pull', `${remote.workDir}/raw/${day}/`, `${join(root, 'raw', day)}/`]);
}

/**
 * 删录音笔原件：拷贝（传了 call 时核对 GPU 主机上的，否则核对本机 raw/）的 sha256 与原件一致，且转写稿已生成，才删。
 * 返回 { removed, kept: [{ file, reasons }] }；录音笔未挂载返回 null。
 */
export async function cleanRecorder(day, root, config, { remote, call, log = console.log }) {
  const recordDir = config.recorder.dir;
  const files = await recorderFiles(day, config);
  if (files === null) { log('录音笔未挂载，跳过删除原件'); return null; }
  const rel = (f) => `raw/${day}/${f}`;
  const there = call ? await remoteHashes(call, remote, files.map(rel)) : null;
  const removed = [];
  const kept = [];
  for (const f of files) {
    const reasons = [];
    const sha = await sha256File(join(recordDir, f));
    if (there) { if (there.get(rel(f)) !== sha) reasons.push('GPU 主机上没有内容一致的拷贝'); }
    else {
      const p = join(root, rel(f));
      if (!(await exists(p)) || (await sha256File(p)) !== sha) reasons.push('本机 raw/ 下没有内容一致的拷贝');
    }
    const md = `transcripts/${day}/${basename(f, extname(f))}.md`;
    if (!(await exists(join(root, md)))) reasons.push(`${md} 还没生成`);
    if (reasons.length) { kept.push({ file: f, reasons }); log(`[保留原件] ${f}：${reasons.join('；')}`); continue; }
    await unlink(join(recordDir, f));
    removed.push(f);
    log(`[删原件] ${f}`);
  }
  return { removed, kept };
}
