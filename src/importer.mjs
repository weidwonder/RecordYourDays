// 从录音笔拷某天的录音到本机 raw/<日期>/，sha256 校验。storage 为 local 时录音就存这里；
// 为 remote 时这是连不上 GPU 主机时的暂存，下次连上补传（见 storage.mjs）。
import { chmod, copyFile, mkdir, rename, stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { recorderFiles } from './recorder.mjs';
import { sha256File as sha256 } from './storage.mjs';

/** 返回 { copied, skipped }（文件名列表）。config 为 loadConfig 的结果。 */
export async function importDay(day, root, config, { log = console.log } = {}) {
  const recordDir = config.recorder.dir;
  const files = await recorderFiles(day, config);
  if (files === null) throw new Error(`找不到 ${recordDir}：录音笔没有挂载`);
  const destDir = join(root, 'raw', day);
  const copied = [];
  const skipped = [];
  if (files.length === 0) {
    log(`录音笔上没有 ${day} 的录音`);
    return { copied, skipped };
  }
  await mkdir(destDir, { recursive: true });
  for (const f of files) {
    const src = join(recordDir, f);
    const dest = join(destDir, f);
    const srcSha = await sha256(src);
    const existed = !!(await stat(dest).catch(() => null));
    if (existed && (await sha256(dest)) === srcSha) {
      skipped.push(f);
      log(`[跳过] ${f} 已存在且内容相同`);
      continue;
    }
    const part = `${dest}.part`;
    await copyFile(src, part);
    if ((await sha256(part)) !== srcSha) {
      await unlink(part);
      throw new Error(`拷贝后校验不一致：${f}`);
    }
    await chmod(part, 0o644);
    await rename(part, dest);
    copied.push(f);
    log(existed ? `[覆盖] ${f} 已存在但内容不同，已按录音笔重拷` : `[拷贝] ${f}`);
  }
  return { copied, skipped };
}
