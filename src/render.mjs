// 带标签的句子 → 转写稿 markdown。
import { OTHER } from './label.mjs';

export const NO_SPEECH_WHOLE = '<no speech detected>';
export const NO_SPEECH_SEGMENT = '<no speech detected in this segment>';
const MERGE_GAP_SEC = 2;
const SILENCE_MARK_SEC = 30;

/** 秒 → mm:ss，满一小时为 h:mm:ss。 */
export function formatTimestamp(sec) {
  const t = Math.max(0, Math.floor(sec));
  const h = Math.floor(t / 3600);
  const mm = String(Math.floor((t % 3600) / 60)).padStart(2, '0');
  const ss = String(t % 60).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/**
 * 按开始时间排序后，把间隔 ≤ 2 s 的相邻同标签句合并成一句。
 * 「其他」不合并：它可能是不同的陌生人，拼成一行会让人误以为是同一个人连续在说。
 */
export function mergeAdjacent(utts) {
  const out = [];
  for (const u of [...utts].sort((a, b) => a.start - b.start)) {
    const last = out[out.length - 1];
    if (last && last.label !== OTHER && last.label === u.label && u.start - last.end <= MERGE_GAP_SEC) {
      last.end = u.end;
      last.text = `${last.text}${u.text}`;
    } else out.push({ ...u });
  }
  return out;
}

/** utts: [{ start, end, text, label }]。每行 `[时间] 标签 文本`；超过 30 s 的静音插一行标记。 */
export function renderTranscript(utts) {
  if (utts.length === 0) return `${NO_SPEECH_WHOLE}\n`;
  const lines = [];
  let prevEnd = 0;
  for (const u of mergeAdjacent(utts)) {
    if (u.start - prevEnd > SILENCE_MARK_SEC) lines.push(`[${formatTimestamp(prevEnd)}] ${NO_SPEECH_SEGMENT}`);
    lines.push(`[${formatTimestamp(u.start)}] ${u.label} ${u.text}`);
    prevEnd = u.end;
  }
  return `${lines.join('\n')}\n`;
}
