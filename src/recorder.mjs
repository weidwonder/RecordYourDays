// 录音笔：挂载目录、文件名规则、文件名所在时区（dairy.config.json 的 recorder）。
// 文件名里带录音开始的时刻；按 recorder.timeZone 解读，再换算到 timeZone 决定归哪一天。
import { readdir, stat } from 'node:fs/promises';
import { formatDay, zonedParts, zonedToInstant } from './dates.mjs';

export const DEFAULT_PATTERN = 'YYYYMMDD_HHmmss';
export const AUDIO_EXTS = ['.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg', '.opus'];
const TOKENS = ['YYYY', 'MM', 'DD', 'HH', 'mm', 'ss'];
const FIELD = { YYYY: 'y', MM: 'mo', DD: 'd', HH: 'h', mm: 'mi', ss: 's' };

/**
 * 文件名规则（不含扩展名）→ { re, order }。占位符 YYYY MM DD HH mm ss 各出现一次，
 * 其余字符原样匹配，只能是字母、数字、_ . -（远程路径校验只认这些字符）。
 */
export function compilePattern(pattern) {
  if (typeof pattern !== 'string' || !pattern) throw new Error('recorder.pattern 不能为空');
  const order = [];
  let re = '';
  for (let i = 0; i < pattern.length;) {
    const tok = TOKENS.find((t) => pattern.startsWith(t, i));
    if (tok) {
      if (order.includes(tok)) throw new Error(`recorder.pattern 里 ${tok} 出现了两次：${pattern}`);
      order.push(tok);
      re += tok === 'YYYY' ? '(\\d{4})' : '(\\d{2})';
      i += tok.length;
    } else {
      const c = pattern[i];
      if (!/[A-Za-z0-9_.-]/.test(c)) throw new Error(`recorder.pattern 只能含字母、数字、_ . - 和占位符：${pattern}`);
      re += /[.-]/.test(c) ? `\\${c}` : c;
      i += 1;
    }
  }
  const missing = TOKENS.filter((t) => !order.includes(t));
  if (missing.length) throw new Error(`recorder.pattern 缺少占位符 ${missing.join(' ')}：${pattern}`);
  return { re: new RegExp(`^${re}$`), order };
}

/** 文件名（不含扩展名）→ 文件名里的墙上时间 { y, mo, d, h, mi, s }；不符合规则或不是真实时刻时返回 null。 */
export function parseStamp(base, pattern = DEFAULT_PATTERN) {
  const { re, order } = compilePattern(pattern);
  const m = re.exec(base);
  if (!m) return null;
  const f = {};
  order.forEach((tok, i) => { f[FIELD[tok]] = Number(m[i + 1]); });
  const t = new Date(Date.UTC(f.y, f.mo - 1, f.d, f.h, f.mi, f.s));
  if (t.getUTCMonth() !== f.mo - 1 || t.getUTCDate() !== f.d || f.h > 23 || f.mi > 59 || f.s > 59) return null;
  return f;
}

/**
 * 录音名 → { day, clock }：归哪一天（timeZone 下）与开始时刻是当天第几秒（录音笔的名义开始时刻）。
 * config 为 loadConfig 的结果（用 recorder.pattern、recorder.timeZone、timeZone）。不符合规则时返回 null。
 */
export function recordingInfo(base, config) {
  const f = parseStamp(base, config.recorder.pattern);
  if (!f) return null;
  const local = config.recorder.timeZone === config.timeZone ? f : zonedParts(zonedToInstant(f, config.recorder.timeZone), config.timeZone);
  return { day: formatDay(local), clock: local.h * 3600 + local.mi * 60 + local.s };
}

/** 扩展名是否在 recorder.extensions 里（不分大小写）。 */
export const hasRecorderExt = (name, config) => config.recorder.extensions.some((e) => name.toLowerCase().endsWith(e.toLowerCase()));

/** 录音笔上某天的录音文件名（排序）；录音笔没挂载时返回 null。 */
export async function recorderFiles(day, config) {
  const dir = config.recorder.dir;
  if (!dir) throw new Error('dairy.config.json 没配 recorder.dir（录音笔上存录音的目录）');
  if (!(await stat(dir).catch(() => null))?.isDirectory()) return null;
  const ext = (f) => f.slice(f.lastIndexOf('.'));
  return (await readdir(dir))
    .filter((f) => f.includes('.') && hasRecorderExt(f, config) && recordingInfo(f.slice(0, -ext(f).length), config)?.day === day)
    .sort();
}
