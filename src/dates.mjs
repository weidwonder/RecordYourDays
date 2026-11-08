// 日期与时区。「昨天」按 dairy.config.json 的 timeZone 算，与本机时区无关。
const DAY_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

/** 本机时区（timeZone 没配时的缺省值）。 */
export const systemTimeZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone;

/** 时区名无效时抛错。 */
export function checkTimeZone(tz, what = 'timeZone') {
  const bad = () => new Error(`${what} 不是有效的 IANA 时区名（例如 Asia/Shanghai）：${tz}`);
  if (typeof tz !== 'string' || !tz) throw bad();
  try { new Intl.DateTimeFormat('en-US', { timeZone: tz }); } catch { throw bad(); }
  return tz;
}

const fmtCache = new Map();
function formatter(tz) {
  if (!fmtCache.has(tz)) fmtCache.set(tz, new Intl.DateTimeFormat('en-US', { timeZone: tz, hourCycle: 'h23',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  return fmtCache.get(tz);
}

/** 时刻 t（毫秒）在时区 tz 的墙上时间：{ y, mo, d, h, mi, s }。 */
export function zonedParts(t, tz) {
  const p = Object.fromEntries(formatter(tz).formatToParts(new Date(t)).map((x) => [x.type, Number(x.value)]));
  return { y: p.year, mo: p.month, d: p.day, h: p.hour, mi: p.minute, s: p.second };
}

const wallUtc = (f) => Date.UTC(f.y, f.mo - 1, f.d, f.h ?? 0, f.mi ?? 0, f.s ?? 0);

/** 时区 tz 的墙上时间 → 时刻（毫秒）。夏令时切换处取两次修正后的结果。 */
export function zonedToInstant(f, tz) {
  const guess = wallUtc(f);
  const offset = (t) => wallUtc(zonedParts(t, tz)) - Math.floor(t / 1000) * 1000;
  const t1 = guess - offset(guess);
  return guess - offset(t1);
}

const pad = (n, w = 2) => String(n).padStart(w, '0');
export const formatDay = (f) => `${pad(f.y, 4)}-${pad(f.mo)}-${pad(f.d)}`;

/** 时区 tz 的昨天，YYYY-MM-DD（按日历减一天，夏令时不影响）。 */
export function yesterdayIn(tz, now = new Date()) {
  const f = zonedParts(now.getTime(), tz);
  const d = new Date(Date.UTC(f.y, f.mo - 1, f.d - 1));
  return d.toISOString().slice(0, 10);
}

/** 缺省取 tz 的昨天；给了就校验是真实存在的 YYYY-MM-DD。 */
export function resolveDay(arg, now = new Date(), tz = systemTimeZone()) {
  if (arg === undefined) return yesterdayIn(tz, now);
  const m = DAY_RE.exec(arg);
  const d = m && new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  if (!d || d.toISOString().slice(0, 10) !== arg) throw new Error(`日期应为 YYYY-MM-DD：${arg}`);
  return arg;
}
