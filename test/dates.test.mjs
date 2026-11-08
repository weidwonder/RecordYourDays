import assert from 'node:assert/strict';
import { test } from 'node:test';
import { resolveDay, yesterdayIn, zonedParts, zonedToInstant } from '../src/dates.mjs';

test('yesterdayIn：按配置的时区算昨天，与本机时区无关', () => {
  // UTC 2026-11-02 02:30 = 上海 11-02 10:30 → 昨天 11-01；= 纽约 11-01 21:30 → 昨天 10-31
  assert.equal(yesterdayIn('Asia/Shanghai', new Date('2026-11-02T02:30:00Z')), '2026-11-01');
  assert.equal(yesterdayIn('America/New_York', new Date('2026-11-02T02:30:00Z')), '2026-10-31');
  // 上海 2026-11-02 00:10（UTC 11-01 16:10）→ 11-01；上海 11-01 23:50 → 10-31
  assert.equal(yesterdayIn('Asia/Shanghai', new Date('2026-11-01T16:10:00Z')), '2026-11-01');
  assert.equal(yesterdayIn('Asia/Shanghai', new Date('2026-11-01T15:50:00Z')), '2026-10-31');
  // 跨年；夏令时结束当天（纽约 2026-11-01 有 25 小时）按日历减一天
  assert.equal(yesterdayIn('Asia/Shanghai', new Date('2027-01-01T01:00:00Z')), '2026-12-31');
  assert.equal(yesterdayIn('America/New_York', new Date('2026-11-02T04:30:00Z')), '2026-10-31');
});

test('resolveDay：缺省为时区的昨天；给定日期须为真实的 YYYY-MM-DD', () => {
  assert.equal(resolveDay(undefined, new Date('2026-11-02T02:30:00Z'), 'Asia/Shanghai'), '2026-11-01');
  assert.equal(resolveDay('2026-11-01'), '2026-11-01');
  for (const bad of ['2026-11-1', '20261101', '2026-02-30', '2026-13-01', 'yesterday'])
    assert.throws(() => resolveDay(bad), /YYYY-MM-DD/);
});

test('zonedToInstant / zonedParts：墙上时间与时刻互换，含夏令时', () => {
  const t = zonedToInstant({ y: 2026, mo: 11, d: 3, h: 8, mi: 2, s: 53 }, 'Asia/Shanghai');
  assert.equal(new Date(t).toISOString(), '2026-11-03T00:02:53.000Z');
  assert.equal(new Date(zonedToInstant({ y: 2026, mo: 7, d: 1, h: 12, mi: 0, s: 0 }, 'America/New_York')).toISOString(), '2026-07-01T16:00:00.000Z');
  assert.equal(new Date(zonedToInstant({ y: 2026, mo: 12, d: 1, h: 12, mi: 0, s: 0 }, 'America/New_York')).toISOString(), '2026-12-01T17:00:00.000Z');
  assert.deepEqual(zonedParts(t, 'UTC'), { y: 2026, mo: 11, d: 3, h: 0, mi: 2, s: 53 });
});
