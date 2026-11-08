// 登记声纹的命令行：npm run voice -- <子命令>。用法见 README「登记声纹」。
// 簇和片段都来自 GPU 方案缓存 cache/<日期>/gpu-<文件名>.{diar,embed}.json；登记的片段裁成 voiceprints/clips/ 下的小 wav。
// 输出只有时间位置、时长、计数和相似度；句子文字只在 clusters --text 时给出。
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync, realpathSync } from 'node:fs';
import { basename, dirname, extname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { cutClips, pullClips, pushClips } from './clips.mjs';
import { loadConfig, ROOT } from './config.mjs';
import { resolveDay } from './dates.mjs';
import { checkEnrollment, clusterReport, parseTime, pickCluster, pickRange, speakerCentroids } from './enroll.mjs';
import { configureInference } from './inference.mjs';
import { setUnknownLabel } from './label.mjs';
import { requireModels } from './models.mjs';
import { recordingInfo } from './recorder.mjs';
import { connectRemote } from './remote.mjs';
import { formatTimestamp } from './render.mjs';
import { CLIP_DIR, DEFAULT_THRESHOLD, ROLES, clipEntry, clipReferences, enrollSeconds, enrollSource, loadSpeakers, writeSpeakers } from './speakers.mjs';
import { embedEnrollments, loadGpuVoiceprints, loadVoiceprints } from './voiceprint.mjs';

export const USAGE = `用法：npm run voice -- <子命令> [参数]（各子命令都可加 --json）
  clusters <日期> [--file <录音名>] [--text] [--out <路径>]      列出当天每个录音的 DiariZen 说话人簇
  add <名字> <录音> <开始> <结束> [--cluster <簇号>|--all-speakers|--whole] [--min 1.5]
                                                            按时间段登记（时间为秒数或 h:mm:ss，录音内的位置）
  add-cluster <名字> <日期> <录音> <簇号> [--max 20]            把整个簇登记成某人
  （add 与 add-cluster 可加 --role family|known --display <标签> --threshold <阈值> --dry-run --local）
  list                                                      每人的角色、显示名、片段数、总时长、来源日期
  remove <名字> [--clip <clip>]... [--yes]                   删掉一个人（家人要加 --yes），或删掉他的某些片段
  rename <旧名字> <新名字> [--display <标签>]
  set <名字> [--role ...] [--display ...] [--threshold ...]
  check                                                     片段相似度检查，找出更像别人的片段
  migrate [--dry-run] [--local]                             把旧写法的片段（引用整段录音）裁成 clip 并改写 speakers.json
  sync                                                      本机与 GPU 主机（gpu.remote）互相补齐 voiceprints/clips/
<录音> 写录音名即可，例如 20261103_081500（缺省扩展名取 recorder.extensions 的第一个）。
--local：storage 为 remote 时也不连 GPU 主机，用本机录音裁 clip。`;

const VALUE_FLAGS = ['--role', '--display', '--threshold', '--max', '--min', '--clip', '--out', '--file', '--cluster'];
const BOOL_FLAGS = ['--json', '--text', '--dry-run', '--whole', '--local', '--yes', '--all-speakers'];
const ARITY = { clusters: 1, add: 4, 'add-cluster': 4, list: 0, remove: 1, rename: 2, set: 1, check: 0, migrate: 0, sync: 0 };

/** 解析参数：{ cmd, args: [位置参数], opts: { role, display, ..., clip: [] } }。不合法时抛出用法。 */
export function parseVoiceArgs(argv) {
  const [cmd, ...rest] = argv;
  if (!(cmd in ARITY)) throw new Error(USAGE);
  const args = [];
  const opts = { clip: [] };
  for (let i = 0; i < rest.length; i += 1) {
    const a = rest[i];
    if (VALUE_FLAGS.includes(a)) {
      const v = rest[i + 1];
      if (v === undefined || v.startsWith('--')) throw new Error(`${a} 缺少值\n${USAGE}`);
      const k = a.slice(2);
      if (k === 'clip') opts.clip.push(v); else opts[k] = v;
      i += 1;
    } else if (BOOL_FLAGS.includes(a)) opts[a.slice(2).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = true;
    else if (a.startsWith('--')) throw new Error(`未知选项 ${a}\n${USAGE}`);
    else args.push(a);
  }
  if (args.length !== ARITY[cmd]) throw new Error(`${cmd} 需要 ${ARITY[cmd]} 个参数\n${USAGE}`);
  if (opts.role !== undefined && !ROLES.includes(opts.role)) throw new Error('--role 只能是 family 或 known');
  for (const k of ['threshold', 'max', 'min']) if (opts[k] !== undefined && !(Number(opts[k]) > 0)) throw new Error(`--${k} 应为正数`);
  if (opts.max !== undefined && !Number.isInteger(Number(opts.max))) throw new Error('--max 应为整数');
  if (opts.display !== undefined && (!opts.display || /\s/.test(opts.display))) throw new Error('--display 不能为空或含空白');
  if ([opts.cluster !== undefined, opts.allSpeakers, opts.whole].filter(Boolean).length > 1) throw new Error('--cluster、--all-speakers、--whole 只能选一个');
  return { cmd, args, opts };
}

/** 录音参数 → { base, source: raw/<日期>/<文件名>, day, clock }。文件名须符合 recorder.pattern。 */
export function resolveRecording(arg, config) {
  const name = basename(arg);
  const ext = extname(name);
  const base = ext ? basename(name, ext) : name;
  const info = recordingInfo(base, config);
  if (!info) throw new Error(`录音文件名应符合 recorder.pattern（${config.recorder.pattern}）：${base}`);
  return { base, ...info, source: `raw/${info.day}/${base}${ext || config.recorder.extensions[0]}` };
}

const readJson = (p) => readFile(p, 'utf8').then(JSON.parse, () => null);

/** 读一个录音的 GPU 方案缓存。缺了就报错，提示先转写。 */
export async function loadGpuData(root, day, base) {
  const [diar, embed] = await Promise.all(['diar', 'embed'].map((k) => readJson(join(root, 'cache', day, `gpu-${base}.${k}.json`))));
  if (!diar || !embed) throw new Error(`cache/${day}/ 下没有 ${base} 的 GPU 方案缓存（gpu-${base}.diar.json / .embed.json），先用 GPU 方案跑 npm run transcribe -- ${day}`);
  return { spans: diar.spans, pieces: embed.pieces, pieceEmbs: embed.piece_embs, excl: embed.excl, exclEmbs: embed.excl_embs };
}

async function gpuBases(root, day) {
  return (await readdir(join(root, 'cache', day)).catch(() => []))
    .map((f) => /^gpu-(.+)\.embed\.json$/.exec(f)?.[1]).filter(Boolean).sort();
}

/** 认人用的声纹：有 GPU 方案算的同一 speakers.json 的声纹就用它（与转写稿一致），否则本机 CAM++ 现算。 */
async function peopleFor(root, config) {
  const gpu = await loadGpuVoiceprints(root);
  if (gpu) return { people: gpu, source: 'GPU 方案' };
  configureInference(config.numThreads);
  const paths = await requireModels(config.modelDir);
  return { people: await loadVoiceprints(root, paths), source: '本机' };
}

/** 改完 speakers.json：本机声纹缓存重算，并提示只重算认人。 */
async function afterChange(root, config, out) {
  try {
    configureInference(config.numThreads);
    const paths = await requireModels(config.modelDir);
    await loadVoiceprints(root, paths);
    out.voiceprints = '本机声纹缓存已重算';
  } catch (e) {
    out.voiceprints = `本机声纹缓存没有重算：${e.message.split('\n')[0]}`;
  }
  out.hint = 'npm run transcribe -- <日期> --relabel-only';
  return out;
}

/** storage 为 remote 时录音在 GPU 主机上，连得上就在那里裁 clip；否则在本机裁。 */
async function openRemote(config, opts) {
  if (opts.local || opts.dryRun || config.storage !== 'remote') return null;
  try { return await connectRemote(config.gpu.remote); } catch { return null; }
}

/** 把挑好的片段登记给某人：裁 clip、改 speakers.json。entries: [{ source, start, end }]。 */
async function enroll(root, config, name, entries, opts) {
  // 还没有 speakers.json（第一次登记）时从空名单开始
  const { speakers } = existsSync(join(root, 'speakers.json')) ? await loadSpeakers(root) : { speakers: [] };
  const items = entries.map((e) => clipEntry(e.source, e.start, e.end));
  const owner = new Map(speakers.flatMap((s) => s.enroll.filter((e) => e.clip).map((e) => [e.clip, s.name])));
  for (const it of items) {
    const who = owner.get(it.clip);
    if (who && who !== name) throw new Error(`片段 ${it.clip} 已登记给 ${who}，先从 ${who} 那里移除`);
  }
  let person = speakers.find((s) => s.name === name);
  const fresh = items.filter((it) => !person?.enroll.some((e) => e.clip === it.clip));
  const out = { name, added: fresh.length, skipped: items.length - fresh.length, seconds: +fresh.reduce((a, e) => a + e.end - e.start, 0).toFixed(1),
    clips: fresh.map((e) => ({ clip: e.clip, at: `[${formatTimestamp(e.start)}]`, seconds: +(e.end - e.start).toFixed(2) })) };
  if (opts.dryRun || fresh.length === 0) return { ...out, dryRun: !!opts.dryRun };
  const conn = await openRemote(config, opts);
  const cut = await cutClips(root, fresh, { conn, remote: config.gpu.remote });
  out.cutOn = cut.where;
  if (!person) {
    person = { name, role: opts.role ?? 'known', display: opts.display ?? name, threshold: opts.threshold !== undefined ? Number(opts.threshold) : DEFAULT_THRESHOLD, enroll: [] };
    speakers.push(person);
    out.newPerson = true;
  } else {
    if (opts.role) person.role = opts.role;
    if (opts.display) person.display = opts.display;
    if (opts.threshold !== undefined) person.threshold = Number(opts.threshold);
  }
  person.enroll = [...person.enroll, ...fresh];
  await writeSpeakers(root, speakers);
  return afterChange(root, config, out);
}

const r1 = (x) => Math.round(x * 10) / 10;

export const commands = {
  async clusters(root, config, [dayArg], opts) {
    const day = resolveDay(dayArg);
    const bases = opts.file ? [resolveRecording(opts.file, config).base] : await gpuBases(root, day);
    if (bases.length === 0) throw new Error(`cache/${day}/ 下没有 GPU 方案缓存，先用 GPU 方案跑 npm run transcribe -- ${day}`);
    const { people, source } = await peopleFor(root, config);
    const files = [];
    for (const base of bases) {
      const clock = recordingInfo(base, config)?.clock ?? 0;
      files.push({ file: base, clusters: clusterReport(await loadGpuData(root, day, base), people, { clock, withText: !!opts.text }) });
    }
    const out = { day, voiceprints: source, note: '钟点按录音文件名推算（不符合 recorder.pattern 的从 0 点算），与实际时间可能有偏差', files };
    if (opts.out) {
      await mkdir(dirname(join(root, opts.out)), { recursive: true });
      await writeFile(join(root, opts.out), `${JSON.stringify(out, null, 2)}\n`, { mode: 0o600 });
      out.written = opts.out;
    }
    return out;
  },

  async add(root, config, [name, rec, a, b], opts) {
    const r = resolveRecording(rec, config);
    const start = parseTime(a);
    const end = parseTime(b);
    if (!(end > start)) throw new Error('结束时间应晚于开始时间');
    if (opts.whole) {
      if (end - start > 120) throw new Error('--whole 整段登记最长 120 秒');
      return enroll(root, config, name, [{ source: r.source, start, end }], opts);
    }
    const data = await loadGpuData(root, r.day, r.base);
    const { picked, bySpeaker, speaker } = pickRange(data.pieces, data.spans, start, end,
      { minSec: opts.min !== undefined ? Number(opts.min) : 1.5, cluster: opts.cluster, allSpeakers: !!opts.allSpeakers });
    if (picked.length === 0) throw new Error(`[${formatTimestamp(start)}]–[${formatTimestamp(end)}] 里没有干净的单人句（≥${opts.min ?? 1.5} 秒）；确认是自己专门录的整段可加 --whole`);
    const out = await enroll(root, config, name, picked.map((i) => ({ source: r.source, start: data.pieces[i].start, end: data.pieces[i].end })), opts);
    return { ...out, cluster: speaker, rangeClusters: bySpeaker.map((g) => ({ cluster: g.speaker, sentences: g.count, seconds: r1(g.seconds) })) };
  },

  async 'add-cluster'(root, config, [name, dayArg, rec, spk], opts) {
    const r = resolveRecording(rec, config);
    if (resolveDay(dayArg) !== r.day) throw new Error(`录音 ${r.base} 不是 ${dayArg} 的`);
    const data = await loadGpuData(root, r.day, r.base);
    const cent = speakerCentroids(data.excl, data.exclEmbs).get(spk);
    if (!cent) throw new Error(`${r.base} 没有簇 ${spk}（或这个簇没有可用声纹）；簇号见 npm run voice -- clusters ${r.day}`);
    const picked = pickCluster(data.pieces, data.pieceEmbs, data.spans, spk, cent, { max: opts.max !== undefined ? Number(opts.max) : 20 });
    if (picked.length === 0) throw new Error(`簇 ${spk} 没有干净的单人句（1.5–10 秒）`);
    const out = await enroll(root, config, name, picked.map(({ i }) => ({ source: r.source, start: data.pieces[i].start, end: data.pieces[i].end })), opts);
    return { ...out, cluster: spk, minScore: Math.round(Math.min(...picked.map((p) => p.score)) * 1000) / 1000 };
  },

  async list(root) {
    const { speakers, hash } = await loadSpeakers(root);
    return {
      hash,
      people: speakers.map((s) => ({
        name: s.name, role: s.role, display: s.display, threshold: s.threshold, clips: s.enroll.length,
        seconds: r1(s.enroll.reduce((a, e) => a + enrollSeconds(e), 0)),
        oldFormat: s.enroll.filter((e) => e.clip === undefined).length,
        days: [...new Set(s.enroll.map((e) => enrollSource(e).day).filter(Boolean))].sort(),
      })),
    };
  },

  async remove(root, config, [name], opts) {
    const { speakers } = await loadSpeakers(root);
    const person = speakers.find((s) => s.name === name);
    if (!person) throw new Error(`speakers.json 里没有 ${name}`);
    let out;
    if (opts.clip.length === 0) {
      if (person.role === 'family' && !opts.yes) throw new Error(`${name} 是家人，确认要删掉整个人请加 --yes`);
      if (speakers.length === 1) throw new Error('不能删掉最后一个人');
      speakers.splice(speakers.indexOf(person), 1);
      out = { name, removedPerson: true, removedClips: person.enroll.length };
    } else {
      const match = (e, c) => (e.clip ?? '') === c || basename(e.clip ?? '') === c || basename(e.clip ?? '', '.wav') === c;
      for (const c of opts.clip) if (!person.enroll.some((e) => match(e, c))) throw new Error(`${name} 没有片段 ${c}`);
      const keep = person.enroll.filter((e) => !opts.clip.some((c) => match(e, c)));
      if (keep.length === 0) throw new Error(`删完 ${name} 就没有片段了；要删整个人用 remove ${name}`);
      out = { name, removedClips: person.enroll.length - keep.length, left: keep.length };
      person.enroll = keep;
    }
    await writeSpeakers(root, speakers);
    // clip 文件不删：可能还要恢复；没人引用的 clip 可以手动删。
    const used = new Set(clipReferences(speakers));
    out.unreferencedClips = (await readdir(join(root, CLIP_DIR)).catch(() => [])).filter((f) => f.endsWith('.wav') && !used.has(`${CLIP_DIR}/${f}`)).length;
    return afterChange(root, config, out);
  },

  async rename(root, config, [from, to], opts) {
    const { speakers } = await loadSpeakers(root);
    const person = speakers.find((s) => s.name === from);
    if (!person) throw new Error(`speakers.json 里没有 ${from}`);
    if (speakers.some((s) => s.name === to)) throw new Error(`已经有 ${to}`);
    person.name = to;
    if (opts.display) person.display = opts.display;
    else if (person.display === from) person.display = to;
    await writeSpeakers(root, speakers);
    return afterChange(root, config, { from, to, display: person.display });
  },

  async set(root, config, [name], opts) {
    const { speakers } = await loadSpeakers(root);
    const person = speakers.find((s) => s.name === name);
    if (!person) throw new Error(`speakers.json 里没有 ${name}`);
    if (opts.role === undefined && opts.display === undefined && opts.threshold === undefined) throw new Error('set 至少要改 --role、--display、--threshold 之一');
    if (opts.role) person.role = opts.role;
    if (opts.display) person.display = opts.display;
    if (opts.threshold !== undefined) person.threshold = Number(opts.threshold);
    await writeSpeakers(root, speakers);
    return afterChange(root, config, { name, role: person.role, display: person.display, threshold: person.threshold });
  },

  async check(root, config) {
    const { speakers } = await loadSpeakers(root);
    configureInference(config.numThreads);
    const paths = await requireModels(config.modelDir);
    const embs = await embedEnrollments(root, speakers, paths);
    const id = (e) => (e.clip ? basename(e.clip, '.wav') : `${basename(e.file)}@${e.start}`);
    const people = [...embs].map(([name, xs]) => ({ name, clips: xs.map((x) => ({ id: id(x.entry), vec: x.vec })) }));
    return { people: checkEnrollment(people) };
  },

  async migrate(root, config, _args, opts) {
    const { speakers } = await loadSpeakers(root);
    const items = [];
    for (const s of speakers)
      s.enroll = s.enroll.map((e) => {
        if (e.clip !== undefined) return e;
        const c = clipEntry(e.file, e.start, e.end);
        items.push(c);
        return c;
      });
    const out = { migrated: items.length, people: speakers.map((s) => ({ name: s.name, clips: s.enroll.length })) };
    if (opts.dryRun) return { ...out, dryRun: true };
    if (items.length) {
      const conn = await openRemote(config, opts);
      out.cutOn = (await cutClips(root, items, { conn, remote: config.gpu.remote })).where;
    }
    // 补上 role / display 的缺省值，统一成新格式（迁移不删任何录音）。
    await writeSpeakers(root, speakers);
    return afterChange(root, config, out);
  },

  async sync(root, config) {
    if (!config.gpu.remote) throw new Error('sync 用于远程 GPU 方案：dairy.config.json 没配 gpu.remote');
    const conn = await connectRemote(config.gpu.remote);
    await pushClips(root, { remote: config.gpu.remote, call: conn.call });
    await pullClips(root, { remote: config.gpu.remote, call: conn.call });
    const { speakers } = await loadSpeakers(root);
    const have = new Set((await readdir(join(root, CLIP_DIR)).catch(() => [])).map((f) => `${CLIP_DIR}/${f}`));
    return { clips: have.size, missing: clipReferences(speakers).filter((c) => !have.has(c)) };
  },
};

/** 人看的简短输出。 */
export function summarize(cmd, out) {
  const L = [];
  const pct = (o) => Object.entries(o).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(' ');
  if (cmd === 'clusters') {
    L.push(`${out.day}（声纹：${out.voiceprints}；${out.note}）`);
    for (const f of out.files) {
      L.push(`${f.file}`);
      for (const c of f.clusters) {
        L.push(`  簇 ${c.speaker}：${c.seconds} 秒（单人 ${c.soloSeconds} 秒），${c.sentences} 句，认成 ${c.label ?? '—'}；行标签 ${pct(c.lineLabels) || '—'}`);
        L.push(`    最像：${c.top.map((t) => `${t.name} ${t.score}`).join('，') || '—'}`);
        L.push(`    分布：${c.hours.map((h) => `${h.hour} 点 ${h.seconds}s`).join('，')}`);
        const withText = c.samples.some((s) => s.text !== undefined);
        L.push(`    代表句：${c.samples.map((s) => `${s.at}${withText ? ` ${s.text}` : ''}`).join(withText ? '\n      ' : ' ')}`);
      }
    }
    if (out.written) L.push(`已写入 ${out.written}`);
  } else if (cmd === 'list') {
    for (const p of out.people)
      L.push(`${p.name}（${p.role}，显示为 ${p.display}，阈值 ${p.threshold}）：${p.clips} 个片段 ${p.seconds} 秒，来源 ${p.days.join(' ') || '—'}${p.oldFormat ? `；${p.oldFormat} 个旧写法片段，可跑 migrate` : ''}`);
  } else if (cmd === 'check') {
    for (const p of out.people) {
      L.push(`${p.name}：${p.clips} 个片段，两两相似度均值 ${p.pairMean ?? '—'}、最小 ${p.pairMin ?? '—'}；最接近 ${p.nearest ? `${p.nearest.name} ${p.nearest.score}` : '—'}`);
      for (const a of p.advice) L.push(`  - ${a}`);
    }
  } else {
    if (out.dryRun) L.push('（--dry-run，未改动）');
    if (cmd === 'add' || cmd === 'add-cluster') {
      if (out.rangeClusters?.length > 1) L.push(`这段里有 ${out.rangeClusters.length} 个簇：${out.rangeClusters.map((g) => `簇 ${g.cluster} ${g.sentences} 句 ${g.seconds}s`).join('，')}；取了簇 ${out.cluster ?? '全部'}`);
      L.push(`${out.name}${out.newPerson ? '（新登记）' : ''}：加 ${out.added} 个片段共 ${out.seconds} 秒${out.skipped ? `，${out.skipped} 个已登记过` : ''}${out.cutOn ? `，在${out.cutOn}裁剪` : ''}`);
      if (out.clips.length) L.push(`  ${out.clips.map((c) => `${c.at} ${c.seconds}s`).join('，')}`);
    } else L.push(JSON.stringify(Object.fromEntries(Object.entries(out).filter(([k]) => !['voiceprints', 'hint'].includes(k)))));
    if (out.voiceprints) L.push(out.voiceprints);
    if (out.hint && !out.dryRun) L.push(`只重算认人：${out.hint}`);
  }
  return L.join('\n');
}

export async function runVoice(argv, { root = ROOT, log = console.log } = {}) {
  const { cmd, args, opts } = parseVoiceArgs(argv);
  const config = await loadConfig(root);
  setUnknownLabel(config.unknownLabel);
  const out = await commands[cmd](root, config, args, opts);
  log(opts.json ? JSON.stringify(out) : summarize(cmd, out));
  return out;
}

if (process.argv[1] && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href)
  runVoice(process.argv.slice(2)).catch((e) => {
    console.error(e.message);
    process.exitCode = 1;
  });
