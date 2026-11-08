// import / transcribe / run 的编排。
// 转写方案：cpu 在本机跑（src/ 的 Node 流程）；gpu 跑 engine/ 的 Python 流程，配了 gpu.remote 就在那台 GPU 主机上跑，
// 否则在本机（有 NVIDIA 显卡的 Linux）跑。GPU 方案连不上或失败时，fallbackToCpu 为真（缺省）就退回本机 CPU 方案。
// 录音存放：storage 为 local 时录音存本机 raw/；为 remote 时存 GPU 主机，连不上才暂存本机（见 storage.mjs）。
import { normalizeEngine, ENGINES } from './config.mjs';
import { importDay } from './importer.mjs';
import { NO_GPU_EXIT, connectRemote, transcribeRemote } from './remote.mjs';
import { cleanRecorder, hasLocalAudio, keepLocalSet, pullDay, syncLocalRaw, uploadRecorder } from './storage.mjs';

const HIDDEN = '（原始错误输出已隐藏以保护隐私）';

/** 选转写方案：--engine，否则 dairy.config.json 的 engine（缺省 cpu）。 */
export function resolveEngine({ engine, config }) {
  return normalizeEngine(engine ?? config.engine ?? 'cpu');
}

/**
 * 需要 GPU 主机时（storage 为 remote，或用远程 GPU 方案）连一次。连不上时：用 GPU 方案且不许退回就报错，否则返回 null 改在本机做。
 * engine 为 null 表示这条命令不转写（import）。
 */
async function openConnection(config, { engine, strict, connect, log }) {
  const remote = config.gpu.remote;
  if (!remote || (config.storage !== 'remote' && engine !== 'gpu')) return null;
  try { return await connect(remote); }
  catch {
    if (engine === 'gpu' && strict) throw new Error(`连不上 GPU 主机 ${remote.host}，且设为不退回 CPU 方案；请检查 gpu.remote 与 ssh 配置` + HIDDEN);
    const store = config.storage === 'remote' ? '录音暂存本机，下次连上时补传' : '';
    log(`连不上 GPU 主机 ${remote.host}${engine === 'gpu' ? '，改在本机用 CPU 方案转写' : ''}${store ? `；${store}` : ''}`);
    return null;
  }
}

const usedEngine = (x) => (ENGINES.includes(x) ? x : 'cpu');

/** 返回 { where: 'remote' | 'local', engine }：在哪转写、实际用的方案。 */
async function transcribe({ root, config, day, force, relabelOnly, engine, strict, conn, runCpu, runGpu, runRemote, log }) {
  const what = relabelOnly ? '只重算认人' : '转写';
  const remote = config.gpu.remote;
  if (engine === 'gpu' && (conn || !remote)) {
    let why;
    try {
      if (remote) {
        log(`在 GPU 主机 ${remote.host} 上${what}（GPU 方案）`);
        await runRemote(root, remote, day, force, conn.call, { relabelOnly, pushAudio: config.storage === 'local' });
        log(`远程${what}完成，transcripts/${day}/ 与 cache/${day}/ 已拉回`);
        return { where: 'remote', engine: 'gpu' };
      }
      log(`在本机${what}（GPU 方案）`);
      await runGpu({ relabelOnly });
      return { where: 'local', engine: 'gpu' };
    } catch (e) {
      const at = remote ? `GPU 主机 ${remote.host} 上` : '本机';
      // 本机的报错是自己写的提示（例如还没建环境），可以显示；远程的一律隐藏。
      why = e?.code === NO_GPU_EXIT ? `${at}没有空闲 GPU` : `${at}的 GPU 方案${what}失败${remote ? '' : `：${e.message}`}`;
    }
    const hidden = remote ? HIDDEN : '';
    if (strict) throw new Error(`${why}；设为不退回 CPU 方案${remote ? `。远程日志见 ${remote.workDir}/logs/transcribe.log` : ''}${hidden}`);
    log(relabelOnly ? `${why}，改在本机按缓存重算认人${hidden}` : `${why}，退回本机 CPU 方案转写${hidden}`);
  }
  if (relabelOnly) return { where: 'local', engine: usedEngine(await runCpu()) };
  if (!(await hasLocalAudio(root, day))) {
    if (!(conn && config.storage === 'remote'))
      throw new Error(`本机 raw/${day}/ 下没有音频${config.storage === 'remote' ? '；录音若已存到 GPU 主机，请在能连上时重试' : '；先跑 npm run import'}`);
    await pullDay(root, { remote, call: conn.call, day });
    log(`已从 GPU 主机临时取回 raw/${day}/ 供本机转写，转写后核对删除`);
  }
  return { where: 'local', engine: usedEngine(await runCpu()) };
}

/**
 * 执行一条命令。返回 { engine, transcribed, recorder }，供测试核对。
 * 依赖可注入：connect（连 GPU 主机，返回 { host, call }）、runCpu（本机 CPU 方案转写或只重算认人，返回实际用的方案）、
 * runGpu（本机 GPU 方案，参数 { relabelOnly }）、runRemote（远程 GPU 方案，参数同 transcribeRemote）。
 */
export async function execute({ cmd, day, force = false, keepRecorder = false, deleteRecorder = false, relabelOnly = false, noFallback = false, engine, root, config,
  connect = connectRemote, runCpu, runGpu, runRemote = transcribeRemote, log = console.log }) {
  if (relabelOnly && (force || cmd !== 'transcribe')) throw new Error('--relabel-only 只用于 transcribe，且不能与 --force 同用');
  if (keepRecorder && deleteRecorder) throw new Error('--keep-recorder 与 --delete-recorder 不能同时使用');
  const chosen = resolveEngine({ engine, config });
  const strict = noFallback || !config.fallbackToCpu;
  const remoteStore = config.storage === 'remote';
  const conn = await openConnection(config, { engine: cmd === 'import' ? null : chosen, strict, connect, log });
  const remote = config.gpu.remote;
  const keep = remoteStore ? await keepLocalSet(root, config) : null;
  const sync = (remove) => syncLocalRaw(root, { remote, call: conn.call, keep, remove, log });
  const out = { engine: chosen, transcribed: null, recorder: null };
  // 先补传本机暂存的录音：远程转写要用；删除等到转写结束，期间本机这份仍可供退回时转写。
  if (remoteStore && conn && cmd !== 'import') await sync(false);
  if (cmd !== 'transcribe') {
    if (remoteStore && conn) await uploadRecorder(day, config, { remote, call: conn.call, log });
    else await importDay(day, root, config, { log });
  }
  if (cmd !== 'import') {
    const r = await transcribe({ root, config, day, force, relabelOnly, engine: chosen, strict, conn, runCpu, runGpu, runRemote, log });
    out.transcribed = r.where;
    out.engine = r.engine;
  }
  if (remoteStore && conn) await sync(true);
  if (cmd === 'run') {
    if (deleteRecorder || (config.recorder.deleteAfterTranscribe && !keepRecorder))
      out.recorder = await cleanRecorder(day, root, config, { remote, call: remoteStore ? conn?.call : undefined, log });
    else log(keepRecorder ? '--keep-recorder：保留录音笔原件' : '保留录音笔原件（要删请开 recorder.deleteAfterTranscribe 或加 --delete-recorder）');
  }
  return out;
}
