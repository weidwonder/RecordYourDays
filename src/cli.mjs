// 命令行入口：node src/cli.mjs <import|transcribe|run|models|setup-gpu> [YYYY-MM-DD] [选项]。用法见 USAGE。
import { realpathSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { loadConfig, normalizeEngine, ROOT } from './config.mjs';
import { resolveDay } from './dates.mjs';
import { execute } from './flow.mjs';
import { setupGpuLocal, transcribeGpuLocal } from './gpu.mjs';
import { configureInference } from './inference.mjs';
import { setUnknownLabel } from './label.mjs';
import { downloadModels, requireModels } from './models.mjs';
import { transcribeDay } from './pipeline.mjs';
import { relabelLocal } from './relabel.mjs';
import { connectRemote, deployRemote } from './remote.mjs';
import { loadVoiceprints } from './voiceprint.mjs';

const USAGE = `用法：
  node src/cli.mjs import [YYYY-MM-DD]                    录音笔 → raw/<日期>/（storage 为 remote 时直接存到 GPU 主机）
  node src/cli.mjs transcribe [YYYY-MM-DD] [--force|--relabel-only] [--engine cpu|gpu] [--no-fallback]
                                                          转写到 transcripts/<日期>/
  node src/cli.mjs run [YYYY-MM-DD] [--force] [--engine cpu|gpu] [--no-fallback] [--keep-recorder|--delete-recorder]
                                                          import + transcribe，按设置核对后删录音笔原件
  node src/cli.mjs models                                 下载 CPU 方案的模型到 modelDir（已有的跳过）
  node src/cli.mjs setup-gpu                              建 GPU 方案的 Python 环境并下载模型（本机或 gpu.remote）
日期缺省为 timeZone 的昨天。
  --engine：cpu 在本机跑（legacy 为旧名）；gpu 在 gpu.remote 或本机 GPU 上跑。缺省取 dairy.config.json 的 engine
  --force：忽略缓存重新转写；--relabel-only：只用缓存重算认人（改了 speakers.json 之后用），不转写
  --no-fallback：GPU 方案连不上或失败时报错，不退回本机 CPU 方案（缺省按 fallbackToCpu）
  --keep-recorder / --delete-recorder：本次保留 / 删除录音笔原件（缺省按 recorder.deleteAfterTranscribe）`;

const COMMANDS = ['import', 'transcribe', 'run', 'models', 'setup-gpu'];
const FLAGS = ['--force', '--no-fallback', '--keep-recorder', '--delete-recorder', '--relabel-only'];

/** 解析参数；--engine 取下一个参数或 --engine=值。参数不合法时抛出用法说明。 */
export function parseArgs(argv) {
  const [cmd, ...args] = argv;
  let engine;
  const rest = [];
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--engine') { engine = args[i + 1] ?? ''; i += 1; }
    else if (args[i].startsWith('--engine=')) engine = args[i].slice('--engine='.length);
    else rest.push(args[i]);
  }
  const has = (f) => rest.includes(f);
  const positional = rest.filter((a) => !FLAGS.includes(a));
  const bare = ['models', 'setup-gpu'].includes(cmd);
  if (!COMMANDS.includes(cmd) || positional.length > 1 || positional[0]?.startsWith('-') || (bare && rest.length + (engine === undefined ? 0 : 1) > 0)
    || (has('--force') && cmd === 'import') || (has('--no-fallback') && cmd === 'import')
    || ((has('--keep-recorder') || has('--delete-recorder')) && cmd !== 'run') || (has('--keep-recorder') && has('--delete-recorder'))
    || (has('--relabel-only') && (cmd !== 'transcribe' || has('--force')))
    || (engine !== undefined && (cmd === 'import' || !['cpu', 'gpu', 'legacy'].includes(engine))))
    throw new Error(USAGE);
  return { cmd, day: positional[0], engine: engine === undefined ? undefined : normalizeEngine(engine), force: has('--force'),
    noFallback: has('--no-fallback'), keepRecorder: has('--keep-recorder'), deleteRecorder: has('--delete-recorder'), relabelOnly: has('--relabel-only') };
}

async function setupGpu(config) {
  const remote = config.gpu.remote;
  if (!remote) {
    console.log(`在本机建 GPU 方案环境 → ${config.gpu.envDir}，模型 → ${config.gpu.modelRoot}`);
    await setupGpuLocal(ROOT, config);
    return;
  }
  console.log(`在 GPU 主机 ${remote.host} 上建环境 → ${remote.envDir}，模型 → ${remote.modelRoot}（耗时较长，日志在远程 ${remote.workDir}/logs/engine-install.log）`);
  const conn = await connectRemote(remote);
  try { await deployRemote(ROOT, remote, conn.call); }
  catch { throw new Error(`GPU 主机上建环境失败，详情见远程 ${remote.workDir}/logs/engine-install.log（本机不回显）`); }
  console.log('GPU 方案环境与模型已就绪');
}

async function main(argv) {
  const { cmd, engine, force, noFallback, keepRecorder, deleteRecorder, relabelOnly, ...a } = parseArgs(argv);
  const config = await loadConfig();
  setUnknownLabel(config.unknownLabel);
  if (cmd === 'models') {
    const r = await downloadModels(config.modelDir);
    console.log(`CPU 方案模型已就绪（新下载 ${r.downloaded.length} 个，已有 ${r.skipped.length} 个）：${config.modelDir}`);
    return;
  }
  if (cmd === 'setup-gpu') return setupGpu(config);
  const day = resolveDay(a.day, new Date(), config.timeZone);
  console.log(`日期 ${day}`);
  const loadPeople = async () => {
    configureInference(config.numThreads);
    const paths = await requireModels(config.modelDir);
    return { paths, people: await loadVoiceprints(ROOT, paths, console.log) };
  };
  await execute({ cmd, day, force, noFallback, keepRecorder, deleteRecorder, relabelOnly, engine, root: ROOT, config,
    runCpu: async () => {
      if (relabelOnly) return relabelLocal(day, ROOT, { engine, loadPeople: async () => (await loadPeople()).people });
      const { paths, people } = await loadPeople();
      await transcribeDay(day, ROOT, { paths, people, force });
      return 'cpu';
    },
    runGpu: ({ relabelOnly: r }) => transcribeGpuLocal(ROOT, config, day, { force, relabelOnly: r }) });
}

// 被测试导入时不执行。
if (process.argv[1] && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href)
  main(process.argv.slice(2)).catch((e) => {
    console.error(e.message);
    process.exitCode = 1;
  });
