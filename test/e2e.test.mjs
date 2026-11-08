// 端到端（CPU 方案）：对一段真实录音跑解码 → VAD+ASR → 分离 → embedding → 标注 → 渲染。
// 需要本机模型（npm run models）、dairy.config.json 与 speakers.json；没设 RYD_E2E_AUDIO 时跳过。
//   RYD_E2E_AUDIO=<录音路径>                         必填，要转写的样本
//   RYD_E2E_EXPECT=<秒>:<标签>[,<秒>:<标签>...]      可选，核对始于这些秒的句子的标签，例如 28:我,120:我
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { ROOT, loadConfig } from '../src/config.mjs';
import { configureInference } from '../src/inference.mjs';
import { setUnknownLabel } from '../src/label.mjs';
import { requireModels } from '../src/models.mjs';
import { transcribeFile } from '../src/pipeline.mjs';
import { loadSpeakers } from '../src/speakers.mjs';
import { loadVoiceprints } from '../src/voiceprint.mjs';

const AUDIO = process.env.RYD_E2E_AUDIO;
const EXPECT = (process.env.RYD_E2E_EXPECT ?? '').split(',').filter(Boolean).map((x) => {
  const i = x.indexOf(':');
  return { sec: Number(x.slice(0, i)), label: x.slice(i + 1) };
});

test('端到端：样本录音转写成合规的转写稿，指定位置的句子标签正确', { skip: !AUDIO && '没设 RYD_E2E_AUDIO', timeout: 20 * 60 * 1000 }, async () => {
  const tmp = await mkdtemp(join(tmpdir(), 'ryd-e2e-'));
  try {
    const config = await loadConfig();
    setUnknownLabel(config.unknownLabel);
    configureInference(config.numThreads);
    const paths = await requireModels(config.modelDir);
    const people = await loadVoiceprints(ROOT, paths);
    const outPath = join(tmp, 'out.md');
    const labeled = await transcribeFile(resolve(AUDIO), { cacheDir: join(tmp, 'cache'), outPath, paths, people, force: true });
    const lines = (await readFile(outPath, 'utf8')).trimEnd().split('\n');
    const labels = new Set([...(await loadSpeakers(ROOT)).speakers.map((s) => s.display), config.unknownLabel]);
    for (const line of lines) {
      const m = /^\[(\d+:)?\d\d:\d\d\] (\S+)/.exec(line);
      assert.ok(line === '<no speech detected>' || (m && (labels.has(m[2]) || line.endsWith('<no speech detected in this segment>'))), '转写行格式无效（内容已隐藏）');
    }
    for (const { sec, label } of EXPECT) {
      const hit = labeled.filter((u) => Math.floor(u.start) === sec);
      assert.equal(hit.length, 1, `应有一句始于第 ${sec} 秒`);
      assert.equal(hit[0].label, label, `第 ${sec} 秒标签错误`);
      assert.ok(lines.some((l) => l.includes(` ${label} `) && l.includes(hit[0].text)), `第 ${sec} 秒渲染标签不符`);
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});
