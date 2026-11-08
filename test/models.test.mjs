// CPU 方案模型下载：用本机造的压缩包替代官方地址，不连网络。
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { COMPONENTS, downloadModels, modelPaths, requireModels } from '../src/models.mjs';
import { tmp } from './helpers.mjs';

/** 每个组件的下载内容：压缩包按 archiveDir/<文件> 打包，单个文件直接给内容。 */
async function payloads(dir) {
  const out = new Map();
  for (const [id, c] of Object.entries(COMPONENTS)) {
    if (!c.archiveDir) { out.set(c.url, Buffer.from(`${id} 模型`)); continue; }
    const src = join(dir, `src-${id}`);
    await mkdir(join(src, c.archiveDir), { recursive: true });
    for (const f of [...Object.values(c.files), 'README.md']) await writeFile(join(src, c.archiveDir, f), `${id}/${f}`);
    const tar = join(dir, `${id}.tar.bz2`);
    assert.equal(spawnSync('tar', ['-cjf', tar, '-C', src, c.archiveDir]).status, 0);
    out.set(c.url, await readFile(tar));
  }
  return out;
}

const fakeFetch = (bodies, seen, { truncate } = {}) => async (url) => {
  seen.push(url);
  const body = bodies.get(url);
  if (!body) return new Response('not found', { status: 404 });
  const sent = truncate === url ? body.subarray(0, 3) : body;
  return new Response(sent, { headers: { 'content-length': String(body.length) } });
};

test('downloadModels：从官方地址下载、解压到 <modelDir>/<id>/<version>/，已有的跳过，不完整的不留文件', { skip: spawnSync('tar', ['--version']).status !== 0 && '没有 tar' }, async () => {
  const dir = await tmp('ryd-models-');
  try {
    const bodies = await payloads(dir);
    for (const url of bodies.keys()) assert.match(url, /^https:\/\/github\.com\/k2-fsa\/sherpa-onnx\/releases\/download\//);
    const modelDir = join(dir, 'models');
    await assert.rejects(requireModels(modelDir), /npm run models/);
    const seen = [];
    const r = await downloadModels(modelDir, { fetchFn: fakeFetch(bodies, seen), log: () => {} });
    assert.deepEqual(r, { downloaded: Object.keys(COMPONENTS), skipped: [] });
    const paths = await requireModels(modelDir);
    assert.deepEqual(paths, modelPaths(modelDir));
    assert.equal(await readFile(paths.asr.encoder, 'utf8'), 'asr/encoder-epoch-99-avg-1.int8.onnx');
    assert.equal(await readFile(paths.vad.model, 'utf8'), 'vad 模型');
    assert.deepEqual((await readdir(join(modelDir, 'asr', COMPONENTS.asr.version))).sort(), Object.values(COMPONENTS.asr.files).sort(), '只留需要的文件');
    assert.ok(!(await readdir(modelDir)).some((f) => f.startsWith('.download-')), '临时目录已清理');
    seen.length = 0;
    assert.deepEqual((await downloadModels(modelDir, { fetchFn: fakeFetch(bodies, seen), log: () => {} })).skipped, Object.keys(COMPONENTS));
    assert.equal(seen.length, 0);
    await rm(join(modelDir, 'vad'), { recursive: true });
    await assert.rejects(downloadModels(modelDir, { fetchFn: fakeFetch(bodies, seen, { truncate: COMPONENTS.vad.url }), log: () => {} }), /不完整/);
    await assert.rejects(requireModels(modelDir), /silero_vad\.onnx/);
    await assert.rejects(downloadModels(modelDir, { fetchFn: fakeFetch(new Map(), seen), log: () => {} }), /HTTP 404/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
