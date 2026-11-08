// CAM++ 说话人 embedding 与向量运算。
import { SAMPLE_RATE } from './decode.mjs';
import { inferenceOptions } from './inference.mjs';
import { loadSherpa } from './sherpa.mjs';

let extractor;
async function getExtractor(paths) {
  if (extractor) return extractor;
  const sherpa = await loadSherpa();
  extractor = new sherpa.SpeakerEmbeddingExtractor({ model: paths['speaker-embed'].model, ...inferenceOptions(), provider: 'cpu', debug: 0 });
  return extractor;
}

/** 对 samples 的 [start, end) 秒区间算 embedding，返回 L2 归一化、保留 6 位小数的数组；音频过短算不出时返回 null。 */
export async function embedRange(samples, start, end, paths) {
  const ex = await getExtractor(paths);
  const stream = ex.createStream();
  stream.acceptWaveform({ samples: samples.subarray(Math.round(start * SAMPLE_RATE), Math.round(end * SAMPLE_RATE)), sampleRate: SAMPLE_RATE });
  stream.inputFinished();
  if (!ex.isReady(stream)) return null;
  return l2(Array.from(ex.compute(stream))).map((x) => +x.toFixed(6));
}

export function l2(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

/** 两个已归一化向量的余弦。 */
export function cos(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

/** 已归一化向量按权重（缺省等权）平均后再归一化。 */
export function meanEmb(vs, ws) {
  const out = new Array(vs[0].length).fill(0);
  vs.forEach((v, k) => {
    const w = ws ? ws[k] : 1;
    for (let i = 0; i < out.length; i += 1) out[i] += w * v[i];
  });
  return l2(out);
}
