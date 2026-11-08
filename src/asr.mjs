// silero VAD 分段 + zipformer transducer 逐段识别（贪心解码）。
import { SAMPLE_RATE } from './decode.mjs';
import { inferenceOptions } from './inference.mjs';
import { loadSherpa } from './sherpa.mjs';

const VAD_WINDOW = 512;
const VAD_CONFIG = {
  threshold: 0.5,
  minSilenceDuration: 0.35,
  minSpeechDuration: 0.25,
  maxSpeechDuration: 20,
  windowSize: VAD_WINDOW,
};

/** 去掉模型输出的特殊 token（<|xx|>、<unk> 之类），保留正文里的普通比较符。 */
export function stripSpecialTokens(text) {
  return text
    .replace(/<\s*\|[^>]*\|\s*>/g, '')
    .replace(/<\s*[A-Za-z_][A-Za-z0-9_|-]*\s*>/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

let recognizer;
async function getRecognizer(paths) {
  if (recognizer) return recognizer;
  const sherpa = await loadSherpa();
  recognizer = new sherpa.OfflineRecognizer({
    featConfig: { sampleRate: SAMPLE_RATE, featureDim: 80 },
    modelConfig: {
      transducer: { encoder: paths.asr.encoder, decoder: paths.asr.decoder, joiner: paths.asr.joiner },
      tokens: paths.asr.tokens,
      ...inferenceOptions(),
      provider: 'cpu',
      debug: 0,
    },
  });
  return recognizer;
}

/**
 * 先 VAD 分段再逐段识别（整段送模型会解码退化）。返回 [{ start, end, text }]（秒），丢弃识别为空的段。
 * paths 为 models.mjs 的 modelPaths 结果。
 */
export async function transcribe(samples, paths) {
  const rec = await getRecognizer(paths);
  const sherpa = await loadSherpa();
  const vad = new sherpa.Vad(
    { sileroVad: { model: paths.vad.model, ...VAD_CONFIG }, sampleRate: SAMPLE_RATE, numThreads: 1, debug: 0 },
    60
  );
  const segments = [];
  const drain = () => {
    while (!vad.isEmpty()) {
      const s = vad.front();
      segments.push({ start: s.start / SAMPLE_RATE, end: (s.start + s.samples.length) / SAMPLE_RATE, samples: s.samples });
      vad.pop();
    }
  };
  for (let i = 0; i + VAD_WINDOW <= samples.length; i += VAD_WINDOW) {
    vad.acceptWaveform(samples.subarray(i, i + VAD_WINDOW));
    drain();
  }
  vad.flush();
  drain();
  const out = [];
  for (const seg of segments) {
    const stream = rec.createStream();
    stream.acceptWaveform({ samples: seg.samples, sampleRate: SAMPLE_RATE });
    rec.decode(stream);
    const text = stripSpecialTokens(rec.getResult(stream).text ?? '');
    if (text) out.push({ start: seg.start, end: seg.end, text });
  }
  return out;
}
