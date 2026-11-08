// 音频文件 → 16 kHz 单声道 Float32Array。经 audio-decode 解码，不依赖外部 ffmpeg。
import { readFile } from 'node:fs/promises';
import decode from 'audio-decode';
import { loadSherpa } from './sherpa.mjs';

export const SAMPLE_RATE = 16000;

export async function decodeToMono16k(filePath) {
  const raw = await decode(await readFile(filePath));
  const channels = raw?.channelData;
  if (!Array.isArray(channels) || !channels[0]?.length) throw new Error(`解码结果为空：${filePath}`);
  const frames = channels[0].length;
  let mono;
  if (channels.length === 1) mono = Float32Array.from(channels[0]);
  else {
    mono = new Float32Array(frames);
    for (let i = 0; i < frames; i += 1) {
      let sum = 0;
      for (const ch of channels) sum += ch[i] ?? 0;
      mono[i] = sum / channels.length;
    }
  }
  if (raw.sampleRate === SAMPLE_RATE) return mono;
  const { LinearResampler } = await loadSherpa();
  return new LinearResampler(raw.sampleRate, SAMPLE_RATE, 8000, true).resample(mono, true);
}
