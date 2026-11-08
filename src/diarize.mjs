// 说话人分离（sherpa-onnx OfflineSpeakerDiarization：pyannote 分割 + CAM++ 聚类）与句子归簇。
import { inferenceOptions } from './inference.mjs';
import { loadSherpa } from './sherpa.mjs';

const MIN_OVERLAP_RATIO = 0.5;

/** 返回 [{ start, end, speaker }]，speaker 是簇编号（不稳定，只在本文件内有意义）。 */
export async function diarize(samples, paths) {
  const sherpa = await loadSherpa();
  const sd = new sherpa.OfflineSpeakerDiarization({
    segmentation: { pyannote: { model: paths['speaker-seg'].model }, ...inferenceOptions(), provider: 'cpu', debug: 0 },
    embedding: { model: paths['speaker-embed'].model, ...inferenceOptions(), provider: 'cpu', debug: 0 },
    clustering: { numClusters: -1, threshold: 0.7 },
    minDurationOn: 0.3,
    minDurationOff: 0.5,
  });
  return sd.process(samples);
}

/** 每句取重叠最长的 span 的簇；重叠不足句长一半时为 null。 */
export function assignSpeakers(utts, spans) {
  return utts.map((u) => {
    const dur = u.end - u.start;
    let best;
    let bestOverlap = 0;
    for (const s of spans) {
      const overlap = Math.min(u.end, s.end) - Math.max(u.start, s.start);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = s;
      }
    }
    return best && dur > 0 && bestOverlap >= MIN_OVERLAP_RATIO * dur ? best.speaker : null;
  });
}
