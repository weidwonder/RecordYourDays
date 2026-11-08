"""在 DiariZen 子 venv 里运行：整文件说话人分离（允许重叠），结果写 JSON。

用法：python -m dairy_engine.diarizen_run --audio <f32 裸数据> --out <json> --model <目录> --embedding <bin> --batch 32
音频以内存 WAV（PCM 16 位）交给 DiariZenPipeline，不落盘。输出 {"spans": [[start, end, speaker], ...]}，speaker 为匿名编号。
"""
import argparse
import io
import json
import os
from pathlib import Path

from .stages import load_f32, lock_down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    lock_down()
    os.umask(0o077)
    import soundfile as sf
    import toml
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 不可用")
    model = Path(a.model)
    config = toml.load(model / "config.toml")
    config["inference"]["args"]["batch_size"] = a.batch
    pipeline = DiariZenPipeline(diarizen_hub=model, embedding_model=a.embedding, config_parse=config, rttm_out_dir=None)
    buf = io.BytesIO()
    sf.write(buf, load_f32(a.audio), 16000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    annotation = pipeline(buf)
    spans = [[round(float(t.start), 3), round(float(t.end), 3), str(spk)] for t, _, spk in annotation.itertracks(yield_label=True)]
    tmp = a.out + ".part"
    Path(tmp).write_text(json.dumps({"spans": spans}))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
