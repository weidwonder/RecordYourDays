#!/usr/bin/env bash
# 建好 GPU 方案的 Python 环境并下载缺的模型，可重复执行：锁文件与本脚本都没变时跳过建环境，已有的模型跳过下载。
# 用法：bash engine/setup_gpu.sh [环境目录]（缺省 ~/.cache/record-your-days/engine-env），在项目根目录执行（模型目录读 dairy.config.json）。
# 要求：Linux、NVIDIA 驱动（支持 CUDA 12）、python3（3.10，可用 DAIRY_PYTHON 指定解释器）、ffmpeg、git。
# 不用 sudo、不装系统 CUDA：CUDA/cuDNN 运行库来自 torch 的 pip wheel。
# 环境变量：DAIRY_PIP_INDEX 改用 pip 镜像（缺省官方 PyPI）；DAIRY_DIARIZEN_REPO 改用 DiariZen 源码的其他来源；
#   DAIRY_ENGINE_MODEL_ROOT 覆盖模型目录；DAIRY_SKIP_MODELS=1 只建环境不下载模型。
#   <环境目录>/venv            主环境：FunASR（Paraformer、FSMN VAD、CT 标点）、MossFormer2、sherpa-onnx CAM++
#   <环境目录>/diarizen-venv   DiariZen 子环境：经 .pth 只读复用主环境，另装 DiariZen 的 pyannote fork
#   <环境目录>/DiariZen        DiariZen 源码，固定 commit
#   <环境目录>/status.json     最近一次建好的时间、键、版本
set -euo pipefail
umask 077
[[ $(uname -s) == Linux ]] || { echo 'GPU 方案只支持 Linux（需要 NVIDIA 显卡）' >&2; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
env_dir=${1:-$HOME/.cache/record-your-days/engine-env}
py=${DAIRY_PYTHON:-python3}
diarizen_commit=844f5555b0a98acd0931511fc641a8c5b8ba92c7
pip_index=(${DAIRY_PIP_INDEX:+--index-url "$DAIRY_PIP_INDEX"})
key=$(cat "$here/requirements-lock.txt" "$here/diarizen-requirements-lock.txt" "$here/setup_gpu.sh" | sha256sum | cut -d ' ' -f 1)

download_models() {
  [[ ${DAIRY_SKIP_MODELS:-} == 1 ]] && return 0
  PYTHONPATH="$here" "$env_dir/venv/bin/python" -m dairy_engine models --root "$root"
}

if [[ -f "$env_dir/.key" && "$(cat "$env_dir/.key")" == "$key" && -x "$env_dir/venv/bin/python" && -x "$env_dir/diarizen-venv/bin/python" ]]; then
  echo "GPU 方案环境已是最新（$key）"
  download_models
  exit 0
fi
echo "建 GPU 方案环境 → $env_dir（$key）"
mkdir -p "$env_dir"
chmod 700 "$env_dir"
rm -f "$env_dir/.key"

# 主环境：完整锁定，--no-deps 保证装出来就是锁里这一套。锁没变且上次装完了就不重装。
main_key=$(sha256sum "$here/requirements-lock.txt" | cut -d ' ' -f 1)
if [[ "$(cat "$env_dir/venv/.dairy-lock" 2>/dev/null || true)" != "$main_key" ]]; then
  rm -rf "$env_dir/venv"
  "$py" -m venv "$env_dir/venv"
  "$env_dir/venv/bin/pip" install -q --timeout 60 --retries 3 "${pip_index[@]}" --no-deps -r "$here/requirements-lock.txt"
  printf '%s' "$main_key" > "$env_dir/venv/.dairy-lock"
fi

# DiariZen 源码（固定 commit）：只浅取这一个 commit，网络不稳时重试。
# DAIRY_DIARIZEN_REPO 可改用其他来源（例如本机已有的 DiariZen 仓库路径），commit 仍须一致。
diarizen_repo=${DAIRY_DIARIZEN_REPO:-https://github.com/BUTSpeechFIT/DiariZen.git}
if ! git -C "$env_dir/DiariZen" cat-file -e "$diarizen_commit^{commit}" 2>/dev/null; then
  rm -rf "$env_dir/DiariZen"
  git init --quiet "$env_dir/DiariZen"
  git -C "$env_dir/DiariZen" remote add origin "$diarizen_repo"
  for i in 1 2 3 4 5; do
    timeout 300 git -C "$env_dir/DiariZen" -c http.version=HTTP/1.1 fetch --quiet --depth 1 origin "$diarizen_commit" && break
    [[ $i == 5 ]] && { echo 'DiariZen 源码下载失败' >&2; exit 1; }
    sleep $((i * 10))
  done
fi
git -C "$env_dir/DiariZen" -c advice.detachedHead=false checkout --quiet --force --detach "$diarizen_commit"

# DiariZen 子环境：先挂主环境，再按锁装 fork 需要的其余依赖。
sub_key=$(cat "$here/diarizen-requirements-lock.txt" "$env_dir/venv/.dairy-lock" | sha256sum | cut -d ' ' -f 1)$diarizen_commit
if [[ "$(cat "$env_dir/diarizen-venv/.dairy-lock" 2>/dev/null || true)" != "$sub_key" ]]; then
rm -rf "$env_dir/diarizen-venv"
"$py" -m venv "$env_dir/diarizen-venv"
site=$("$env_dir/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
sub=$("$env_dir/diarizen-venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$site" > "$sub/dairy_engine_base.pth"
grep -E '^[A-Za-z0-9_.-]+==[^ ]+$' "$here/diarizen-requirements-lock.txt" > "$env_dir/diarizen-constraints.txt"
"$env_dir/diarizen-venv/bin/pip" install -q --timeout 60 --retries 3 "${pip_index[@]}" \
  -c "$env_dir/diarizen-constraints.txt" -r "$env_dir/DiariZen/pyannote-audio/requirements.txt" \
  toml lightning-fabric wheel psutil accelerate==1.6.0
"$env_dir/diarizen-venv/bin/pip" install -q --no-deps --no-build-isolation "$env_dir/DiariZen/pyannote-audio"
printf '%s' "$sub_key" > "$env_dir/diarizen-venv/.dairy-lock"
fi

# 自检：GPU 可用、各模块能导入。
"$env_dir/venv/bin/python" - <<'PY'
import torch, funasr, sherpa_onnx, pynvml, yaml  # noqa: E401
from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K  # noqa: F401
assert torch.cuda.is_available(), "torch 看不到 GPU"
print("主环境可用：torch", torch.__version__, "funasr", funasr.__version__, "sherpa_onnx", sherpa_onnx.__version__)
PY
PYTHONPATH="$env_dir/DiariZen" "$env_dir/diarizen-venv/bin/python" -c 'from diarizen.pipelines.inference import DiariZenPipeline; print("DiariZen 子环境可用")'

printf '%s' "$key" > "$env_dir/.key"
"$env_dir/venv/bin/python" - "$env_dir" "$key" "$diarizen_commit" <<'PY'
import json, sys, time, torch, funasr, sherpa_onnx
env, key, commit = sys.argv[1:]
json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S %z"), "key": key, "torch": torch.__version__, "funasr": funasr.__version__,
           "sherpa_onnx": sherpa_onnx.__version__, "diarizen_commit": commit}, open(f"{env}/status.json", "w"), ensure_ascii=False)
PY
echo "GPU 方案环境已建好"
download_models
