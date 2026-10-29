#!/usr/bin/env bash
# Stage 0: Miniforge, conda 환경(geo, sr, osediff), 가중치 설치. 여러 번 실행해도 안전하다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cfg() { sed -n "s/^$1: *//p" "$ROOT/configs/default.yaml"; }
MF=$(cfg miniforge); W=$(cfg weights_dir); CACHE=$(cfg cache_dir)
TP=$ROOT/third_party
export CONDA_PKGS_DIRS=$CACHE/conda PIP_CACHE_DIR=$CACHE/pip HF_HOME=$W/hf TORCH_HOME=$W/torch
# RunPod 템플릿이 HF_HUB_ENABLE_HF_TRANSFER=1을 설정하지만 osediff 환경에는 hf_transfer가 없다.
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

MF_VER=26.9.4-1
if [ ! -x "$MF/bin/conda" ]; then
  mkdir -p /root/dl
  wget -q -O /root/dl/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/download/$MF_VER/Miniforge3-$MF_VER-Linux-x86_64.sh"
  bash /root/dl/Miniforge3.sh -b -p "$MF"
fi
CONDA=$MF/bin/conda

ENVS="geo sr"
[ "$(cfg enable_osediff)" = true ] && ENVS="$ENVS osediff"
for n in $ENVS; do
  if [ -d "$MF/envs/$n" ]; then
    $CONDA env update -n "$n" -f "$ROOT/env/$n.yml"
  else
    $CONDA env create -n "$n" -f "$ROOT/env/$n.yml"
  fi
done

# sr: basicsr 1.4.2는 setup.py가 torch를 import하므로 --no-build-isolation.
# facexlib 버전은 ResShift environment.yml, gfpgan은 Real-ESRGAN requirements.txt(>=1.3.5)를 따른다.
PY=$MF/envs/sr/bin/python
$PY -m pip install --no-build-isolation basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8
$PY -m pip install --no-build-isolation --no-deps -e "$TP/Real-ESRGAN"

if [ "$(cfg enable_osediff)" = true ]; then
  $MF/envs/osediff/bin/python -m pip install --no-build-isolation basicsr==1.4.2
fi

for n in $ENVS; do
  { $CONDA list -n "$n" --explicit; echo "# pip freeze"; $MF/envs/$n/bin/python -m pip freeze; } > "$ROOT/env/$n.lock.txt"
done
$CONDA clean -a -y
rm -rf "$PIP_CACHE_DIR"

# 가중치 (공식 릴리스 URL)
mkdir -p "$W/yolo" "$W/swinir" "$W/realesrgan"
wget -q -nc -P "$W/yolo" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt
wget -q -nc -P "$W/swinir" https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
wget -q -nc -P "$W/realesrgan" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
wget -q -nc -P "$W/realesrgan" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget -q -nc -P "$W/realesrgan" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth

if [ "$(cfg enable_resshift)" = true ]; then
  # inference_resshift.py는 cwd 기준 ./weights에서 가중치를 찾는다.
  mkdir -p "$W/resshift"
  wget -q -nc -P "$W/resshift" https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth
  wget -q -nc -P "$W/resshift" https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth
  ln -sfn "$W/resshift" "$TP/ResShift/weights"
fi

if [ "$(cfg enable_osediff)" = true ]; then
  # osediff.pkl, DAPE.pth는 OSEDiff 저장소 preset/models에 포함되어 있어 그대로 쓴다.
  mkdir -p "$W/osediff"
  wget -q -nc -P "$W/osediff" https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
  $MF/envs/osediff/bin/python - "$W" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
w = sys.argv[1]
# OSEDiff README가 링크한 SD2.1-base 미러. diffusers가 읽는 fp32 safetensors와 설정만 받는다.
snapshot_download('Manojb/stable-diffusion-2-1-base', revision='0094d483a120f3f33dafbd187ea4aa60d10de75c',
                  local_dir=f'{w}/sd21_base',
                  allow_patterns=['model_index.json', '*/*.json', 'tokenizer/*', 'text_encoder/model.safetensors',
                                  'unet/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors'])
# ram/models/utils.py가 저자 로컬 경로에서 읽는 bert-base-uncased 토크나이저(주석 처리된 원래 ID)
snapshot_download('google-bert/bert-base-uncased', local_dir=f'{w}/bert-base-uncased',
                  allow_patterns=['config.json', 'tokenizer_config.json', 'vocab.txt', 'tokenizer.json'])
PYEOF
  mkdir -p /home/notebook/data/group/LowLevelLLM/LLM
  ln -sfn "$W/bert-base-uncased" /home/notebook/data/group/LowLevelLLM/LLM/bert-base-uncased
fi
