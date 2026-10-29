#!/usr/bin/env bash
# Sets up the MotionSight environment: Python 3.10 venv, packages, and model weights.
# Usage: bash setup.sh            (packages + checkpoints)
#        SKIP_WEIGHTS=1 bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv -p 3.10 .venv
source .venv/bin/activate

uv pip install torch==2.5.1 torchvision==0.20.1
uv pip install setuptools wheel \
  transformers==4.51.3 accelerate==0.34.2 lmdeploy==0.7.3 qwen-vl-utils==0.0.10 \
  openai==1.67.0 decord==0.6.0 moviepy==2.1.2 opencv-python==4.11.0.86 \
  supervision==0.25.1 hydra-core==1.3.2 iopath==0.1.10 numpy==1.26.4 pillow==10.2.0 \
  scipy==1.15.2 tqdm matplotlib==3.7.0 imageio==2.37.0 av==14.2.0 einops==0.6.1 \
  timm==1.0.13 safetensors==0.5.2 huggingface-hub

# SAM2 + GroundingDINO from the bundled Grounded-SAM-2 sources
(cd Grounded-SAM-2 && SAM2_BUILD_CUDA=${SAM2_BUILD_CUDA:-0} uv pip install --no-build-isolation -e .)
uv pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino

# flash-attn only builds on a machine with a CUDA GPU + nvcc
if command -v nvcc >/dev/null; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn build failed; continuing without it"
fi

if [ "${SKIP_WEIGHTS:-0}" != 1 ]; then
  curl -L -o Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  huggingface-cli download IDEA-Research/grounding-dino-base --local-dir models/grounding-dino-base
  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
  (cd RAFT && curl -L -o models.zip https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip && unzip -o models.zip)
fi

echo "Done. Activate with: source .venv/bin/activate"
