# MotionSight setup

This repo bundles [MotionSight](https://github.com/NJU-PCALab/MotionSight) with the sources it depends on:
[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) (commit `dd4c514`) and
[RAFT](https://github.com/princeton-vl/RAFT).

## Install

```bash
bash setup.sh
```

It creates a Python 3.10 environment in `.venv`, installs the packages, and downloads the model weights.
Use `SKIP_WEIGHTS=1 bash setup.sh` to install packages only.

## What you need

- Linux with an NVIDIA GPU (24 GB+ VRAM for Qwen2.5-VL-7B) and CUDA 12.x drivers
- About 40 GB of disk (packages ~8 GB, Qwen2.5-VL-7B ~16 GB)
- Network access to pypi.org, huggingface.co, dl.fbaipublicfiles.com, dl.dropboxusercontent.com

## Model weights (downloaded by setup.sh)

| Model | Location |
|---|---|
| SAM2.1 Hiera Large | `Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt` |
| Grounding DINO base | `models/grounding-dino-base` |
| Qwen2.5-VL-7B-Instruct | `models/Qwen2.5-VL-7B-Instruct` |
| RAFT | `RAFT/models/raft-things.pth` |
| MotionChat (optional, fine-tuned) | download from https://www.modelscope.cn/models/Lollikit/MotionChat/files |

## Run

```bash
source .venv/bin/activate
lmdeploy serve api_server models/Qwen2.5-VL-7B-Instruct --server-port 23333 --tp 1
```

```bash
cd Grounded-SAM-2 && python track_utils.py --p 1 --step 10000
```

```bash
python -m eval.motionsight.eval_motionbench
```

## Benchmark data (not included)

Download MotionBench and/or FAVOR-Bench yourself, then replace the `/path/to/...` dataset paths in
`eval/motionsight/eval_motionbench.py` and `eval/motionsight/eval_favorbench.py`.
