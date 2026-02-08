# FD-DB: Frequency-Decoupled Dual-Branch Network for Unpaired Syn2Real Translation

[Chinese README](README_zh.md)

Official code release for the FD-DB paper.

## Main Idea
FD-DB decouples translation into two branches:
- `G_edit`: low-frequency, interpretable parametric editing
- `G_free`: high-frequency residual compensation

## Reported YCB-V Segmentation Results (mIoU)
| Setting | mIoU |
|---|---:|
| real | 0.7018 |
| synthetic | 0.2768 |
| synthetic_real | 0.4942 |
| free_only | 0.6283 |
| edit_only | 0.5720 |
| edit_free (hp=8) | 0.6533 |

## Repository Layout
```text
FD-DB/
|- configs/               # training configs
|- dataloader/            # BOP adapters and unpaired loaders
|- models/                # generator, discriminator, losses, ops
|- trains/                # training entrypoint
|- convert/               # translation/export
|- evaluation/yolo/       # YOLOv8-seg downstream evaluation
|- scripts/               # helper wrappers
|- REPRODUCIBILITY.md     # reproducibility checklist
|- THIRD_PARTY_NOTICES.md # third-party notices
`- CITATION.cff           # citation metadata
```

## Environment Requirements
- OS: Ubuntu 20.04/22.04 (tested on Ubuntu 22.04.5 LTS)
- Python: 3.10 (tested on 3.10.19)
- PyTorch stack: `torch==2.3.1`, `torchvision==0.18.1`
- GPU (optional): tested with `torch==2.3.1+cu121` / CUDA 12.1 runtime
- NVIDIA driver: must be compatible with CUDA 12.1 when using GPU
- Network access is required during install (`bop_toolkit` is pulled from GitHub in `requirements.txt`)

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional for downstream YOLO evaluation:
```bash
pip install -r evaluation/yolo/requirements.txt
```

## Data Preparation
This repository does not redistribute datasets.

Update BOP paths in one training config (`configs/default*.yaml`):
- `data.bop.datasets_root`
- `data.bop.dataset_name`
- `data.bop.syn_dir`
- `data.bop.real_source.*`

Update downstream evaluation paths in:
- `evaluation/yolo/configs/ycbv_default.yaml`

## Training
Main config:
```bash
python trains/trainer.py --config configs/default.yaml
```

Wrapper scripts:
```bash
bash scripts/train.sh
bash scripts/train_edit_free.sh
bash scripts/train_free_only.sh
```

## Translation / Export
Direct command:
```bash
python convert/translate_bop.py \
  --checkpoint runs/checkpoint/step0000500_epoch0001.pt \
  --datasets-root /path/to/bop/datasets \
  --dataset-name ycbv \
  --split ycbv_train_pbr \
  --output-dir /path/to/output/ycbv_translated
```

Wrapper command:
```bash
bash scripts/convert_translation.sh <checkpoint.pt> <output_dir> [datasets_root] [dataset_name] [split]
```

## Downstream Evaluation (YOLOv8-seg)
```bash
cd evaluation/yolo
bash sh/run_prepare.sh configs/ycbv_default.yaml all
bash sh/run_train.sh configs/ycbv_default.yaml real
bash sh/run_eval.sh configs/ycbv_default.yaml real
bash sh/run_summary.sh configs/ycbv_default.yaml
```

Staged mode (synthetic pretrain -> real finetune):
```bash
bash sh/run_train.sh configs/ycbv_default.yaml staged
```

Main outputs:
- `evaluation/yolo/runs/<variant>/metrics.json`
- `evaluation/yolo/runs/<variant>/per_class_metrics.csv`
- `evaluation/yolo/runs/summaries/summary.csv`

## Citation
Please cite using metadata in `CITATION.cff`.

## License
- Code: MIT (`LICENSE`)
- Third-party notices: `THIRD_PARTY_NOTICES.md`
