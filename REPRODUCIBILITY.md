# Reproducibility Checklist

Use this checklist when reporting a run in the paper or supplementary material.

## 1. Code Snapshot
- Repository URL
- Commit hash (`git rev-parse HEAD`)
- Any local modifications (should be none for official results)

## 2. Environment
- OS version (for example, Ubuntu 22.04)
- Python version
- `torch` / `torchvision` versions
- CUDA runtime and NVIDIA driver versions (if GPU is used)
- Full dependency snapshot (`pip freeze > requirements_lock.txt`)

## 3. Data
- Dataset root path
- Dataset name and split names used in config
- Any filtering/subsampling rules
- Data conversion settings for downstream YOLO evaluation

## 4. Training Configuration
- Exact config file(s) under `configs/`
- Final values for:
  - `data.*`
  - `training.*`
  - `phase_schedule.*`
  - `generator.param_edit.*`
- Random seeds used for train/val splitting and training

## 5. Commands
- Full training command
- Full conversion/export command
- Full downstream evaluation commands

## 6. Artifacts to Archive
- Checkpoints (`runs/checkpoint/*.pt`)
- Training logs and TensorBoard events
- Converted images used for downstream tasks
- Evaluation outputs:
  - `evaluation/yolo/runs/<variant>/metrics.json`
  - `evaluation/yolo/runs/<variant>/per_class_metrics.csv`
  - `evaluation/yolo/runs/summaries/summary.csv`

## 7. Reported Metrics
- Main table values (mIoU, Dice, etc.)
- Mean/std over repeated runs (if reported)
- Mapping between each table row and its checkpoint/config
