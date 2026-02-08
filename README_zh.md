# FD-DB：用于无配对 Syn2Real 翻译的频率解耦双分支网络

[English README](README.md)

FD-DB 论文代码开源仓库。

## 核心思路
FD-DB 将翻译分为两个分支：
- `G_edit`：低频、可解释的参数化编辑
- `G_free`：高频残差补偿

## YCB-V 下游分割结果（mIoU）
| 设置 | mIoU |
|---|---:|
| real | 0.7018 |
| synthetic | 0.2768 |
| synthetic_real | 0.4942 |
| free_only | 0.6283 |
| edit_only | 0.5720 |
| edit_free (hp=8) | 0.6533 |

## 仓库结构
```text
FD-DB/
|- configs/               # 训练配置
|- dataloader/            # BOP 适配与无配对数据加载
|- models/                # 生成器、判别器、损失与算子
|- trains/                # 训练入口
|- convert/               # 翻译/导出
|- evaluation/yolo/       # YOLOv8-seg 下游评估
|- scripts/               # 辅助脚本
|- REPRODUCIBILITY.md     # 可复现清单
|- THIRD_PARTY_NOTICES.md # 第三方声明
`- CITATION.cff           # 引用信息
```

## 环境要求
- 操作系统：Ubuntu 20.04/22.04（已在 Ubuntu 22.04.5 LTS 验证）
- Python：3.10（已在 3.10.19 验证）
- PyTorch 组件：`torch==2.3.1`、`torchvision==0.18.1`
- GPU（可选）：已验证 `torch==2.3.1+cu121`（CUDA 12.1 runtime）
- NVIDIA 驱动：使用 GPU 时需与 CUDA 12.1 兼容
- 安装时需要联网（`requirements.txt` 中的 `bop_toolkit` 通过 GitHub 拉取）

## 环境安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如需运行 YOLO 下游评估：
```bash
pip install -r evaluation/yolo/requirements.txt
```

## 数据准备
本仓库不分发数据集。

在训练配置（`configs/default*.yaml`）中设置：
- `data.bop.datasets_root`
- `data.bop.dataset_name`
- `data.bop.syn_dir`
- `data.bop.real_source.*`

在 `evaluation/yolo/configs/ycbv_default.yaml` 中设置评估相关路径。

## 训练
主配置：
```bash
python trains/trainer.py --config configs/default.yaml
```

脚本方式：
```bash
bash scripts/train.sh
bash scripts/train_edit_free.sh
bash scripts/train_free_only.sh
```

## 翻译/导出
直接运行：
```bash
python convert/translate_bop.py \
  --checkpoint runs/checkpoint/step0000500_epoch0001.pt \
  --datasets-root /path/to/bop/datasets \
  --dataset-name ycbv \
  --split ycbv_train_pbr \
  --output-dir /path/to/output/ycbv_translated
```

脚本方式：
```bash
bash scripts/convert_translation.sh <checkpoint.pt> <output_dir> [datasets_root] [dataset_name] [split]
```

## 下游评估（YOLOv8-seg）
```bash
cd evaluation/yolo
bash sh/run_prepare.sh configs/ycbv_default.yaml all
bash sh/run_train.sh configs/ycbv_default.yaml real
bash sh/run_eval.sh configs/ycbv_default.yaml real
bash sh/run_summary.sh configs/ycbv_default.yaml
```

分阶段训练（合成预训练 -> 真实微调）：
```bash
bash sh/run_train.sh configs/ycbv_default.yaml staged
```

主要输出：
- `evaluation/yolo/runs/<variant>/metrics.json`
- `evaluation/yolo/runs/<variant>/per_class_metrics.csv`
- `evaluation/yolo/runs/summaries/summary.csv`

## 引用
请使用 `CITATION.cff` 中的信息进行引用。

## 许可证
- 代码：MIT（`LICENSE`）
- 第三方声明：`THIRD_PARTY_NOTICES.md`
