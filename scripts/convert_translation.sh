#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-}"
OUT_DIR="${2:-}"
DATASETS_ROOT="${3:-/path/to/bop/datasets}"
DATASET_NAME="${4:-ycbv}"
SPLIT="${5:-ycbv_train_pbr}"

if [[ -z "$CKPT" || -z "$OUT_DIR" ]]; then
  echo "usage: scripts/convert_translation.sh <checkpoint.pt> <output_dir> [datasets_root] [dataset_name] [split]"
  exit 1
fi

python convert/translate_bop.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT_DIR" \
  --datasets-root "$DATASETS_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --split "$SPLIT" \
  --device auto \
  --noise-mode fixed \
  --preserve-resolution
