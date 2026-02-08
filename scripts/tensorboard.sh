#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fd-db}"
LOGDIR="${1:-$ROOT_DIR/runs/tensorboard}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/tensorboard.sh [logdir] [tensorboard args]"
  echo "Defaults to runs/tensorboard in conda env ${ENV_NAME}."
  exit 0
fi

if [[ $# -gt 0 ]]; then
  shift
fi

echo "[fd-db] starting tensorboard with logdir $LOGDIR"
conda run -n "$ENV_NAME" --no-capture-output tensorboard --logdir "$LOGDIR" "$@"
