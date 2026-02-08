#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fd-db}"
CONFIG="$ROOT_DIR/configs/default.yaml"
# Stream python output while running via this wrapper.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/train.sh [config_path] [extra trainer args...]"
  echo "Defaults to configs/default.yaml and runs inside conda env ${ENV_NAME}."
  exit 0
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "[fd-db] training with config $CONFIG"
conda run -n "$ENV_NAME" --no-capture-output python "$ROOT_DIR/trains/trainer.py" --config "$CONFIG" "$@"
