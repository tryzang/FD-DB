#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fd-db}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH."
  exit 1
fi

echo "[fd-db] installing requirements into env ${ENV_NAME}"
conda run -n "$ENV_NAME" --no-capture-output pip install -r "$ROOT_DIR/requirements.txt"
