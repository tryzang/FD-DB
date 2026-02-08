#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CFG="${1:-${PROJ_ROOT}/configs/ycbv_default.yaml}"
VARIANT="${2:-real}"

cd "${PROJ_ROOT}"
python scripts/eval.py --config "${CFG}" --variant "${VARIANT}"
