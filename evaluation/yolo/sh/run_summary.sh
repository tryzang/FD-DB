#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CFG="${1:-${PROJ_ROOT}/configs/ycbv_default.yaml}"

cd "${PROJ_ROOT}"
python scripts/summarize.py --config "${CFG}"
