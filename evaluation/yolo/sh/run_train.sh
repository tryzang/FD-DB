#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CFG="${1:-${PROJ_ROOT}/configs/ycbv_default.yaml}"
MODE="${2:-real}"

cd "${PROJ_ROOT}"
if [[ "${MODE}" == "staged" ]]; then
  python scripts/train.py --config "${CFG}" --staged
else
  python scripts/train.py --config "${CFG}" --variant "${MODE}"
fi
