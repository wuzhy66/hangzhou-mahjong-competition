#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${RESIDENT_PYTHON:-$ROOT/.venv-resident/bin/python}"
[[ -x "$PY" ]] || { echo '请先执行 bash setup-resident.sh，准备独立双策略 CUDA 环境。' >&2; exit 1; }
export PYTHONUTF8=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$PY" -u "$ROOT/run.py" --resident-strategies --device cuda "$@"
