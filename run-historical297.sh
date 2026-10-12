#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 使用客户端内的独立 PyTorch 环境，不依赖上级项目。
PY="${HISTORICAL297_PYTHON:-$ROOT/.venv-historical297/bin/python}"
[[ -x "$PY" ]] || { echo '请先执行 bash setup-historical297.sh，或用 HISTORICAL297_PYTHON 指定兼容 Python。' >&2; exit 1; }
export PYTHONUTF8=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$PY" -u "$ROOT/run.py" --inference-mode historical297 --device cuda "$@"
