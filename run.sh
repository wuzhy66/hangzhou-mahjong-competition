#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PY="$ROOT/.venv-windows/Scripts/python.exe" ;;
  Linux*) PY="$ROOT/.venv-linux/bin/python" ;;
  *) echo '支持 Windows Git Bash 和 Linux Bash。' >&2; exit 1 ;;
esac
[[ -x "$PY" ]] || { echo '请先执行 bash setup.sh' >&2; exit 1; }
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$PY" -X utf8 -u "$ROOT/run.py" "$@"
