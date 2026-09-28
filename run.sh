#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    CPU_PY="$ROOT/.venv-windows/Scripts/python.exe"
    CUDA_PY="$ROOT/.venv-windows-cuda/Scripts/python.exe" ;;
  Linux*)
    CPU_PY="$ROOT/.venv-linux/bin/python"
    CUDA_PY="$ROOT/.venv-linux-cuda/bin/python" ;;
  *) echo '支持 Windows Git Bash 和 Linux Bash。' >&2; exit 1 ;;
esac
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ROUTER="$CPU_PY"
[[ -x "$ROUTER" ]] || ROUTER="$CUDA_PY"
[[ -x "$ROUTER" ]] || { echo '请先执行 bash setup.sh；CUDA 集成另执行 bash setup.sh --cuda' >&2; exit 1; }
DEVICE="$("$ROUTER" -X utf8 "$ROOT/runtime_config.py" "$@")"
if [[ "$DEVICE" == cuda ]]; then
  PY="$CUDA_PY"
  [[ -x "$PY" ]] || { echo '请先执行 bash setup.sh --cuda 安装独立 GPU 环境' >&2; exit 1; }
else
  PY="$CPU_PY"
  # 只有 GPU Runtime 时，仍可显式使用其 CPU provider。
  [[ -x "$PY" ]] || PY="$CUDA_PY"
fi
exec "$PY" -X utf8 -u "$ROOT/run.py" "$@"
