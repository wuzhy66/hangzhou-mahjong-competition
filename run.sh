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
RESIDENT_PY="${RESIDENT_PYTHON:-$ROOT/.venv-resident/bin/python}"
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ROUTER="$CPU_PY"
[[ -x "$ROUTER" ]] || ROUTER="$CUDA_PY"
[[ -x "$ROUTER" ]] || ROUTER="$RESIDENT_PY"
[[ -x "$ROUTER" ]] || { echo '请先执行 bash setup.sh；CUDA 集成另执行 bash setup.sh --cuda' >&2; exit 1; }
RUNTIME="$("$ROUTER" -X utf8 "$ROOT/runtime_config.py" --print-runtime "$@")"
if [[ "$RUNTIME" == resident ]]; then
  PY="$RESIDENT_PY"
  [[ -x "$PY" ]] || { echo '请先在 WSL/Linux 执行 bash setup-resident.sh，或设置 RESIDENT_PYTHON 指向同时具备 PyTorch CUDA 和 ORT GPU 的环境' >&2; exit 1; }
elif [[ "$RUNTIME" == cuda ]]; then
  PY="$CUDA_PY"
  [[ -x "$PY" ]] || { echo '请先执行 bash setup.sh --cuda 安装独立 GPU 环境' >&2; exit 1; }
else
  PY="$CPU_PY"
  # 只有 GPU Runtime 时，仍可显式使用其 CPU provider。
  [[ -x "$PY" ]] || PY="$CUDA_PY"
  [[ -x "$PY" ]] || PY="$RESIDENT_PY"
fi
exec "$PY" -X utf8 -u "$ROOT/run.py" "$@"
