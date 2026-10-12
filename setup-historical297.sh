#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# WSL/Linux：在客户端目录建立环境，模型和源码均来自本目录。
"${PYTHON:-python3}" -m venv "$ROOT/.venv-historical297"
PY="$ROOT/.venv-historical297/bin/python"
"$PY" -m pip install -r "$ROOT/requirements-historical297.txt"
"$PY" -X utf8 bootstrap.py
"$PY" -X utf8 run.py --inference-mode historical297 --device cpu --no-resident-strategies --check
echo 'historical297 已准备好；使用 bash run-historical297.sh 启动。'
