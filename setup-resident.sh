#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ ! -e "$ROOT/.venv-resident" ]] || { echo '.venv-resident 已存在；拒绝原地修改可能运行中的环境。' >&2; exit 1; }
"${PYTHON:-python3}" -m venv "$ROOT/.venv-resident"
PY="$ROOT/.venv-resident/bin/python"
"$PY" -m pip install -r requirements-resident.txt
"$PY" -X utf8 bootstrap.py
"$PY" -X utf8 run.py --resident-strategies --device cuda --check
