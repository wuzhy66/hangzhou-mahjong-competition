#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DEVICE=cpu
SUFFIX=
REQUIREMENTS=requirements.txt
case "${1:-}" in
  '') ;;
  --cpu) ;;
  --cuda) DEVICE=cuda; SUFFIX=-cuda; REQUIREMENTS=requirements-cuda.txt ;;
  *) echo '用法：bash setup.sh [--cpu|--cuda]' >&2; exit 1 ;;
esac
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM=windows
    mkdir -p "$ROOT/.tools"
    # Windows PowerShell 5 使用自己的模块路径，避免继承 PowerShell 7 的不兼容模块。
    cat > "$ROOT/.tools/bootstrap-uv.ps1" <<'POWERSHELL'
param([Parameter(Mandatory=$true)][string]$InstallDirectory)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = Join-Path $PSHOME 'Modules'
$env:UV_UNMANAGED_INSTALL = $InstallDirectory
$env:UV_NO_MODIFY_PATH = '1'
$installerFile = Join-Path (Split-Path $InstallDirectory -Parent) 'install-uv.ps1'
Invoke-WebRequest 'https://astral.sh/uv/install.ps1' -UseBasicParsing -OutFile $installerFile
& $installerFile
if (-not (Test-Path -LiteralPath (Join-Path $InstallDirectory 'uv.exe'))) {
    throw 'uv installation failed'
}
POWERSHELL
    UV="$ROOT/.tools/uv-windows/uv.exe"
    if [[ ! -x "$UV" ]]; then
      powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
        "$(cygpath -w "$ROOT/.tools/bootstrap-uv.ps1")" \
        -InstallDirectory "$(cygpath -w "$ROOT/.tools/uv-windows")"
    fi
    export UV_PROJECT_ENVIRONMENT="$(cygpath -w "$ROOT/.venv-windows${SUFFIX}")"
    export UV_PYTHON_INSTALL_DIR="$(cygpath -w "$ROOT/.tools/python-windows")"
    PY="$ROOT/.venv-windows${SUFFIX}/Scripts/python.exe"
    ;;
  Linux*)
    PLATFORM=linux
    UV="$ROOT/.tools/uv-linux/uv"
    if [[ ! -x "$UV" ]]; then
      mkdir -p "$ROOT/.tools"
      INSTALLER="$ROOT/.tools/install-uv.sh"
      if command -v curl >/dev/null 2>&1; then
        curl --fail --location --proto '=https' https://astral.sh/uv/install.sh -o "$INSTALLER"
      else
        wget -O "$INSTALLER" https://astral.sh/uv/install.sh
      fi
      UV_UNMANAGED_INSTALL="$ROOT/.tools/uv-linux" UV_NO_MODIFY_PATH=1 sh "$INSTALLER"
    fi
    export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-linux${SUFFIX}"
    export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python-linux"
    PY="$ROOT/.venv-linux${SUFFIX}/bin/python"
    ;;
  *) echo '支持 Windows Git Bash 和 Linux Bash。' >&2; exit 1 ;;
esac
export UV_LINK_MODE=copy
if [[ ! -x "$PY" ]]; then
  "$UV" venv --python 3.12 --managed-python "$UV_PROJECT_ENVIRONMENT"
fi
"$PY" -X utf8 bootstrap.py
"$UV" pip install --python "$UV_PROJECT_ENVIRONMENT/$([[ "$PLATFORM" == windows ]] && echo Scripts/python.exe || echo bin/python)" -r "$REQUIREMENTS"
"$PY" -X utf8 run.py --init-config
"$PY" -X utf8 run.py --device "$DEVICE" --check
echo "环境已准备好 ($PLATFORM)。编辑 config.json 填写 token 和 onnx，然后执行 bash run.sh"
