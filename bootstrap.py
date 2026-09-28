"""首次安装时补齐仓库本地 Git LFS，并获取 ONNX；不读取平台 token。"""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = 'v3.8.0'
CHECKSUMS_SHA256 = 'ca28b1768b1889964350910aabe304f37c9ef3245e90520309c13118c5ff8e07'


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'mahjong-competition-setup'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def install_lfs():
    machine = platform.machine().lower()
    arch = {'amd64': 'amd64', 'x86_64': 'amd64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(machine)
    system = {'Windows': 'windows', 'Linux': 'linux'}.get(platform.system())
    if not arch or not system:
        raise RuntimeError('当前系统请自行安装 Git LFS，然后重新运行 setup.sh')
    extension = 'zip' if system == 'windows' else 'tar.gz'
    name = f'git-lfs-{system}-{arch}-{VERSION}.{extension}'
    base = f'https://github.com/git-lfs/git-lfs/releases/download/{VERSION}/'
    # 固定官方校验清单的 SHA，安装不依赖有匿名限额的 GitHub API。
    checksums = download(base + 'sha256sums.asc')
    if hashlib.sha256(checksums).hexdigest() != CHECKSUMS_SHA256:
        raise RuntimeError('Git LFS 官方校验清单校验失败')
    expected = next((parts[0] for line in checksums.decode('ascii').splitlines()
                     if len(parts := line.split()) == 2 and parts[1].lstrip('*') == name), None)
    if expected is None:
        raise RuntimeError('官方 Git LFS 没有当前平台的资产，请自行安装 Git LFS 后重试')
    data = download(base + name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Git LFS 下载校验失败')
    binary_name = 'git-lfs.exe' if system == 'windows' else 'git-lfs'
    # 只取已校验压缩包中的可执行文件，不展开任意路径。
    if extension == 'zip':
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            candidates = [n for n in archive.namelist() if n.split('/')[-1] == binary_name]
            if len(candidates) != 1:
                raise RuntimeError('Git LFS 压缩包格式异常')
            binary = archive.read(candidates[0])
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            candidates = [m for m in archive.getmembers() if m.isfile() and m.name.split('/')[-1] == binary_name]
            if len(candidates) != 1:
                raise RuntimeError('Git LFS 压缩包格式异常')
            with archive.extractfile(candidates[0]) as handle:
                binary = handle.read()
    directory = ROOT / '.tools' / ('git-lfs-' + system + '-' + arch)
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / binary_name
    executable.write_bytes(binary)
    executable.chmod(0o755)
    os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')


def main():
    os.chdir(ROOT)
    config_file = ROOT / 'config.json'
    config = json.loads(config_file.read_text(encoding='utf-8-sig')) if config_file.exists() else {}
    mode = config.get('inference_mode', 'single')
    if mode not in ('single', 'probability_ensemble'):
        raise ValueError('未知 inference_mode')
    names = (config.get('ensemble_onnx', [f'bc-mortal-v4-r1-{n}.onnx' for n in (11, 14, 12)])
             if mode == 'probability_ensemble' else [config.get('onnx', 'bc-mortal-v4-r1-13.onnx')])
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or mode == 'probability_ensemble' and len(names) != 3):
        raise ValueError('ensemble_onnx 必须包含三个 ONNX 路径')
    pointers = []
    for name in names:
        model = Path(name).expanduser()
        if not model.is_absolute():
            model = ROOT / model
        if not model.is_file():
            raise RuntimeError(f'缺少模型 {model}，请先放置对应完整 ONNX 权重')
        with model.open('rb') as handle:
            if handle.read(256).startswith(b'version https://git-lfs.github.com/spec/v1'):
                pointers.append(model)
    if not pointers:
        return
    if not shutil.which('git'):
        raise RuntimeError('请先安装 Git，再运行 setup.sh')
    version = subprocess.run(['git', 'lfs', 'version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if version.returncode:
        install_lfs()
    # 保存绝对可执行路径，使没有全局 Git LFS 的机器后续也能使用过滤器。
    executable = shutil.which('git-lfs')
    if not executable:
        raise RuntimeError('Git LFS 安装后仍不可用')
    executable = Path(executable).as_posix()
    for key, command in {'clean': 'clean -- %f', 'smudge': 'smudge -- %f', 'process': 'filter-process'}.items():
        subprocess.run(['git', 'config', '--local', 'filter.lfs.' + key,
                        shlex.quote(executable) + ' ' + command], check=True)
    subprocess.run(['git', 'config', '--local', 'filter.lfs.required', 'true'], check=True)
    for model in pointers:
        try:
            model_name = model.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            raise RuntimeError('仓库外的 ONNX 是 LFS 指针，请先在其所属仓库下载完整模型')
        subprocess.run(['git', 'lfs', 'pull', '--include=' + model_name], check=True)
        with model.open('rb') as handle:
            if handle.read(256).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError('ONNX 仍是 LFS 指针；请仓库发布者检查模型是否已上传至远端 LFS')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('模型准备失败：' + str(error))
