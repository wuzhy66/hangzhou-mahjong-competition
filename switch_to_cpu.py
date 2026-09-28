"""跨平台热切换控制入口；只写本地请求，不连接比赛平台。"""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        deadline = time.monotonic() + .25
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.005)
    finally:
        temporary.unlink(missing_ok=True)


def request_fallback(output):
    output = Path(output)
    active = json.loads((output/'active_model.json').read_text(encoding='utf-8'))
    if not active.get('hot_switch_enabled') or active.get('state') != 'running':
        raise RuntimeError('该客户端尚未启用热切换或已经退出')
    request = dict(run_id=active['run_id'], request_id=uuid.uuid4().hex,
                   action='fallback_cpu', requested_at=time.time())
    write_json(output/'model-control.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/client')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error('timeout 必须大于 0 且不超过 300 秒')
    request = request_fallback(args.output)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            status = json.loads((args.output/'model-control-status.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            status = {}
        if status.get('request_id') == request['request_id'] and status.get('run_id') == request['run_id']:
            if status.get('status') == 'applied':
                print(json.dumps(status, ensure_ascii=False))
                return
            if status.get('status') == 'failed':
                raise RuntimeError(status.get('error', '切换失败，保留原模型'))
        time.sleep(.1)
    raise RuntimeError('等待切换确认超时；请求仍可能执行，请检查 model-control-status.json，勿当作成功')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'热切换失败：{error}')