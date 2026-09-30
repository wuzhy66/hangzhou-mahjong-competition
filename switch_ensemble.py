"""向运行中的客户端提交已校验的三模型概率集成热更请求。"""
import argparse
import json
from pathlib import Path
import time
import uuid

from switch_to_cpu import write_json


def request_switch(output, config_path):
    output = Path(output).resolve()
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if Path(config['output']).resolve() != output:
        raise ValueError('目标配置 output 与运行实例不一致')
    if config.get('inference_mode') != 'probability_ensemble' or config.get('discard_postprocess') is not False:
        raise ValueError('仅接受关闭后处理的概率集成配置')
    active = json.loads((output / 'active_model.json').read_text(encoding='utf-8'))
    if not active.get('hot_switch_enabled') or active.get('state') != 'running':
        raise RuntimeError('该客户端尚未启用热更或已经退出')
    request = dict(run_id=active['run_id'], request_id=uuid.uuid4().hex,
                   action='switch_ensemble', requested_at=time.time(),
                   ensemble_onnx=config['ensemble_onnx'],
                   ensemble_sha256=config['ensemble_sha256'], device=config['device'])
    write_json(output / 'model-control.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error('timeout 必须大于0且不超过300秒')
    request = request_switch(args.output, args.config)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            status = json.loads((args.output / 'model-control-status.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            status = {}
        if status.get('run_id') == request['run_id'] and status.get('request_id') == request['request_id']:
            if status.get('status') == 'applied':
                model = status['active_model']
                print(json.dumps(dict(status='applied', run_id=request['run_id'],
                                      request_id=request['request_id'], sha256=model['sha256'],
                                      device=model['device']), ensure_ascii=False))
                return
            if status.get('status') == 'failed':
                raise RuntimeError(status.get('error', '热更失败，保留原模型'))
        time.sleep(.1)
    raise RuntimeError('等待热更确认超时，请检查 model-control-status.json，勿当作成功')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'热更失败：{error}')
