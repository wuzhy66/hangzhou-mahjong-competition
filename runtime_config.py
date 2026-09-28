"""按推理模式选择默认设备；不加载 ORT，供启动脚本选择独立环境。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_device(config):
    mode = config.get('inference_mode', 'single')
    if mode not in ('single', 'probability_ensemble'):
        raise ValueError('inference_mode 必须为 single 或 probability_ensemble')
    device = config.get('device')
    if device is None:
        return 'cuda' if mode == 'probability_ensemble' else 'cpu'
    if device not in ('cpu', 'cuda'):
        raise ValueError('device 必须为 cpu 或 cuda')
    return device


def launch_device(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--inference-mode', choices=('single', 'probability_ensemble'))
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--init-config', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if args.init_config:
        return 'cpu'
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    elif args.config != 'config.json':
        raise ValueError('指定的配置文件不存在')
    for key in ('inference_mode', 'device'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return resolve_device(config)


if __name__ == '__main__':
    print(launch_device())