"""按推理模式选择默认设备；不加载 ORT，供启动脚本选择独立环境。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_device(config):
    mode = config.get('inference_mode', 'historical297')
    if mode not in ('single', 'probability_ensemble', 'historical297'):
        raise ValueError('inference_mode 必须为 single、probability_ensemble 或 historical297')
    device = config.get('device')
    if device is None:
        return 'cpu' if mode == 'single' else 'cuda'
    if device not in ('cpu', 'cuda'):
        raise ValueError('device 必须为 cpu 或 cuda')
    return device


def resolve_resident(config):
    value = config.get('resident_strategies')
    if value is None:
        return config.get('inference_mode', 'historical297') == 'historical297' and resolve_device(config) == 'cuda'
    if type(value) is not bool:
        raise ValueError('resident_strategies 必须为 true/false')
    return value


def launch_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--inference-mode', choices=('single', 'probability_ensemble', 'historical297'))
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--resident-strategies', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--init-config', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if args.init_config:
        return {'inference_mode': 'single', 'device': 'cpu', 'resident_strategies': False}
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    elif args.config != 'config.json':
        raise ValueError('指定的配置文件不存在')
    for key in ('inference_mode', 'device', 'resident_strategies'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def launch_device(argv=None):
    return resolve_device(launch_config(argv))


def launch_runtime(argv=None):
    config = launch_config(argv)
    if resolve_resident(config) or config.get('inference_mode', 'historical297') == 'historical297':
        return 'resident'
    return resolve_device(config)


if __name__ == '__main__':
    import sys
    if '--print-runtime' in sys.argv[1:]:
        print(launch_runtime([arg for arg in sys.argv[1:] if arg != '--print-runtime']))
    else:
        print(launch_device())
