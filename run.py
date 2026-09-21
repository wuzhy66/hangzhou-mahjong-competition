"""跨平台统一入口；配置、离线校验、身份检查与唯一运行锁。"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent



# 完整复用 WSL OnnxPolicy，默认 CPU / cuda_graph=False。
import inference_runtime
import numpy as np
from onnx_tools.policy import OnnxPolicy

from contextlib import contextmanager
from types import SimpleNamespace
from observation import ACTION_N, observation_layout
from platform_client import write_json

def load_model(model_path, device='cpu', *, threads=1, expected_sha256=None):
    if device != 'cpu':
        raise ValueError('只支持 CPU ONNX')
    model_path = Path(model_path)
    with model_path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('ONNX SHA256 与 config.json 中的 onnx_sha256 不一致')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads 必须为正整数')
    policy = OnnxPolicy(model_path, device='cpu', threads=threads, cuda_graph=False, spin_wait=False)
    if policy.dim != observation_layout().flat_dim or policy.action_count != ACTION_N:
        raise ValueError('模型必须使用 v2 的 6122 维输入和 74 维动作')
    # 开赛前初始化 Session；这个合法 mask 仅用于预热，不提交任何动作。
    warmup = np.zeros(policy.dim, dtype=np.float32)
    warmup[-1] = 1
    policy.predict(warmup)
    return SimpleNamespace(policy=policy, sha256=digest, directory=model_path.parent, manifest={})

class ModelStore:
    def __init__(self, model_path, device, output, record, *, threads=1, expected_sha256=None):
        self.current = load_model(model_path, device, threads=threads, expected_sha256=expected_sha256)
        model = self.current
        write_json(Path(output) / 'active_model.json', dict(
            directory=str(model.directory), sha256=model.sha256,
            source_sha256=model.manifest.get('source_sha256'), device='cpu'))

    @contextmanager
    def decision(self):
        yield self.current


def path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def choose_mode(mode, me, tournament_id):
    """参赛专用令牌跟随绑定比赛；全局令牌默认打榜。"""
    bound = me.get('tournament_id')
    if bound and tournament_id and bound != tournament_id:
        raise ValueError('指定的比赛 ID 与令牌绑定比赛不同')
    if mode == 'auto':
        return 'tournament' if bound or tournament_id else 'ranking'
    if mode == 'ranking' and (bound or tournament_id):
        raise ValueError('打榜不能使用参赛专用令牌或 --tournament-id')
    if mode == 'tournament' and not (bound or tournament_id):
        raise ValueError('全局令牌参加比赛时，请加 --tournament-id 官方比赛ID')
    return mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=("auto", "tournament", "ranking"))
    for name in ("server", "output", "tournament-id", "expected-uid", "resume-room", "ca-file", "onnx", "onnx-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=None)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--check", action="store_true", help="离线检查模型 SHA、输入输出和 Session 预热")
    offline.add_argument("--init-config", action="store_true", help="创建本地配置模板，不连接平台")
    offline.add_argument("--check-platform", action="store_true", help="检查模型与平台身份，不报名、不匹配、不提交动作")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = {'discard_postprocess': False, 'token': '', 'onnx': 'bc-mortal-v4-r1-13.onnx', 'onnx_sha256': None, 'threads': 1, 'mode': 'auto', 'server': 'https://10.240.169.190:18080', 'output': 'runs/client', 'tournament_id': None, 'expected_uid': None, 'resume_room': None, 'insecure': False, 'ca_file': None}
    if args.init_config:
        destination = path(args.config)
        if not destination.exists():
            destination.write_text(json.dumps({key: config[key] for key in ('token', 'onnx', 'mode', 'server')}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('配置文件: ' + str(destination))
        return 0
    config_path = path(args.config)
    if config_path.exists():
        overrides = json.loads(config_path.read_text(encoding="utf-8-sig"))
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"未知配置项：{sorted(unknown)}")
        config.update(overrides)
    elif args.config != "config.json":
        raise ValueError("指定的配置文件不存在")
    config.update({key: value for key, value in vars(args).items() if key in config and value is not None})
    if not isinstance(config["discard_postprocess"], bool):
        raise ValueError("discard_postprocess 必须为 true/false")
    if config["discard_postprocess"]:
        from postprocess import warmup_postprocess
        warmup_postprocess()
    model_path = path(config["onnx"])
    if args.check:
        model = load_model(model_path, threads=config["threads"], expected_sha256=config["onnx_sha256"])
        print(json.dumps({"sha256": model.sha256, "status": "passed"}))
        return 0
    if config["mode"] not in ("auto", "tournament", "ranking"):
        raise ValueError("mode 必须为 auto、tournament 或 ranking")
    url = urlsplit(config["server"])
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("server 必须是无凭据和路径的 HTTP(S) 平台地址")
    token = config['token']
    if not isinstance(token, str):
        raise ValueError('token 必须为字符串')
    token = token.strip()
    if not token or any(char.isspace() for char in token):
        raise ValueError('请在配置文件中填写 token，不能为空或包含空白字符')
    from filelock import FileLock
    from platform_client import Client
    from platform_client import tournament, ranking, request
    output = path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # 同一台机器、同一 OS 用户，不同目录也不能重复运行同一令牌。
    identity = hashlib.sha256((config["server"].rstrip("/") + "\n" + token).encode()).hexdigest()
    lock_root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "mahjong-competition/locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    def stopping(signum, frame):
        stop.set()
    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGTERM, stopping)
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(lock_root / (identity + ".lock")), timeout=0))
        stack.enter_context(FileLock(str(output / "runtime.lock"), timeout=0))
        client = Client(config["server"], token, output, config["insecure"],
                        ca_file=path(config['ca_file']) if config['ca_file'] else None)
        policy = ModelStore(model_path, "cpu", output, client.record, threads=config["threads"], expected_sha256=config["onnx_sha256"])
        policy.discard_postprocess = config["discard_postprocess"]
        me = request(client, "/api/me", stop)
        if config["expected_uid"] and me.get("user_id") != config["expected_uid"]:
            raise ValueError("令牌实际 UID 与 expected_uid 不同，已拒绝运行")
        config['mode'] = choose_mode(config['mode'], me, config['tournament_id'])
        if args.check_platform:
            print(json.dumps(dict(status='passed', mode=config['mode'], user_id=me.get('user_id'),
                                  tournament_id=me.get('tournament_id') or config['tournament_id'],
                                  model_sha256=policy.current.sha256)))
            return 0
        client.record("client_started", mode=config["mode"], user_id=me.get("user_id"),
                      model_sha256=policy.current.sha256, device="cpu", postprocess="shape_lookahead" if policy.discard_postprocess else None,
                      pid=os.getpid(), platform=sys.platform)
        # 比赛期间模型冻结；不启动热更监听，也不执行训练。
        if config["mode"] == "tournament":
            tournament(client, policy, me, config["tournament_id"], stop=stop)
        else:
            ranking(client, policy, me, config["resume_room"], stop=stop)
        client.record("client_stopped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, InterruptedError):
        sys.exit(130)
    except Exception as error:
        print(f"启动或运行失败：{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
