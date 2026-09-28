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



# 完整复用 WSL OnnxPolicy，单模型默认 CPU、概率集成默认 CUDA / cuda_graph=False。
import inference_runtime
from runtime_config import resolve_device
import numpy as np
from onnx_tools.policy import OnnxPolicy

from contextlib import contextmanager
from types import SimpleNamespace
from observation import ACTION_N, observation_layout
from platform_client import write_json
from ensemble_policy import DEFAULT_MODELS, ProbabilityEnsemble, default_workers
from batched_ensemble import BatchedProbabilityEnsemble

def load_model(model_path, device='cpu', *, threads=1, expected_sha256=None, gpu_memory_mb=1024):
    if device not in ('cpu', 'cuda'):
        raise ValueError('device 必须为 cpu 或 cuda')
    if type(gpu_memory_mb) is not int or gpu_memory_mb < 1:
        raise ValueError('gpu_memory_mb 必须为正整数')
    if device == 'cuda':
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('当前环境没有 CUDAExecutionProvider，请使用独立的 ONNX Runtime GPU 环境')
    model_path = Path(model_path)
    with model_path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('ONNX SHA256 与 config.json 中的 onnx_sha256 不一致')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads 必须为正整数')
    # 多桌共享 Session，不使用要求固定调用线程/缓冲区的 CUDA Graph。
    policy = OnnxPolicy(model_path, device=device, threads=threads, cuda_graph=False,
                        spin_wait=False, gpu_memory_mb=gpu_memory_mb, conv_max_workspace=False)
    if device == 'cuda':
        # CUDA 初始化与运行异常直接报错，不自动换 backend 掩盖异常。
        policy.session.disable_fallback()
    if policy.dim != observation_layout().flat_dim or policy.action_count != ACTION_N:
        raise ValueError('模型必须使用 v2 的 6122 维输入和 74 维动作')
    # 开赛前初始化 Session；这个合法 mask 仅用于预热，不提交任何动作。
    warmup = np.zeros(policy.dim, dtype=np.float32)
    warmup[-1] = 1
    policy.predict(warmup)
    return SimpleNamespace(policy=policy, sha256=digest, directory=model_path.parent, manifest={})

def load_configured_model(config):
    mode = config['inference_mode']
    device = resolve_device(config)
    gpu_memory_mb = config.get('gpu_memory_mb', 1024)
    if mode == 'single':
        return load_model(path(config['onnx']), device, threads=config['threads'],
                          expected_sha256=config['onnx_sha256'], gpu_memory_mb=gpu_memory_mb)
    if mode != 'probability_ensemble':
        raise ValueError('inference_mode 必须为 single 或 probability_ensemble')
    if config['discard_postprocess']:
        raise ValueError('概率集成直接取概率和 argmax，请关闭 discard_postprocess')
    names = config['ensemble_onnx']
    hashes = config['ensemble_sha256']
    if (not isinstance(names, list) or len(names) != 3
            or any(not isinstance(name, str) or not name for name in names)
            or len({path(name).resolve() for name in names}) != 3):
        raise ValueError('ensemble_onnx 必须包含三个不同的 ONNX 路径')
    if hashes is not None and (not isinstance(hashes, list) or len(hashes) != 3
                              or any(not isinstance(h, str) or len(h) != 64 for h in hashes)):
        raise ValueError('ensemble_sha256 必须为三个 SHA256 字符串')
    models = [load_model(path(name), device, threads=config['threads'], expected_sha256=expected,
                         gpu_memory_mb=gpu_memory_mb)
              for name, expected in zip(names, hashes or [None] * 3)]
    if len({m.sha256 for m in models}) != 3:
        raise ValueError('集成模型权重重复，请提供三个不同模型')
    workers = config.get('ensemble_workers')
    if workers is None:
        workers = 3 if device == 'cuda' else default_workers()
    manifest = dict(method='sum_legal_softmax_argmax', device=device,
                    components=[dict(path=str(path(name).resolve()), sha256=m.sha256)
                                for name, m in zip(names, models)],
                    parallel_workers=workers, threads_per_model=config['threads'],
                    providers=[m.policy.session.get_providers() for m in models])
    if device == 'cuda':
        manifest['gpu_memory_mb_per_model'] = gpu_memory_mb
        manifest['batching'] = dict(batch_size=config.get('gpu_batch_size', 16),
                                    max_wait_ms=config.get('gpu_batch_wait_ms', 20.0))
    # 内容指纹与本地路径、并行调度无关，只标识决策算法和权重。
    fingerprint = dict(method=manifest['method'], components=[m.sha256 for m in models])
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    if device == 'cuda':
        policy = BatchedProbabilityEnsemble([m.policy for m in models], workers=workers,
                    batch_size=manifest['batching']['batch_size'],
                    max_wait_ms=manifest['batching']['max_wait_ms'])
    else:
        policy = ProbabilityEnsemble([m.policy for m in models], workers=workers)
    try:
        warmup = np.zeros(policy.dim, dtype=np.float32)
        warmup[-1] = 1
        if device == 'cuda':
            # 启动时预热所有允许的 batch 大小，避免首个多桌请求触发初始化。
            for size in range(1, policy.batch_size + 1):
                policy.predict(np.repeat(warmup[None], size, axis=0))
        else:
            policy.predict(warmup)
    except BaseException:
        policy.close()
        raise
    return SimpleNamespace(policy=policy, sha256=digest, directory=ROOT, manifest=manifest)


from model_store import ModelStore


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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=("auto", "tournament", "ranking"))
    for name in ("server", "output", "tournament-id", "expected-uid", "resume-room", "ca-file", "onnx", "onnx-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--threads", type=int)
    parser.add_argument('--inference-mode', choices=('single', 'probability_ensemble'))
    parser.add_argument('--ensemble-onnx', nargs=3, metavar='ONNX')
    parser.add_argument('--ensemble-workers', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--gpu-memory-mb', type=int)
    parser.add_argument('--gpu-batch-size', type=int)
    parser.add_argument('--gpu-batch-wait-ms', type=float)
    parser.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=None)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--check", action="store_true", help="离线检查模型 SHA、输入输出和 Session 预热")
    offline.add_argument("--init-config", action="store_true", help="创建本地配置模板，不连接平台")
    offline.add_argument("--check-platform", action="store_true", help="检查模型与平台身份，不报名、不匹配、不提交动作")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = {'discard_postprocess': False, 'token': '', 'onnx': 'bc-mortal-v4-r1-13.onnx', 'onnx_sha256': None, 'threads': 1, 'mode': 'auto', 'server': 'https://10.240.169.190:18080', 'output': 'runs/client', 'tournament_id': None, 'expected_uid': None, 'resume_room': None, 'insecure': False, 'ca_file': None}
    # 老配置省略 inference_mode 时保持原单模型；新模板也默认单模型，设备随推理模式选择。
    config.update(inference_mode='single', ensemble_onnx=list(DEFAULT_MODELS), ensemble_sha256=None,
                  ensemble_workers=None, device=None, gpu_memory_mb=1024,
                  gpu_batch_size=16, gpu_batch_wait_ms=20.0)
    if args.init_config:
        destination = path(args.config)
        if not destination.exists():
            template = {key: config[key] for key in ('token', 'onnx', 'mode', 'server', 'ensemble_onnx', 'gpu_batch_size', 'gpu_batch_wait_ms')}
            template['inference_mode'] = 'single'
            destination.write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
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
    if config['inference_mode'] == 'probability_ensemble' and config['discard_postprocess']:
        raise ValueError('概率集成直接取概率和 argmax，请关闭 discard_postprocess')
    if config["discard_postprocess"]:
        from postprocess import warmup_postprocess
        warmup_postprocess()
    if args.check:
        model = load_configured_model(config)
        try:
            print(json.dumps(dict(sha256=model.sha256, status='passed', device=resolve_device(config),
                                  inference_mode=config['inference_mode'], manifest=model.manifest)))
        finally:
            close = getattr(model.policy, 'close', None)
            if close:
                close()
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
        # 固定回退目标与启动时 SHA，热切换不接受任意模型路径或改变平台配置。
        fallback_config = dict(config, inference_mode='single', device='cpu', onnx=str(path(config['onnx']).resolve()))
        if not fallback_config['onnx_sha256']:
            with Path(fallback_config['onnx']).open('rb') as handle:
                fallback_config['onnx_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        policy = ModelStore(config, output, loader=load_configured_model, fallback_config=fallback_config)
        stack.callback(policy.close)
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
                      model_sha256=policy.current.sha256, device=resolve_device(config), postprocess="shape_lookahead" if policy.discard_postprocess else None,
                      pid=os.getpid(), platform=sys.platform,
                      inference_mode=config['inference_mode'], manifest=policy.current.manifest)
        # 打榜和比赛都支持运行时回退到已校验的原单模型 + CPU。
        policy.start_control(config['mode'], client.record)
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
