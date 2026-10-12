"""第297轮公开模型：校验固定权重、7113维观测和41动作 argmax。"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np

EXPECTED_SHA256 = '90799eafbb2a11a3eb586fcec6f945411a1bc4315b8cd329c1340a5732649453'

class Historical297Policy:
    strategy = 'historical297'
    dim, action_count = 7113, 41

    def __init__(self, path, device, threads):
        import torch
        from historical297_source.public_student import load_public_bundle
        if type(threads) is not int or threads < 1:
            raise ValueError('threads 必须为正整数')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用，不自动回退 CPU')
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.device = device
        self.lock = threading.Lock()
        self.model = load_public_bundle(torch.load(path, map_location='cpu', weights_only=True)).to(device).eval().requires_grad_(False)

    def scores(self, observations):
        import torch
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None]
        if obs.ndim != 2 or obs.shape[1] != self.dim or not np.isfinite(obs).all():
            raise ValueError('historical297 需要有限值的 [batch,7113] 观测')
        mask = obs[:, -41:]
        if not np.isin(mask, [0, 1]).all() or not (mask.sum(-1) > 0).all():
            raise ValueError('historical297 合法动作 mask 无效')
        with self.lock, torch.inference_mode():
            logits = self.model(torch.as_tensor(obs, device=self.device))['policy_logits'].cpu().numpy()
        return np.where(mask > 0, logits, -np.inf)

    def predict(self, observations):
        return self.scores(observations).argmax(-1).astype(np.int64)


def load_historical297(path, device='cpu', threads=1):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('historical297 模型 SHA256 不匹配')
    policy = Historical297Policy(path, device, threads)
    obs = np.zeros((2, 7113), np.float32)
    obs[0, -1] = 1
    obs[1, -3] = 1
    assert policy.predict(obs).tolist() == [40, 38]
    return SimpleNamespace(policy=policy, sha256=digest, directory=path.parent,
        manifest=dict(strategy='historical297 · argmax', observation_dim=7113,
                      action_count=41, device=device, method='legal_argmax'))
