"""三个独立 ONNX Session 并行推理，合法动作概率等权集成。"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


DEFAULT_MODELS = [f'bc-mortal-v4-r1-{number}.onnx' for number in (17, 12, 14)]


def default_workers():
    """为并发对局保留容量，上限为 16 桌各三路；按可用 CPU 数限流。"""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return max(3, min(48, cpus))


def sum_probabilities(logit_rows, legal):
    """每个模型先做合法动作 softmax，再按固定模型顺序相加。"""
    legal = np.asarray(legal, dtype=bool)
    if len(logit_rows) != 3 or legal.ndim != 2 or not legal.any(axis=1).all():
        raise ValueError('集成需要三个输出和每行非空的合法动作 mask')
    total = np.zeros(legal.shape, dtype=np.float64)
    for raw in logit_rows:
        logits = np.asarray(raw, dtype=np.float64)
        if logits.shape != legal.shape or not np.isfinite(logits[legal]).all():
            raise ValueError('集成模型的 logits 维度错误或合法动作分数非有限值')
        selected = np.where(legal, logits, -np.inf)
        weights = np.exp(selected - selected.max(axis=1, keepdims=True))
        total += weights / weights.sum(axis=1, keepdims=True)
    return total


class ProbabilityEnsemble:
    """全客户端复用有界 worker 池；每步提交三路，多桌共享池容量。"""

    def __init__(self, policies, *, workers=None):
        self.policies = tuple(policies)
        if len(self.policies) != 3:
            raise ValueError('概率集成恰需三个模型')
        self.dim = self.policies[0].dim
        self.action_count = self.policies[0].action_count
        if any((p.dim, p.action_count) != (self.dim, self.action_count) for p in self.policies):
            raise ValueError('三个模型的 observation/action 维度必须一致')
        self.workers = default_workers() if workers is None else workers
        if type(self.workers) is not int or not 3 <= self.workers <= 48:
            raise ValueError('ensemble_workers 必须为 3 至 48 的整数')
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='onnx-ensemble')

    def _input(self, observations):
        # 独立副本供三个 Session 只读共享，避免调用方并发修改输入。
        obs = np.array(observations, dtype=np.float32, copy=True, order='C')
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.ndim != 2 or obs.shape[1] != self.dim or not len(obs):
            raise ValueError('集成 observation 维度错误')
        mask = obs[:, -self.action_count:]
        if (not np.isfinite(obs).all() or not ((mask == 0) | (mask == 1)).all()
                or not mask.any(axis=1).all()):
            raise ValueError('observation 必须有限，合法动作 mask 必须为非空 0/1')
        obs.setflags(write=False)
        return obs

    def probabilities(self, observations, *, parallel=True):
        obs = self._input(observations)
        if parallel:
            futures = [self.executor.submit(p.scores, obs) for p in self.policies]
            try:
                rows = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        else:
            rows = [p.scores(obs) for p in self.policies]
        return sum_probabilities(rows, obs[:, -self.action_count:] == 1)

    def predict(self, observations):
        # 与 Round v8 对齐：概率和转 float32 后取 argmax，保持近似并列时的规则。
        return self.probabilities(observations).astype(np.float32).argmax(axis=1)

    def scores(self, observations):
        # 保持 policy.scores 的 logits 接口；softmax 后等于三个分布的均值。
        with np.errstate(divide='ignore'):
            return np.log(self.probabilities(observations) / 3.0)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
