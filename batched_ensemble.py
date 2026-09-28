"""多桌共用一个 GPU 调度器，按最早请求的可配置期限合批。"""
from collections import Counter, deque
from concurrent.futures import Future
import math
import threading
import time

import numpy as np

from ensemble_policy import ProbabilityEnsemble


class BatchedProbabilityEnsemble(ProbabilityEnsemble):
    """保留每行 observation/mask 和回包顺序，所有模型处理同一个 batch。"""

    def __init__(self, policies, *, workers=3, batch_size=16, max_wait_ms=20.0, queue_size=256):
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError('gpu_batch_size 必须为 1 至 16 的整数')
        if (type(max_wait_ms) not in (float, int) or not math.isfinite(max_wait_ms)
                or not 0 <= max_wait_ms <= 20):
            raise ValueError('gpu_batch_wait_ms 必须在 0 至 20 ms 内')
        if type(queue_size) is not int or queue_size < batch_size:
            raise ValueError('queue_size 必须是至少 batch_size 的整数')
        policies = tuple(policies)
        if any(isinstance(getattr(p, 'model_batch', None), int) for p in policies):
            raise ValueError('GPU 合批需要动态 batch ONNX，拒绝固定 batch 模型')
        super().__init__(policies, workers=workers)
        self.batch_size, self.max_wait_ms, self.queue_size = batch_size, max_wait_ms, queue_size
        self._pending = deque()
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._closing = False
        self._fatal = None
        self._batches = Counter()
        self._completed = 0
        self._max_queue_wait_ms = 0.0
        self._thread = threading.Thread(target=self._dispatch, name='onnx-batch-dispatch', daemon=True)
        self._thread.start()

    def probabilities(self, observations, *, parallel=True):
        # 单模型串行对照只供离线基准；正式 predict/scores 全部经合批队列。
        if not parallel:
            with self._condition:
                self._ensure_open()
            return super().probabilities(observations, parallel=False)
        obs = self._input(observations)
        futures = [Future() for _ in obs]
        with self._condition:
            self._ensure_open()
            if len(self._pending) + len(obs) > self.queue_size:
                raise RuntimeError('GPU 合批队列已满，请降低并发')
            received = time.monotonic()
            self._pending.extend((row, future, received) for row, future in zip(obs, futures))
            self._condition.notify()
        return np.stack([future.result() for future in futures])

    def _ensure_open(self):
        if self._fatal is not None:
            raise RuntimeError('GPU 合批推理已失败，拒绝继续使用异常 Session') from self._fatal
        if self._closing:
            raise RuntimeError('GPU 合批推理已关闭')

    def _dispatch(self):
        active = []
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if not self._pending:
                        return
                    # 从入队时刻计时，GPU 繁忙时消耗掉的窗口不再重复等待。
                    deadline = self._pending[0][2] + self.max_wait_ms / 1000.0
                    while len(self._pending) < self.batch_size and not self._closing:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                    active = [self._pending.popleft() for _ in range(min(self.batch_size, len(self._pending)))]
                    self._max_queue_wait_ms = max(self._max_queue_wait_ms,
                        (time.monotonic() - active[0][2]) * 1000)
                inputs = np.stack([row for row, _, _ in active])
                values = super().probabilities(inputs)
                with self._condition:
                    self._batches[len(active)] += 1
                    self._completed += len(active)
                for (_, future, _), value in zip(active, values):
                    future.set_result(value)
                active = []
        except BaseException as error:
            # 一批失败时唤醒所有等待方，后续请求直接报错，避免悬挂或静默降级。
            with self._condition:
                self._fatal = error
                self._closing = True
                affected = active + list(self._pending)
                self._pending.clear()
                self._condition.notify_all()
            for _, future, _ in affected:
                if not future.done():
                    future.set_exception(error)

    def stats(self):
        with self._condition:
            return dict(batch_size=self.batch_size, max_wait_ms=self.max_wait_ms,
                        queue_limit=self.queue_size, pending=len(self._pending),
                        completed=self._completed, batch_histogram=dict(self._batches),
                        max_queue_wait_ms=self._max_queue_wait_ms, failed=self._fatal is not None)

    def close(self):
        with self._close_lock:
            with self._condition:
                self._closing = True
                self._condition.notify_all()
            # 排空已接受的请求；关闭期间不再为合批额外等待。
            self._thread.join()
            super().close()
