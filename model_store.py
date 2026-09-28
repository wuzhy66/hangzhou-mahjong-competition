"""模型版本引用计数与本地控制监听；打榜和比赛共用。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid

from runtime_config import resolve_device
from switch_to_cpu import write_json


class ModelStore:
    def __init__(self, config, output, *, loader, fallback_config, writer=write_json):
        self._config = dict(config)
        self._fallback = dict(fallback_config)
        self.output = Path(output)
        self.loader, self.writer = loader, writer
        self.run_id = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._switch_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._closing = False
        self._entries = []
        self._record = lambda *args, **kwargs: None
        self._seen = set()
        self.discard_postprocess = config['discard_postprocess']
        model = loader(config)
        self._active = dict(model=model, refs=0, retired=False)
        self._entries.append(self._active)
        try:
            self.writer(self.output/'active_model.json', self._metadata(model, config))
        except BaseException:
            self._close_model(model)
            raise

    @property
    def current(self):
        with self._condition:
            return self._active['model']

    @staticmethod
    def _close_model(model):
        close = getattr(model.policy, 'close', None)
        if close:
            close()

    def _metadata(self, model, config, *, enabled=False, state='running'):
        return dict(run_id=self.run_id, pid=os.getpid(), state=state,
                    hot_switch_enabled=enabled, mode=config.get('mode'),
                    directory=str(model.directory), sha256=model.sha256,
                    source_sha256=model.manifest.get('source_sha256'),
                    device=resolve_device(config), inference_mode=config['inference_mode'],
                    manifest=model.manifest, fallback_onnx=self._fallback['onnx'],
                    fallback_sha256=self._fallback['onnx_sha256'])

    @contextmanager
    def decision(self):
        with self._condition:
            if self._closing:
                raise RuntimeError('模型管理器已关闭')
            entry = self._active
            entry['refs'] += 1
        try:
            yield entry['model']
        finally:
            dispose = None
            with self._condition:
                entry['refs'] -= 1
                if entry['retired'] and entry['refs'] == 0:
                    self._entries.remove(entry)
                    dispose = entry['model']
                self._condition.notify_all()
            if dispose is not None:
                self._close_model(dispose)

    def fallback_cpu(self):
        with self._switch_lock:
            with self._condition:
                if self._closing:
                    raise RuntimeError('客户端正在退出')
                if self._config['inference_mode'] == 'single' and resolve_device(self._config) == 'cpu':
                    return self._metadata(self.current, self._config, enabled=True)
            # 加载/SHA 校验/预热均不持有决策锁；失败不影响现有模型。
            config = dict(self._fallback, mode=self._config.get('mode'))
            model = self.loader(config)
            dispose = None
            try:
                with self._condition:
                    if self._closing:
                        raise RuntimeError('客户端正在退出')
                    metadata = self._metadata(model, config, enabled=True)
                    self.writer(self.output/'active_model.json', metadata)
                    old = self._active
                    old['retired'] = True
                    self._active = dict(model=model, refs=0, retired=False)
                    self._entries.append(self._active)
                    self._config = config
                    if old['refs'] == 0:
                        self._entries.remove(old)
                        dispose = old['model']
            except BaseException:
                self._close_model(model)
                raise
        if dispose is not None:
            self._close_model(dispose)
        return metadata

    def start_control(self, mode, record=None):
        if mode not in ('ranking', 'tournament'):
            raise ValueError('热切换仅用于打榜或比赛')
        with self._condition:
            if self._thread is not None or self._closing:
                raise RuntimeError('控制监听已启动或模型管理器已关闭')
            self._config['mode'] = mode
            self._record = record or self._record
            self.writer(self.output/'active_model.json', self._metadata(self.current, self._config, enabled=True))
            self._thread = threading.Thread(target=self._watch, name='model-control', daemon=True)
            self._thread.start()

    def process_control(self):
        request = json.loads((self.output/'model-control.json').read_text(encoding='utf-8'))
        request_id = request.get('request_id')
        if not isinstance(request_id, str) or not request_id or request_id in self._seen:
            return
        self._seen.add(request_id)
        status = dict(run_id=self.run_id, request_id=request_id, time=time.time())
        try:
            if request.get('run_id') != self.run_id:
                raise ValueError('请求属于其他客户端实例，拒绝旧请求')
            if request.get('action') != 'fallback_cpu':
                raise ValueError('仅支持 fallback_cpu')
            status.update(status='loading')
            self.writer(self.output/'model-control-status.json', status)
            metadata = self.fallback_cpu()
            status.update(status='applied', active_model=metadata)
        except Exception as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
        self.writer(self.output/'model-control-status.json', status)
        self._record('model_hot_switch', **status)

    def _watch(self):
        while not self._stop.wait(.5):
            try:
                self.process_control()
            except FileNotFoundError:
                pass
            except Exception as error:
                # 控制文件/日志写入失败不停止对局；错误可见于 stderr。
                import sys
                print(f'热切换控制异常：{type(error).__name__}: {error}', file=sys.stderr, flush=True)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._switch_lock:
            with self._condition:
                if self._closing:
                    return
                self._closing = True
                self._condition.wait_for(lambda: all(e['refs'] == 0 for e in self._entries))
                entries, self._entries = self._entries, []
                metadata = self._metadata(self.current, self._config, state='stopped')
            try:
                self.writer(self.output/'active_model.json', metadata)
            finally:
                for entry in entries:
                    self._close_model(entry['model'])