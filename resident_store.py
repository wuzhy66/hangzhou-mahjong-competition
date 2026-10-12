"""显式启用的双策略 CUDA 常驻；复用原决策引用与退出等待，不在切换时加载模型。"""
from pathlib import Path
import sys

from model_store import ModelStore
from runtime_config import resolve_device

NAMES = [f'bc-mortal-v4-r1-{n}.onnx' for n in (17, 12, 14)]
HASHES = [
    '7342a41f9f4ece71f97698592fdfc54bee2d967611397544d30d972a5c4a9f90',
    '3a50657ded3fa3347eba237c720c0448bfd64d64f64a38366652ef60fd2c414b',
    '81091d2678d965872600d0569d6be9ad329db9ba814f17db9c3c863cc74fd134',
]


class ResidentModelStore(ModelStore):
    def __init__(self, config, output, *, loader, fallback_config, **kwargs):
        mode = config['inference_mode']
        if (mode not in ('historical297', 'probability_ensemble')
                or resolve_device(config) != 'cuda' or config['discard_postprocess']):
            raise ValueError('双策略常驻要求 CUDA、关闭后处理，以三模型或 historical297 启动')
        names = config.get('ensemble_onnx', NAMES)
        if names != NAMES or config.get('ensemble_sha256') not in (None, HASHES):
            raise ValueError('双策略常驻固定 r1-17/r1-12/r1-14 及其 SHA 顺序')
        if config.get('historical297_model', 'historical297.pt') != 'historical297.pt':
            raise ValueError('双策略常驻固定 historical297.pt')
        config = dict(config, device='cuda', ensemble_onnx=list(NAMES),
                      ensemble_sha256=list(HASHES), historical297_model='historical297.pt')
        self._resident = {}
        super().__init__(config, output, loader=loader, fallback_config=fallback_config, **kwargs)
        self._resident[mode] = (self._active, dict(config))
        other = 'probability_ensemble' if mode == 'historical297' else 'historical297'
        candidate = dict(config, inference_mode=other)
        try:
            # 在连接平台/接入比赛前完成两种真实 loader 的 SHA 校验及 CUDA 预热。
            model = loader(candidate)
            entry = dict(model=model, refs=0, retired=False)
            self._entries.append(entry)
            self._resident[other] = (entry, candidate)
            self.writer(self.output/'active_model.json', self._metadata(self.current, self._config))
        except BaseException:
            self.close()
            raise

    def _metadata(self, model, config, **kwargs):
        metadata = super()._metadata(model, config, **kwargs)
        metadata.update(resident_switch_enabled=True,
                        resident_ready=all(k in self._resident for k in ('historical297', 'probability_ensemble')),
                        control_poll_ms=20)
        return metadata

    def _activate(self, key):
        # 调用者持有 switch_lock；两条固定 GPU 策略全生命周期保留，不退休/卸载。
        with self._condition:
            if self._closing:
                raise RuntimeError('客户端正在退出')
            entry, saved = self._resident[key]
            config = dict(saved, mode=self._config.get('mode'))
            metadata = self._metadata(entry['model'], config, enabled=True)
            self.writer(self.output/'active_model.json', metadata)
            self._active, self._config = entry, config
            self.discard_postprocess = config['discard_postprocess']
            return metadata

    def switch_ensemble(self, names, hashes, device):
        if names != NAMES or hashes != HASHES or device != 'cuda':
            raise ValueError('快速切换仅接受已经常驻的固定三模型 CUDA 策略')
        with self._switch_lock:
            return self._activate('probability_ensemble')

    def switch_historical297(self, name, device):
        if name != 'historical297.pt' or device != 'cuda':
            raise ValueError('快速切换仅接受已经常驻的 historical297 CUDA 策略')
        with self._switch_lock:
            return self._activate('historical297')

    def fallback_cpu(self):
        # 保留原紧急 CPU 回退能力；首次 CPU 回退仍需加载，不属于快速双策略切换。
        with self._switch_lock:
            with self._condition:
                if self._closing:
                    raise RuntimeError('客户端正在退出')
            if 'fallback_cpu' not in self._resident:
                config = dict(self._fallback, mode=self._config.get('mode'))
                model = self.loader(config)
                entry = dict(model=model, refs=0, retired=False)
                self._entries.append(entry)
                self._resident['fallback_cpu'] = (entry, config)
            return self._activate('fallback_cpu')

    def _watch(self):
        while not self._stop.wait(.02):
            try:
                self.process_control()
            except FileNotFoundError:
                pass
            except Exception as error:
                print(f'快速热切换控制异常：{type(error).__name__}: {error}', file=sys.stderr, flush=True)
