"""仅依赖 NumPy/ONNX Runtime 的麻将 Actor 推理入口。"""
import argparse
import importlib.util
import json
from pathlib import Path
import threading

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    """Session 只创建一次；predict 返回 action ID，scores 返回 masked logits。

    CUDA Graph 使用固定 batch、设备 buffer 和唯一调用线程；普通 CUDA 支持并发。
    """
    @classmethod
    def from_directory(cls, directory, *, device='cpu', **overrides):
        """读取 verify_benchmark 生成的本机最快且动作一致的配置。"""
        directory = Path(directory)
        profiles = json.loads((directory/'deployment.json').read_text())
        if device not in profiles:
            raise ValueError(f'该目录没有经过验证的 {device} 配置')
        profile = dict(profiles[device])
        model = profile.pop('model')
        profile.update(overrides)
        return cls(directory/model, device=device, **profile)

    def __init__(self, model, *, device='cpu', threads=1, cuda_graph=False,
                 batch_size=1, gpu_memory_mb=2048, spin_wait=False,
                 conv_max_workspace=False, conv_nc1d=False):
        if device not in ('cpu', 'cuda') or threads < 1 or batch_size < 1:
            raise ValueError('device 必须为 cpu/cuda，threads/batch_size 必须为正数')
        if cuda_graph and device != 'cuda':
            raise ValueError('CUDA Graph 只能用于 cuda')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 多个对局实例并行时不让空闲线程持续忙等占满 CPU。
        options.add_session_config_entry('session.intra_op.allow_spinning', str(int(spin_wait)))
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            # 可从已有 PyTorch 或 nvidia-* runtime 包加载 CUDA/cuDNN，无需导入 Ray。
            # 兼容虚拟环境复用其他 site-packages：ORT 默认只查自身相邻的 nvidia/。
            if importlib.util.find_spec('torch') is not None:
                import torch  # noqa: F401，仅预载已有 CUDA 库，不执行 PyTorch 推理
            ort.preload_dlls()
            providers = [('CUDAExecutionProvider', {
                'enable_cuda_graph': str(int(cuda_graph)), 'use_tf32': '0',
                'gpu_mem_limit': str(gpu_memory_mb * 1024**2),
                'cudnn_conv_algo_search': 'HEURISTIC',
                'cudnn_conv_use_max_workspace': str(int(conv_max_workspace)),
                'cudnn_conv1d_pad_to_nc1d': str(int(conv_nc1d)),
            }), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        if device == 'cuda' and self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider 未启动，请检查 CUDA/cuDNN；不静默降级为 CPU')
        inputs = self.session.get_inputs()
        outputs = {o.name: o for o in self.session.get_outputs()}
        if len(inputs) != 1 or inputs[0].name != 'obs' or inputs[0].type != 'tensor(float)' or set(outputs) != {'logits', 'actions'}:
            raise ValueError('模型不符合 export_bc 的 obs/logits/actions 接口')
        self.dim = inputs[0].shape[1]
        self.model_batch = inputs[0].shape[0]
        self.action_count = outputs['logits'].shape[1]
        self.cuda_graph, self.batch_size = cuda_graph, batch_size
        self.lock = threading.Lock()
        self._graph_thread_id = None
        if cuda_graph:
            self.binding = self.session.io_binding()
            self.input_buffer = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, self.dim), np.float32, 'cuda', 0)
            self.output_buffers = {
                'logits': ort.OrtValue.ortvalue_from_shape_and_type((batch_size, self.action_count), np.float32, 'cuda', 0),
                'actions': ort.OrtValue.ortvalue_from_shape_and_type((batch_size,), np.int64, 'cuda', 0),
            }
            self.binding.bind_ortvalue_input('obs', self.input_buffer)
            for name, value in self.output_buffers.items():
                self.binding.bind_ortvalue_output(name, value)

    def _input(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.ndim != 2 or obs.shape[1] != self.dim or not len(obs):
            raise ValueError(f'观测必须为 [{self.dim}] 或 [batch,{self.dim}]')
        if isinstance(self.model_batch, int) and len(obs) != self.model_batch:
            raise ValueError(f'此模型固定 batch={self.model_batch}；批量推理请选择动态 batch 导出')
        mask = obs[:, -self.action_count:]
        if not np.isfinite(obs).all() or not ((mask == 0) | (mask == 1)).all() or not (mask.sum(-1) > 0).all():
            raise ValueError('观测须为有限值，每行 action mask 须为 0/1 且至少一个合法动作；终局无需推理')
        if self.cuda_graph and len(obs) != self.batch_size:
            raise ValueError(f'此 CUDA Graph 实例固定 batch={self.batch_size}')
        return np.ascontiguousarray(obs)

    def _run(self, observations, output):
        obs = self._input(observations)
        if not self.cuda_graph:
            return self.session.run([output], {'obs': obs})[0]
        with self.lock:
            thread_id = threading.get_ident()
            if self._graph_thread_id is None:
                self._graph_thread_id = thread_id
            elif self._graph_thread_id != thread_id:
                raise RuntimeError('CUDA Graph 必须由同一线程执行；多线程服务请设置 cuda_graph=False')
            self.input_buffer.update_inplace(obs)
            self.session.run_with_iobinding(self.binding)
            # predict 只取回少量 action ID，不拷贝完整 logits。
            return self.output_buffers[output].numpy()

    def predict(self, observations):
        return self._run(observations, 'actions')

    def scores(self, observations):
        return self._run(observations, 'logits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('observations', type=Path, help='float32 .npy 文件')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--cuda-graph', action='store_true', default=None)
    parser.add_argument('--spin-wait', action='store_true', default=None)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in ('threads', 'cuda_graph', 'spin_wait', 'batch_size')
                 if getattr(args, k) is not None}
    policy = (OnnxPolicy.from_directory(args.model, device=args.device, **overrides) if args.model.is_dir()
              else OnnxPolicy(args.model, device=args.device, **overrides))
    print(policy.predict(np.load(args.observations, allow_pickle=False)).tolist())


if __name__ == '__main__':
    main()
