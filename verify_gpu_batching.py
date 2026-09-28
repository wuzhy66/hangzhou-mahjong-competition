"""离线检查 GPU 合批与 CPU 动作一致性、延迟和持续并发稳定性。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import gzip
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from run import ROOT, load_configured_model
from ensemble_policy import DEFAULT_MODELS, ProbabilityEnsemble
from batched_ensemble import BatchedProbabilityEnsemble
from mahjong.platform.decision import decide


def gpu_status():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True, help='包含 obs 数组的 npz')
    parser.add_argument('--fixtures', type=Path, help='可选：含 cases 的决策快照 json.gz')
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/ensemble-gpu-batching.json')
    parser.add_argument('--wait-ms', type=float, default=20.0)
    parser.add_argument('--baseline-wait-ms', type=float, help='指定后对照另一凑批窗口，否则对照逐桌推理')
    parser.add_argument('--rooms', type=int, nargs='+', default=[1, 4, 10, 16])
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or any(room < 1 for room in args.rooms):
        parser.error('repeats 和 rooms 必须为正整数')
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration 必须是有限正数')
    with np.load(args.observations, allow_pickle=False) as data:
        observations = list(data['obs'].copy())
    cases = []
    if args.fixtures:
        with gzip.open(args.fixtures, 'rt', encoding='utf-8') as stream:
            cases = json.load(stream)['cases']
        observations.extend(np.asarray(item['observation'], dtype=np.float32) for item in cases)
    observations = np.unique(np.asarray(observations), axis=0)
    if not len(observations):
        raise ValueError('observation 不能为空')
    config = dict(device='cpu', inference_mode='probability_ensemble', ensemble_onnx=DEFAULT_MODELS,
                  ensemble_sha256=None, ensemble_workers=3, threads=1, discard_postprocess=False)
    cpu = load_configured_model(config).policy
    try:
        reference = [cpu.probabilities(obs, parallel=False) for obs in observations]
        payloads = [decide(cpu, c['case']['snapshot'], c['case']['ycbk'], c['case']['events']) for c in cases]
    finally:
        cpu.close()
    config.update(device='cuda', gpu_batch_size=16, gpu_batch_wait_ms=args.wait_ms)
    model = load_configured_model(config)
    p = model.policy
    # 复用相同 GPU Session，对照之前的逐桌 batch=1 路径；两模式串行测量。
    baseline = (ProbabilityEnsemble(p.policies, workers=3) if args.baseline_wait_ms is None
                else BatchedProbabilityEnsemble(p.policies, workers=3, max_wait_ms=args.baseline_wait_ms))
    baseline_name = 'unbatched' if args.baseline_wait_ms is None else f'batch_wait_{args.baseline_wait_ms:g}ms'
    candidate_name = f'batch_wait_{args.wait_ms:g}ms'
    max_error = 0.0
    try:
        for size in range(1, 17):
            for start in range(0, len(observations), size):
                actual = p.probabilities(observations[start:start+size])
                expected = np.concatenate(reference[start:start+size])
                np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        for item, expected in zip(cases, payloads):
            c = item['case']
            assert decide(p, c['snapshot'], c['ycbk'], c['events']) == expected
        print(json.dumps(dict(preflight='passed', observations=len(observations), payloads=len(cases),
                              max_probability_sum_error=max_error, gpu=gpu_status())), flush=True)
        results = []
        for rooms in args.rooms:
            modes = {baseline_name: baseline, candidate_name: p}
            samples = {name: [] for name in modes}
            walls = {name: 0.0 for name in modes}
            cpus = {name: 0.0 for name in modes}
            with ThreadPoolExecutor(max_workers=rooms) as executor:
                for repeat in range(args.repeats):
                    for name in (list(modes) if repeat % 2 == 0 else list(reversed(modes))):
                        def timed(index):
                            started = time.perf_counter()
                            actual = modes[name].probabilities(observations[index])
                            elapsed = (time.perf_counter()-started)*1000
                            np.testing.assert_array_equal(actual.argmax(1), reference[index].argmax(1))
                            return elapsed
                        cpu0, started = time.process_time(), time.perf_counter()
                        samples[name].extend(executor.map(timed, range(len(observations))))
                        walls[name] += time.perf_counter()-started
                        cpus[name] += time.process_time()-cpu0
            for name in modes:
                row = dict(rooms=rooms, mode=name, decisions=len(samples[name]),
                           median_ms=float(np.median(samples[name])), p95_ms=float(np.percentile(samples[name], 95)),
                           decisions_per_second=len(samples[name])/walls[name], cpu_core_equivalents=cpus[name]/walls[name])
                results.append(row)
                print(json.dumps(row), flush=True)
        start, cpu0 = time.perf_counter(), time.process_time()
        def soak(room):
            index = room
            samples = []
            while time.perf_counter()-start < args.duration:
                tick = time.perf_counter()
                actual = p.probabilities(observations[index % len(observations)])
                elapsed = (time.perf_counter()-tick)*1000
                expected = reference[index % len(reference)]
                np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                samples.append(elapsed)
                index += 1
            return samples
        snapshots = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(soak, room) for room in range(16)]
            while not all(f.done() for f in futures):
                wait(futures, timeout=15)
                row = dict(seconds=round(time.perf_counter()-start, 1), gpu=gpu_status(), batching=p.stats())
                snapshots.append(row)
                print(json.dumps(row), flush=True)
            samples = [value for future in futures for value in future.result()]
        duration = time.perf_counter()-start
        report = dict(observations=len(observations), payloads=len(cases), tested_batch_sizes=list(range(1,17)),
                      cpu_gpu_action_mismatches=0, max_probability_sum_error=max_error, manifest=model.manifest,
                      performance=results, baseline_batching=baseline.stats() if args.baseline_wait_ms is not None else None, batching=p.stats(), soak=dict(rooms=16, duration_s=duration,
                      decisions=len(samples), median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples,95)),
                      cpu_core_equivalents=(time.process_time()-cpu0)/duration, mismatches=0, samples=snapshots))
    finally:
        baseline.close()
        p.close()
    report['worker_threads_after_close'] = [t.name for t in threading.enumerate()
        if t.name.startswith(('onnx-ensemble', 'onnx-batch-dispatch'))]
    assert not report['worker_threads_after_close']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report['soak'].items() if k!='samples'}), flush=True)


if __name__ == '__main__':
    main()
