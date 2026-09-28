"""用真实 observation 离线比较单模型、串行集成与并行集成，不连接平台。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from run import ROOT, load_configured_model
from ensemble_policy import DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True, help='包含 obs 数组的 npz')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ensemble-workers', type=int)
    parser.add_argument('--rooms', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/ensemble-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.rooms):
        parser.error('repeats 和 rooms 必须为正数')
    with np.load(args.observations, allow_pickle=False) as data:
        observations = data['obs'].copy()
    if not len(observations):
        raise ValueError('observation 不能为空')
    model = load_configured_model(dict(inference_mode='probability_ensemble',
        ensemble_onnx=DEFAULT_MODELS, ensemble_sha256=None, threads=1, discard_postprocess=False,
        device=args.device, ensemble_workers=args.ensemble_workers))
    p = model.policy
    try:
        reference = [p.probabilities(obs, parallel=False) for obs in observations]
        for obs, expected in zip(observations, reference):
            np.testing.assert_array_equal(p.probabilities(obs), expected)
        modes = dict(single=p.policies[0].predict,
                     serial=lambda obs: p.probabilities(obs, parallel=False).argmax(1),
                     parallel=p.predict)
        results = []
        for rooms in args.rooms:
            # 交替测量顺序，降低固定先后顺序造成的预热偏差。
            samples = {name: [] for name in modes}
            walls = {name: [] for name in modes}
            with ThreadPoolExecutor(max_workers=rooms) as executor:
                for repeat in range(args.repeats):
                    names = list(modes) if repeat % 2 == 0 else list(reversed(modes))
                    for name in names:
                        def timed(item):
                            index, obs = item
                            started = time.perf_counter()
                            action = modes[name](obs).item()
                            elapsed = (time.perf_counter() - started) * 1000
                            if name != 'single' and action != reference[index].argmax(1).item():
                                raise AssertionError('并发集成与串行参考动作不同')
                            return elapsed
                        started = time.perf_counter()
                        samples[name].extend(executor.map(timed, enumerate(observations)))
                        walls[name].append(time.perf_counter() - started)
            for name, values in samples.items():
                results.append(dict(rooms=rooms, mode=name, decisions=len(values),
                    median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                    decisions_per_second=len(values)/sum(walls[name])))
        report = dict(observations=len(observations), probability_arrays_exact_match=True,
                      manifest=model.manifest, results=results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        p.close()


if __name__ == '__main__':
    main()
