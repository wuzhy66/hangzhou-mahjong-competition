"""ONNX 弃牌后处理的一层进张树，仅接收自手与去重公开剩余计数。"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from mahjong.rule_ai.policy import value, native

PACKAGE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def native_search():
    source = PACKAGE / 'discard_search_native.cpp'
    base = PACKAGE.parent / 'rule_ai' / 'search_native.cpp'
    digest = hashlib.sha256(source.read_bytes() + base.read_bytes()).hexdigest()[:16]
    target = PACKAGE / '.build' / f'discard_search_{digest}.so'
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='discard_', suffix='.so', dir=target.parent)
        os.close(fd)
        try:
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-pthread',
                            str(source), '-o', temporary], check=True, capture_output=True)
            os.replace(temporary, target)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('弃牌搜索编译失败；启用后处理需要 Linux/WSL 和 g++') from exc
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.rb_shape_one.argtypes = [ptr, ptr] + [ctypes.c_int]*4 + [ctypes.POINTER(ctypes.c_double), ptr]
    lib.rb_shape_one.restype = None
    return lib


def warmup():
    """匹配对局前完成 native 加载，默认关闭时不调用。"""
    native_search()
    native()


def feature(counts, remaining, melds, action, ycbk, horizon=2):
    """单个候选的完整摸牌分支；已弃出的已知牌不回到未知池。"""
    if horizon not in (1, 2):
        raise ValueError('前瞻 horizon 必须为 1 或 2')
    if (len(counts) != 34 or len(remaining) != 34 or not 0 <= melds <= 4
            or not 0 <= action < 33 or sum(counts) != 14-3*melds
            or counts[action] <= 0 or sum(remaining) <= 1
            or any(not isinstance(n, int) or not isinstance(q, int)
                   or n < 0 or q < 0 or n+q > 4 for n, q in zip(counts, remaining))):
        raise ValueError('前瞻搜索手牌、副露或公开剩余计数不合法')
    out = (ctypes.c_double*8)()
    raw = (ctypes.c_int*(34*5))()
    native_search().rb_shape_one(
        (ctypes.c_int*34)(*counts), (ctypes.c_int*34)(*remaining),
        melds, action, int(ycbk), horizon, out, raw)
    branches = [dict(draw=tile, weight=raw[tile*5], discard=raw[tile*5+1],
                     gap=raw[tile*5+2], ukeire=raw[tile*5+3], win=bool(raw[tile*5+4]))
                for tile in range(34) if raw[tile*5]]
    if sum(branch['weight'] for branch in branches) != sum(remaining):
        raise RuntimeError('前瞻分支未覆盖全部公开未知牌权重')
    return dict(action=action, gap=int(out[0]), ukeire=int(out[1]), horizon=horizon,
                expected_gap_after_horizon=out[2], immediate_win_probability=out[3],
                expected_leaf_gap=out[4], expected_leaf_ukeire=out[5],
                unknown_total=int(out[6]), quality_numerator=int(out[7]), branches=branches)


def choose(features):
    """要求当前缺口和 DP 不退步；同等前瞻指标沿用模型排序。"""
    original = features[0]
    valid = [candidate for candidate in features[1:]
             if candidate['gap'] <= original['gap']
             and candidate['dp_score']+1e-10 >= original['dp_score']
             and candidate['quality_numerator'] < original['quality_numerator']]
    # 所有 root 候选共用分母；整数比较避免浮点误差影响决策。
    return min(valid, key=lambda candidate: candidate['quality_numerator']) if valid else original


def evaluate(job):
    counts, remaining, melds, choices, depth, unit, ycbk = job
    features = []
    for action in choices:
        candidate = feature(counts, remaining, melds, action, ycbk, depth)
        after = counts.copy()
        after[action] -= 1
        estimate = value(after, remaining, melds, depth, unit=unit, bonus=0., ycbk=ycbk)
        candidate.update(dp_score=estimate.score, dp_win=estimate.win, nodes=estimate.nodes)
        features.append(candidate)
    best = choose(features)
    changed = best is not features[0]
    return dict(status='changed' if changed else 'kept', changed=changed,
                final_action=best['action'], features=features)
