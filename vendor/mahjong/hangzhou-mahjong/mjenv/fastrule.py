"""hu.py 热点的加速版（纯标准库，不修改 hzmj）。

原实现的瓶颈不在花色内 DFS（`_group_profile` 已按 9/6 位分组缓存，命中率高），
而在两处：

  1. `_standard_missing` 的 lru_cache 键是完整 34 元组，状态空间过大，命中率低；
     每次 miss 都要重跑 4 组切片 + dict 版 dp 合并。
  2. `useful_tiles` / `is_baotou` / `winning_tiles` / `best_discard` 都是
     「对 34 张牌逐个改一张再全量重算」，而改一张只影响一个花色组，
     另外三组的 profile 完全可以复用。

本模块据此做两件事：

  * `_combine` 把 dp 合并的缓存键换成「4 个 profile + 需要的面子数」，
    profile 状态空间远小于手牌状态空间，命中率大幅提升，且 dp 用扁平数组。
  * `missing_all_discards` / `missing_all_draws` 一次性算出 34 种变体，
    只重算受影响的那一组 profile。

`patch()` 把加速版注入 hzmj 各模块的命名空间（它们用的是 from-import，
必须逐个模块替换），`unpatch()` 可完全还原。
"""

from functools import lru_cache

from hzmj import adapter as _adapter
from hzmj import bots as _bots
from hzmj import engine as _engine
from hzmj import fan as _fan
from hzmj import hu as _hu
from hzmj.hu import _group_profile, hand_size, qidui_missing
from hzmj.tiles import GOD_IDX, TILE_KINDS

_INF = 99

# (起始下标, 结束下标, 是否字牌)
_GROUPS = ((0, 9, False), (9, 18, False), (18, 27, False), (27, 33, True))
_GROUP_OF = [0] * 9 + [1] * 9 + [2] * 9 + [3] * 6 + [4]


@lru_cache(maxsize=1 << 17)
def _combine(p0, p1, p2, p3, sets_need):
    """4 组 profile 合成整手缺口（未扣财神）。

    dp 扁平索引 = 已用面子数 * 2 + 已用将数。

    注：试过「预折叠不变的三组 + 逐张只并入变动那一组」的增量写法，
    实测反而慢一半 —— 批量变体里 4 组 profile 的组合重复率极高，
    这份 lru_cache 的命中才是真正的加速来源，增量写法等于绕开了它。
    """
    width = (sets_need + 1) * 2
    dp = [_INF] * width
    dp[0] = 0
    for prof in (p0, p1, p2, p3):
        nxt = [_INF] * width
        for used_s in range(sets_need + 1):
            row = used_s * 2
            for used_p in range(2):
                cost = dp[row + used_p]
                if cost >= _INF:
                    continue
                for m in range(sets_need - used_s + 1):
                    slot = (used_s + m) * 2
                    v = cost + prof[m * 2]
                    if v < nxt[slot + used_p]:
                        nxt[slot + used_p] = v
                    if used_p == 0:
                        v2 = cost + prof[m * 2 + 1]
                        if v2 < nxt[slot + 1]:
                            nxt[slot + 1] = v2
        dp = nxt
    return dp[sets_need * 2 + 1]


def _profiles(counts):
    return (_group_profile(tuple(counts[0:9]), False),
            _group_profile(tuple(counts[9:18]), False),
            _group_profile(tuple(counts[18:27]), False),
            _group_profile(tuple(counts[27:33]), True))


def standard_missing(counts, meld_count=0):
    sets_need = 4 - meld_count
    if sets_need < 0:
        return _INF
    p0, p1, p2, p3 = _profiles(counts)
    return max(0, _combine(p0, p1, p2, p3, sets_need) - counts[GOD_IDX])


def missing(counts, meld_count=0):
    q = qidui_missing(counts, meld_count)
    s = standard_missing(counts, meld_count)
    return s if s < q else q


def shanten(counts, meld_count=0):
    return missing(counts, meld_count) - 1


# ---------------- 批量变体 ----------------

def _variants(counts, meld_count, delta):
    """对 34 种牌各做 counts[t] += delta 后的 missing。

    delta = -1 打牌（手上没有该牌 -> None）
    delta = +1 摸牌（已有 4 张 -> None）
    """
    sets_need = 4 - meld_count
    out = [None] * TILE_KINDS
    if sets_need < 0:
        return out
    base = _profiles(counts)
    god = counts[GOD_IDX]
    raw_base = _combine(base[0], base[1], base[2], base[3], sets_need)
    probe = list(counts)
    limit = 1 if delta < 0 else 4
    for t in range(TILE_KINDS):
        c = counts[t]
        if (delta < 0 and c < limit) or (delta > 0 and c >= limit):
            continue
        if t == GOD_IDX:
            std = max(0, raw_base - (god + delta))
        else:
            g = _GROUP_OF[t]
            lo, hi, honor = _GROUPS[g]
            seg = list(counts[lo:hi])
            seg[t - lo] += delta
            args = list(base)
            args[g] = _group_profile(tuple(seg), honor)
            std = max(0, _combine(args[0], args[1], args[2], args[3],
                                  sets_need) - god)
        probe[t] += delta
        q = qidui_missing(probe, meld_count)
        probe[t] -= delta
        out[t] = std if std < q else q
    return out


def missing_all_discards(counts, meld_count=0):
    return _variants(counts, meld_count, -1)


def missing_all_draws(counts, meld_count=0):
    return _variants(counts, meld_count, 1)


# ---------------- 基于批量的派生判定 ----------------

def winning_tiles(counts, meld_count=0):
    if sum(counts) != hand_size(meld_count, drawn=False):
        return set()
    if missing(counts, meld_count) > 1:      # 与 hu.winning_tiles 同步的剪枝
        return set()
    draws = missing_all_draws(counts, meld_count)
    return {t for t, m in enumerate(draws) if m == 0}


def is_tenpai(counts, meld_count=0):
    return bool(winning_tiles(counts, meld_count))


def is_baotou(counts, meld_count=0):
    # 与 hu.is_baotou 同步：平台实测 4 白板同样判爆头，且与 4白板×2 叠加
    if sum(counts) != hand_size(meld_count, drawn=False):
        return False
    if missing(counts, meld_count) != 1:
        return False
    draws = missing_all_draws(counts, meld_count)
    for t in range(TILE_KINDS):
        if counts[t] >= 4:
            continue
        if draws[t] != 0:
            return False
    return True


def useful_tiles(counts, meld_count=0):
    base = missing(counts, meld_count)
    if base == 0:
        return {}
    draws = missing_all_draws(counts, meld_count)
    return {t: m for t, m in enumerate(draws) if m is not None and m < base}


def can_hu(counts, meld_count=0, you_cai_bi_kao=False, baotou=False):
    if sum(counts) != hand_size(meld_count, drawn=True):
        return False
    if missing(counts, meld_count) != 0:
        return False
    if you_cai_bi_kao and counts[GOD_IDX] > 0 and not baotou:
        return False
    return True


# ---------------- 出牌选择（语义与 hzmj.bots.best_discard 完全一致） ----------------

def best_discard(game, seat, discards, keep_god_bonus=True):
    p = game.players[seat]
    table = missing_all_discards(p.counts, p.meld_count)
    best_m = None
    for a in discards:
        m = table[a[1]]
        if m is None:
            continue
        if best_m is None or m < best_m:
            best_m = m
    finalists = [a for a in discards if table[a[1]] == best_m]
    if not finalists:
        return discards[0]
    if len(finalists) == 1:
        return finalists[0]

    seen = _bots.seen_counts(game, seat)
    remain = [max(0, 4 - seen[t]) for t in range(TILE_KINDS)]
    best, best_key = None, None
    for a in finalists:
        t = a[1]
        p.counts[t] -= 1
        draws = missing_all_draws(p.counts, p.meld_count)
        gain = sum(remain[x] for x, m in enumerate(draws)
                   if m is not None and m < best_m)
        baotou_bonus = 1 if _baotou_from_draws(
            p.counts, p.meld_count, best_m, draws) else 0
        p.counts[t] += 1
        god_penalty = 1 if (keep_god_bonus and t == GOD_IDX) else 0
        key = (-baotou_bonus, god_penalty, -gain, -t)
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


def _baotou_from_draws(counts, meld_count, base_missing, draws):
    """复用已算好的 draws 表判爆头，省掉一次 34 连算。"""
    if base_missing != 1:
        return False
    if sum(counts) != hand_size(meld_count, drawn=False):
        return False
    for t in range(TILE_KINDS):
        if counts[t] >= 4:
            continue
        if draws[t] != 0:
            return False
    return True


# ---------------- 注入 / 还原 ----------------

_FAST = {
    "missing": missing,
    "standard_missing": standard_missing,
    "shanten": shanten,
    "winning_tiles": winning_tiles,
    "is_tenpai": is_tenpai,
    "is_baotou": is_baotou,
    "useful_tiles": useful_tiles,
    "can_hu": can_hu,
}

# hzmj 各模块用的是 from-import，必须逐个模块替换其命名空间
_TARGETS = (_hu, _engine, _bots, _adapter, _fan)
_ORIGINAL = {}
_PATCHED = [False]


def patch():
    """把加速版注入 hzmj。幂等。"""
    if _PATCHED[0]:
        return
    for mod in _TARGETS:
        for name, fn in _FAST.items():
            if hasattr(mod, name):
                _ORIGINAL[(mod.__name__, name)] = getattr(mod, name)
                setattr(mod, name, fn)
    _ORIGINAL[("hzmj.bots", "best_discard")] = _bots.best_discard
    _bots.best_discard = best_discard
    _PATCHED[0] = True


def unpatch():
    if not _PATCHED[0]:
        return
    for (modname, name), fn in _ORIGINAL.items():
        for mod in _TARGETS:
            if mod.__name__ == modname:
                setattr(mod, name, fn)
    _ORIGINAL.clear()
    _PATCHED[0] = False


def is_patched():
    return _PATCHED[0]
