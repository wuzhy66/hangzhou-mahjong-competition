"""胡牌判定 / 向听数 / 听牌 / 爆头（杭州麻将：白板 = 财神百搭）。

核心概念 —— **缺口数 missing**：把当前手牌补成合法胡牌型，还需要摸进的牌张数。
    missing == 0  已经是胡牌型
    missing == 1  听牌（再摸 1 张即可胡）
    向听数 shanten = missing - 1

财神（白板）作为通配符，先用于填补缺口：
    missing = max(0, 实牌缺口 - 财神张数)

规则约束：
    普通胡  4 组面子 + 1 对将；吃碰杠折算面子
    七对子  7 个对子，禁止任何副露；4 张同种算 2 对（豪华七对）
"""

from functools import lru_cache

from .tiles import GOD_IDX, TILE_KINDS

_INF = 99


@lru_cache(maxsize=1 << 20)
def _dfs(counts, pos, sets_need, pair_need, honor):
    """单组（一个花色 9 位 / 字牌 6 位）内的最小缺口数。

    缺口 = 凑齐 sets_need 个面子与 pair_need 个将还差的牌张数；
    该组牌不足时允许「空面子」（缺 3）与「空将」（缺 2）。
    """
    if sets_need == 0 and pair_need == 0:
        return 0
    size = len(counts)
    while pos < size and counts[pos] == 0:
        pos += 1
    if pos >= size:
        return sets_need * 3 + pair_need * 2

    lst = list(counts)
    n = lst[pos]

    # 分支 1：本位置的牌全部弃用
    lst[pos] = 0
    best = _dfs(tuple(lst), pos + 1, sets_need, pair_need, honor)
    lst[pos] = n

    # 分支 2：将（对子），用 k 张实牌
    if pair_need:
        for k in (2, 1):
            if n >= k:
                lst[pos] = n - k
                r = _dfs(tuple(lst), pos, sets_need, 0, honor) + (2 - k)
                lst[pos] = n
                if r < best:
                    best = r

    if sets_need:
        # 分支 3：刻子
        for k in (3, 2, 1):
            if n >= k:
                lst[pos] = n - k
                r = _dfs(tuple(lst), pos, sets_need - 1, pair_need, honor) \
                    + (3 - k)
                lst[pos] = n
                if r < best:
                    best = r
        # 分支 4：顺子，起手位置 == pos
        if not honor and pos <= size - 3:
            for use_b in (1, 0):
                for use_c in (1, 0):
                    if use_b and lst[pos + 1] == 0:
                        continue
                    if use_c and lst[pos + 2] == 0:
                        continue
                    lst[pos] = n - 1
                    if use_b:
                        lst[pos + 1] -= 1
                    if use_c:
                        lst[pos + 2] -= 1
                    r = _dfs(tuple(lst), pos, sets_need - 1, pair_need,
                             honor) + (2 - use_b - use_c)
                    lst[pos] = n
                    if use_b:
                        lst[pos + 1] += 1
                    if use_c:
                        lst[pos + 2] += 1
                    if r < best:
                        best = r
        # 分支 5：顺子，起手位置 == pos - 1（即缺最低位那张，如 89 听 7）
        #
        # 扫描指针只前进、且经过的位置都已清零，因此 pos 以下必然无牌；
        # 分支 4 只枚举「起手位就是 pos」的顺子，这类缺低位的搭子永远进不了
        # 搜索空间，导致缺口数偏大 1。起手位再往下（pos - 2）时组内实牌
        # 最多 1 张，代价 >= 2，不优于刻子分支，无需枚举。
        if not honor and pos >= 1 and pos + 1 < size and lst[pos + 1] > 0:
            lst[pos] = n - 1
            lst[pos + 1] -= 1
            r = _dfs(tuple(lst), pos, sets_need - 1, pair_need, honor) + 1
            lst[pos] = n
            lst[pos + 1] += 1
            if r < best:
                best = r
    return best


@lru_cache(maxsize=1 << 16)
def _group_profile(counts, honor):
    """该组对 (面子数 m, 将数 p) 的缺口表，扁平索引 m * 2 + p。"""
    return tuple(_dfs(counts, 0, m, p, honor)
                 for m in range(5) for p in range(2))


@lru_cache(maxsize=1 << 18)
def _standard_missing(counts, meld_count):
    sets_need = 4 - meld_count
    if sets_need < 0:
        return _INF
    groups = (
        _group_profile(counts[0:9], False),
        _group_profile(counts[9:18], False),
        _group_profile(counts[18:27], False),
        _group_profile(counts[27:33], True),
    )
    # dp[(已用面子, 已用将)] = 最小缺口
    dp = {(0, 0): 0}
    for prof in groups:
        nxt = {}
        for (used_s, used_p), cost in dp.items():
            for m in range(sets_need - used_s + 1):
                for p in range(2 - used_p):
                    add = prof[m * 2 + p]
                    key = (used_s + m, used_p + p)
                    val = cost + add
                    if nxt.get(key, _INF) > val:
                        nxt[key] = val
        dp = nxt
    raw = dp.get((sets_need, 1), _INF)
    return max(0, raw - counts[GOD_IDX])


def standard_missing(counts, meld_count=0):
    """普通胡（4 面子 + 1 将）的缺口数，counts 长度 34（含财神）。"""
    return _standard_missing(tuple(counts), meld_count)


def qidui_missing(counts, meld_count=0):
    """七对子缺口数；有任何副露直接判不可能。"""
    if meld_count:
        return _INF
    jokers = counts[GOD_IDX]
    pairs = 0
    singles = 0
    for c in counts[:33]:
        pairs += c // 2
        singles += c % 2
    if pairs >= 7:
        return 0
    need = 7 - pairs

    # 财神优先与单张配对（1 财神 + 1 单张 = 1 对）
    x = min(need, singles, jokers)
    need -= x
    singles -= x
    jokers -= x
    # 其次两张财神自成一对
    y = min(need, jokers // 2)
    need -= y
    jokers -= 2 * y
    if need == 0:
        return 0
    # 剩余对子：每对若能贴 1 张现成牌则缺 1，否则缺 2
    avail = singles + jokers
    covered = min(need, avail)
    return covered + (need - covered) * 2


def missing(counts, meld_count=0):
    """胡牌缺口数：普通胡与七对取较小者。"""
    return min(standard_missing(counts, meld_count),
               qidui_missing(counts, meld_count))


def shanten(counts, meld_count=0):
    """向听数：0 = 听牌，-1 = 已胡。"""
    return missing(counts, meld_count) - 1


def hand_size(meld_count, drawn=False):
    """合法手牌张数：未摸牌 13-3n，摸牌后 14-3n。"""
    return (14 if drawn else 13) - 3 * meld_count


def can_hu(counts, meld_count=0, you_cai_bi_kao=False, baotou=False):
    """判胡。counts 须为摸牌后的完整手牌（14 - 3*meld_count 张）。

    you_cai_bi_kao：有财必拷响 —— 手上有财神时不允许平胡，须爆头/杠开。
    baotou：当前是否爆头态（含杠开等链上状态，由调用方传入）。
    """
    total = sum(counts)
    if total != hand_size(meld_count, drawn=True):
        return False
    if missing(counts, meld_count) != 0:
        return False
    if you_cai_bi_kao and counts[GOD_IDX] > 0 and not baotou:
        return False
    return True


def winning_tiles(counts, meld_count=0):
    """13 张（折算后）听哪些牌：返回可胡的牌索引集合。"""
    if sum(counts) != hand_size(meld_count, drawn=False):
        return set()
    if missing(counts, meld_count) > 1:
        return set()
    out = set()
    probe = list(counts)
    for t in range(TILE_KINDS):
        if probe[t] >= 4:
            continue
        probe[t] += 1
        if missing(probe, meld_count) == 0:
            out.add(t)
        probe[t] -= 1
    return out


def is_tenpai(counts, meld_count=0):
    return bool(winning_tiles(counts, meld_count))


def is_baotou(counts, meld_count=0):
    """爆头：听牌态摸任意 1 张牌上来即胡。

    平台规则：财神数不限、支持副露。已在手上凑满 4 张的牌不可能再摸到，
    判定时跳过。

    ⚠️ 指南 §1.2 字面写「正好 4 张白板不视为爆头」、§1.3 又写爆头
    「正好 4 白板除外」，**但平台实现并非如此**。对 `fan-calc` 实测：

        白白白白 1w1w1w 2w2w2w 3w3w3w + 9w -> baotou=true, 七对,4个白板,爆头
        白白白白 1w2w3w4w5w6w7w8w9w + 5b -> baotou=true, 平胡,4个白板,爆头  fan=4

    即「4 白板 ×2」与「爆头 ×2」**叠加**，手持 4 张白板同样判爆头。
    曾按字面加 `if counts[GOD_IDX] == 4: return False`，实测少算 ×2 番。
    """
    if sum(counts) != hand_size(meld_count, drawn=False):
        return False
    if missing(counts, meld_count) != 1:
        return False
    probe = list(counts)
    for t in range(TILE_KINDS):
        if probe[t] >= 4:
            continue
        probe[t] += 1
        ok = missing(probe, meld_count) == 0
        probe[t] -= 1
        if not ok:
            return False
    return True


def useful_tiles(counts, meld_count=0):
    """进张：摸哪些牌能让缺口数下降。返回 {牌索引: 下降后的缺口}。"""
    base = missing(counts, meld_count)
    if base == 0:
        return {}
    out = {}
    probe = list(counts)
    for t in range(TILE_KINDS):
        if probe[t] >= 4:
            continue
        probe[t] += 1
        m = missing(probe, meld_count)
        probe[t] -= 1
        if m < base:
            out[t] = m
    return out


def four_kind_groups(counts):
    """七对牌型里「四张一组」的组数（用于豪华七对分支）。"""
    return sum(1 for c in counts if c == 4)
