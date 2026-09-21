"""牌码与索引互转（杭州麻将 136 张，无花牌，白板 = 财神）。

索引约定：
    0-8   1w-9w
    9-17  1b-9b
    18-26 1t-9t
    27-32 东 南 西 北 中 发
    33    白（财神 / 百搭）
"""

SUITS = "wbt"
HONORS = ["东", "南", "西", "北", "中", "发"]
GOD = "白"
GOD_IDX = 33
TILE_KINDS = 34
TILES_PER_KIND = 4

ALL_TILES = (
    ["%d%s" % (n, s) for s in SUITS for n in range(1, 10)] + HONORS + [GOD]
)
_CODE_TO_IDX = {code: i for i, code in enumerate(ALL_TILES)}


def tile_index(code):
    """牌码 -> 0..33；非法牌码返回 None。"""
    return _CODE_TO_IDX.get(code)


def tile_code(idx):
    return ALL_TILES[idx]


def is_god(idx):
    return idx == GOD_IDX


def is_suited(idx):
    return 0 <= idx < 27


def suit_of(idx):
    """数牌返回 0/1/2，字牌返回 3，财神返回 4。"""
    if idx < 27:
        return idx // 9
    if idx < 33:
        return 3
    return 4


def rank_of(idx):
    """数牌返回 0..8（对应 1-9）；非数牌返回 -1。"""
    return idx % 9 if idx < 27 else -1


def to_counts(tiles):
    """牌码列表 -> 长度 34 的计数数组（下标 33 即财神数）。"""
    counts = [0] * TILE_KINDS
    for t in tiles:
        i = tile_index(t) if isinstance(t, str) else t
        if i is None:
            raise ValueError("非法牌码: %r" % (t,))
        counts[i] += 1
    return counts


def counts_to_tiles(counts):
    out = []
    for i, c in enumerate(counts):
        out.extend([i] * c)
    return out


def codes(indices):
    return [ALL_TILES[i] for i in indices]


def indices(codes_list):
    out = []
    for c in codes_list:
        i = tile_index(c)
        if i is None:
            raise ValueError("非法牌码: %r" % (c,))
        out.append(i)
    return out


def sort_key(idx):
    return idx


def format_hand(tiles):
    """tiles 可为索引或牌码列表，返回排序后的可读串。"""
    idxs = [t if isinstance(t, int) else tile_index(t) for t in tiles]
    return " ".join(ALL_TILES[i] for i in sorted(idxs))
