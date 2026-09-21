"""离散动作空间与合法动作掩码。

编号布局（共 74）：
    [ 0, 34)  discard 牌索引 t
    [34, 68)  gang   牌索引 t（an / bu / ming 由上下文唯一确定，不占额外编号）
    68        peng
    [69, 72)  chi 第 0/1/2 个可吃组合（组合顺序由 engine._chi_options 决定，稳定）
    72        hu
    73        pass

杠为何不区分模式：同一张牌在同一决策点上，暗杠(counts==4) / 补杠(counts==1 且已碰)
/ 明杠(响应窗口，牌 == last_discard) 三者互斥，牌索引即可唯一定位。
"""

DISCARD_OFFSET = 0
GANG_OFFSET = 34
PENG_ID = 68
CHI_OFFSET = 69
CHI_SLOTS = 3
HU_ID = 72
PASS_ID = 73
ACTION_N = 74

_KIND_ORDER = {"hu": 0, "gang": 1, "peng": 2, "chi": 3, "discard": 4,
               "pass": 5}


def to_id(action, actions):
    """引擎动作元组 -> 动作编号。chi 需要在 actions 中定位组合序号。"""
    kind = action[0]
    if kind == "discard":
        return DISCARD_OFFSET + action[1]
    if kind == "gang":
        return GANG_OFFSET + action[1]
    if kind == "peng":
        return PENG_ID
    if kind == "hu":
        return HU_ID
    if kind == "pass":
        return PASS_ID
    if kind == "chi":
        chis = [a for a in actions if a[0] == "chi"]
        for i, a in enumerate(chis):
            if a == action:
                return CHI_OFFSET + min(i, CHI_SLOTS - 1)
        return CHI_OFFSET
    raise ValueError("未知动作 %r" % (action,))


def from_id(aid, actions):
    """动作编号 -> 引擎动作元组；不在合法集合内返回 None。"""
    for a in actions:
        try:
            if to_id(a, actions) == aid:
                return a
        except ValueError:
            continue
    return None


def action_mask(actions):
    """长度 74 的 0/1 掩码。"""
    mask = [0] * ACTION_N
    for a in actions:
        mask[to_id(a, actions)] = 1
    return mask


def legal_ids(actions):
    return sorted({to_id(a, actions) for a in actions})


def describe_id(aid):
    from hzmj.tiles import tile_code
    if aid < GANG_OFFSET:
        return "打 " + tile_code(aid)
    if aid < PENG_ID:
        return "杠 " + tile_code(aid - GANG_OFFSET)
    if aid == PENG_ID:
        return "碰"
    if aid < HU_ID:
        return "吃#%d" % (aid - CHI_OFFSET)
    if aid == HU_ID:
        return "胡"
    return "过"


def sort_actions(actions):
    """按动作类别稳定排序，便于打印与调试。"""
    return sorted(actions, key=lambda a: (_KIND_ORDER.get(a[0], 9), a[1:]))
