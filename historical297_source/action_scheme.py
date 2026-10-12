"""项目层 41 动作方案：把引擎 74 制动作元组重映射为紧凑 41 动作。

编号布局（共 41）：
    [ 0, 34)  弃/自杠合并：**自己回合**该牌可暗杠(counts==4)/补杠(counts==1 且已碰)
              则优先杠，否则打出该牌
    34        碰 peng
    [35, 38)  吃 chi 第 0/1/2 个组合（顺序由 engine 给出的 actions 决定，稳定）
    38        胡 hu
    39        明杠 gang（**响应**他家弃牌，不在自己回合）
    40        pass

设计要点
--------
本模块只在 `mahjong/` 层做重映射，**不修改子模块引擎**。引擎内部仍用 74 制动作
元组 `("discard",t) / ("gang",t) / ("peng",) / ("chi",...) / ("hu",) / ("pass",)`；
env 的 action_space、观测尾部 mask 与策略输出统一为 41。

自己回合 vs 响应窗口的判定：**合法集合里是否存在 discard**。
引擎保证响应窗口永远没有 discard 动作，自己回合永远至少有一个 discard，因此：
    ("gang",t) 在自己回合 = 暗杠/补杠 -> 槽 t（与弃牌共享同一槽）
    ("gang",t) 在响应窗口 = 明杠       -> 槽 39

弃/自杠同槽冲突由「优先杠」解决：`from_id(t)` 若该牌存在自杠则返回杠，否则返回弃牌。
**代价**：能杠的牌无法再通过策略选择「打出」（符合需求「能杠则优先杠自己的手牌」）。
因此 `from_id(to_id(("discard",t)))` 在该牌可自杠时会返回 `("gang",t)`，
即弃牌-自杠同槽时回路不是恒等——这是有意为之，不是 bug。
"""

import math

DISCARD_GANG_OFFSET = 0   # [0,34) 弃 / 自杠（暗杠、补杠）共享槽
PENG_ID = 34
CHI_OFFSET = 35           # [35,38)
CHI_SLOTS = 3
HU_ID = 38
MING_GANG_ID = 39         # 响应他家弃牌的明杠（不在自己回合）
PASS_ID = 40
ACTION_N = 41

_KIND_ORDER = {"hu": 0, "gang": 1, "peng": 2, "chi": 3, "discard": 4, "pass": 5}


def _own_turn(actions):
    """合法集合里存在 discard <=> 自己回合（响应窗口永远没有 discard）。"""
    return any(a[0] == "discard" for a in actions)


def to_id(action, actions):
    """引擎动作元组 -> 41 动作编号。chi 需在 actions 中定位组合序号。"""
    if action not in actions:
        raise ValueError("动作不在当前合法集合内: %r" % (action,))
    kind = action[0]
    if kind == "discard":
        return DISCARD_GANG_OFFSET + action[1]
    if kind == "gang":
        # 自己回合(暗/补杠)与弃牌共享牌槽；响应窗口(明杠)用独立槽 39。
        return (DISCARD_GANG_OFFSET + action[1]) if _own_turn(actions) else MING_GANG_ID
    if kind == "peng":
        return PENG_ID
    if kind == "chi":
        chis = [a for a in actions if a[0] == "chi"]
        for i, a in enumerate(chis):
            if a == action:
                return CHI_OFFSET + min(i, CHI_SLOTS - 1)
        raise ValueError("吃牌组合不在当前合法集合内: %r" % (action,))
    if kind == "hu":
        return HU_ID
    if kind == "pass":
        return PASS_ID
    raise ValueError("未知动作 %r" % (action,))


def from_id(aid, actions):
    """41 动作编号 -> 引擎动作元组；不在合法集合内返回 None。"""
    aid = int(aid)
    if DISCARD_GANG_OFFSET <= aid < PENG_ID:          # [0,34) 弃/自杠
        tile = aid - DISCARD_GANG_OFFSET
        if not _own_turn(actions):
            return None                               # 响应窗口没有 [0,34) 动作
        gang = next((a for a in actions
                     if a[0] == "gang" and a[1] == tile), None)
        if gang is not None:
            return gang                               # 优先杠
        return next((a for a in actions
                     if a[0] == "discard" and a[1] == tile), None)
    if aid == MING_GANG_ID:                           # 39 响应明杠
        if _own_turn(actions):
            return None                               # 自己回合的杠走 [0,34)，不走 39
        return next((a for a in actions if a[0] == "gang"), None)
    if aid == PENG_ID:                                # 34
        return next((a for a in actions if a[0] == "peng"), None)
    if CHI_OFFSET <= aid < HU_ID:                     # [35,38)
        chis = [a for a in actions if a[0] == "chi"]
        idx = aid - CHI_OFFSET
        return chis[idx] if idx < len(chis) else None
    if aid == HU_ID:                                  # 38
        return next((a for a in actions if a[0] == "hu"), None)
    if aid == PASS_ID:                                # 40
        return next((a for a in actions if a[0] == "pass"), None)
    return None


def action_mask(actions):
    """长度 41 的 0/1 掩码。弃/自杠共享槽只置同一位。"""
    mask = [0] * ACTION_N
    for a in actions:
        mask[to_id(a, actions)] = 1
    return mask


def legal_ids(actions):
    return sorted({to_id(a, actions) for a in actions})


def select_id_with_self_gang_bias(logits, actions, bias=0.0):
    """Select a legal 41 ID, optionally boosting only legal own-turn gangs.

    This inference-only adjustment needs the engine's legal action tuples. It
    does not modify neural logits used for policy-gradient training.
    """
    if len(logits) != ACTION_N or not math.isfinite(bias) or bias < 0:
        raise ValueError('Expected 41 logits and a finite nonnegative self-gang bias')
    ids = legal_ids(actions)
    if not ids:
        raise ValueError('Cannot select from an empty legal action set')
    own_gangs = ({a[1] for a in actions if a[0] == 'gang'}
                 if _own_turn(actions) else set())
    scores = {aid: float(logits[aid]) + (bias if aid in own_gangs else 0.0)
              for aid in ids}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Legal action logits must be finite')
    return max(ids, key=scores.__getitem__)


def describe_id(aid):
    from hzmj.tiles import tile_code
    aid = int(aid)
    if aid < PENG_ID:
        return "打/自杠 " + tile_code(aid)
    if aid == PENG_ID:
        return "碰"
    if aid < HU_ID:
        return "吃#%d" % (aid - CHI_OFFSET)
    if aid == HU_ID:
        return "胡"
    if aid == MING_GANG_ID:
        return "明杠(响应)"
    return "过"


def sort_actions(actions):
    """按动作类别稳定排序，便于打印与调试。"""
    return sorted(actions, key=lambda a: (_KIND_ORDER.get(a[0], 9), a[1:]))
