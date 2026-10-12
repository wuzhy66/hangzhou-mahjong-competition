"""公开观测：布局、可见信息边界和派生牌效统一在这里维护。

仅支持 v2 观测（布局见 LAYOUT），不读取他家手牌或牌墙。标量已折叠进平面：
副露/动作元数据用「对齐平面」（标记落在对应牌位上），全局标量用广播平面。
GameView 可传入 public_events（与引擎公开事件同格式）；缺失时明确标记未知，
仍从快照提供各家有序牌河、副露。历史属于当前可见状态，不需要 recurrent 网络。
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .action_scheme import ACTION_N  # 41 动作方案单一来源；不再硬编码 74

TILE_KINDS = 34
RECENT_DISCARDS = 18  # 每家最近弃牌，最新的在 slot 0；引擎牌河保留被吃碰的牌
# 有序牌河现同时编码手切/摸切；非零值仍表示该槽的牌种。
# 公开事件缺少标记时保留「未知」，不从暗手或牌墙推断。
TEDASHI = 1.0
TSUMOGIRI = 0.5
DISCARD_ORIGIN_UNKNOWN = 0.75
RECENT_ACTIONS = 24  # 全桌最近公开动作，最新的在 slot 0；摸牌事件永不进入编码
MELD_SLOTS = 4


@dataclass(frozen=True)
class ObservationLayout:
    version: int
    tile_channels: int
    scalars: int

    @property
    def tile_features(self):
        return self.tile_channels * TILE_KINDS

    @property
    def flat_dim(self):
        return self.tile_features + self.scalars + ACTION_N


DEFAULT_VERSION = 2
MELD_START = 15
DISCARD_START = MELD_START + 4 * MELD_SLOTS            # 31
ACTION_START = DISCARD_START + 4 * RECENT_DISCARDS     # 103
CANDIDATE_START = ACTION_START + RECENT_ACTIONS        # 127
EFFECTIVE_START = CANDIDATE_START + 4                  # 131
STRUCT_PLANES = EFFECTIVE_START + TILE_KINDS           # 165 个结构化牌面平面

# 对齐的副露/动作元数据平面：标记落在对应牌位上，取代逐槽标量广播（紧凑折叠）。
MELD_KIND_START = STRUCT_PLANES                        # 165；6 类：吃/碰/明杠/暗杠/补杠/未知
MELD_KIND_N = 6
MELD_SRC_START = MELD_KIND_START + MELD_KIND_N         # 171；5 来源：相对座次0..3 + 未知
MELD_SRC_N = 5
ACTION_TYPE_START = MELD_SRC_START + MELD_SRC_N        # 176；6 类型：弃/碰/吃/明杠/暗杠/补杠
ACTION_TYPE_N = 6
ACTION_SEAT_START = ACTION_TYPE_START + ACTION_TYPE_N  # 182；4 相对座次 0..3
ACTION_SEAT_N = 4

# 全局标量（无牌位可对齐，仍广播成常量平面）：原始16 + 庄家5 + 历史1 = 22。
GLOBAL_START = ACTION_SEAT_START + ACTION_SEAT_N       # 186
DEALER_SCALAR_START = 16                                # 全局标量内部：庄家 one-hot(4)+未知(1) 起点
GLOBAL_HISTORY = 21                                     # 全局标量内部：历史可用性位
GLOBAL_N = 22
# v2 planes：结构化165 + 对齐元数据(6+5+6+4=21) + 全局广播22 = 208；无独立标量段(scalars=0)。
TILE_CHANNELS = GLOBAL_START + GLOBAL_N                 # 208
LAYOUT = ObservationLayout(DEFAULT_VERSION, TILE_CHANNELS, 0)


def observation_layout(version=DEFAULT_VERSION):
    if type(version) is not int or version != DEFAULT_VERSION:
        raise ValueError(f"不支持的 observation_version: {version!r}，仅支持 2")
    return LAYOUT


def layout_from_shape(shape):
    if tuple(shape) != (LAYOUT.flat_dim,):
        raise ValueError(f"不支持的麻将观测 shape: {shape}；仅支持 ({LAYOUT.flat_dim},)")
    return LAYOUT


def _relative_seat(value, seat):
    return (int(value) - seat) % 4 if isinstance(value, (int, np.integer)) and 0 <= value < 4 else 4


@lru_cache(maxsize=256)
def _discard_analysis(counts, meld_count):
    """只缓存自家手牌推导，最多约 0.3 MiB 数组；公开剩余牌每次重算。"""
    from mjenv import fastrule

    missing = np.zeros(TILE_KINDS, dtype=np.float32)
    effective = np.zeros((TILE_KINDS, TILE_KINDS), dtype=bool)
    for discard, gap in enumerate(fastrule.missing_all_discards(counts, meld_count)):
        if gap is None:
            continue
        missing[discard] = min(1.0, gap / 8.0)
        after = list(counts)
        after[discard] -= 1
        effective[discard] = [value is not None and value < gap
                              for value in fastrule.missing_all_draws(after, meld_count)]
    missing.setflags(write=False)
    effective.setflags(write=False)
    return missing, effective


def _discard_origins(game, public_events):
    events = public_events if public_events is not None else getattr(game, "events", None)
    if events is None:
        return [None] * 4
    per_seat = [[] for _ in range(4)]
    for event in events:
        if event.get("kind") != "tile_discarded":
            continue
        data = event.get("data", {})
        actor, tile = data.get("seat"), data.get("tile")
        if isinstance(actor, (int, np.integer)) and 0 <= actor < 4:
            marker = data.get("is_tedashi")
            per_seat[actor].append((tile, marker if isinstance(marker, (bool, np.bool_)) else None))
    result = []
    for actor, player in enumerate(game.players):
        records = per_seat[actor]
        if len(records) == len(player.discards) and all(
                tile == discard for (tile, _), discard in zip(records, player.discards)):
            result.append([marker for _, marker in records])
        else:
            result.append(None)
    return result


def _encode_melds_and_rivers(game, seat, planes, public_events):
    kinds = {"chi": 0, "peng": 1, "gang_ming": 2, "gang_an": 3, "gang_bu": 4,
             "吃": 0, "碰": 1}
    origins = _discard_origins(game, public_events)
    for relative in range(4):
        actor = (seat + relative) % 4
        player = game.players[actor]
        for slot, index in enumerate(range(len(player.discards) - 1,
                                          max(-1, len(player.discards) - RECENT_DISCARDS - 1), -1)):
            tile = player.discards[index]
            marker = None if origins[actor] is None else origins[actor][index]
            planes[DISCARD_START + relative * RECENT_DISCARDS + slot, tile] = (
                DISCARD_ORIGIN_UNKNOWN if marker is None else
                TEDASHI if marker else TSUMOGIRI)
        for slot, meld in enumerate(player.melds[:MELD_SLOTS]):
            group = relative * MELD_SLOTS + slot
            kind_id = kinds.get(meld.kind, 5)              # 未知种类=5
            src = _relative_seat(meld.from_seat, seat)     # 0..3，未知=4
            for tile in meld.tiles:
                planes[MELD_START + group, tile] += 0.25   # 每组保留牌面分组
                # 对齐元数据：把「种类/来源」标在这组牌所在的牌位上，取代逐槽标量。
                planes[MELD_KIND_START + kind_id, tile] += 0.25
                planes[MELD_SRC_START + src, tile] += 0.25


def _encode_public_actions(game, seat, planes, globals_, public_events):
    # 严格 allowlist；tile_drawn/window_open 可能含私有信息，不能整体序列化。
    events = public_events if public_events is not None else getattr(game, "events", None)
    globals_[GLOBAL_HISTORY] = float(events is not None)
    if events is None:
        return
    slot = 0
    for event in reversed(events):
        kind, data = event.get("kind"), event.get("data", {})
        action = {"tile_discarded": 0, "peng": 1, "chi": 2}.get(kind)
        if kind == "gang":
            action = {"ming": 3, "an": 4, "bu": 5}.get(data.get("mode"))
        if action is None:
            continue
        relative = _relative_seat(data.get("seat"), seat)
        tile = data.get("tile")
        if relative == 4 or not isinstance(tile, (int, np.integer)) or not 0 <= tile < TILE_KINDS:
            continue
        # tile 是动作指向的牌；有序牌面(presence)保留硬顺序。
        planes[ACTION_START + slot, tile] = 1.0
        # 类型/座次对齐平面以「新近度」为值（最新=1，越旧越小），显式保留动作时序。
        recency = (RECENT_ACTIONS - slot) / RECENT_ACTIONS
        p_type = ACTION_TYPE_START + action
        p_seat = ACTION_SEAT_START + relative
        if planes[p_type, tile] < recency:      # 同牌同类只保留最新一次的新近度
            planes[p_type, tile] = recency
        if planes[p_seat, tile] < recency:
            planes[p_seat, tile] = recency
        slot += 1
        if slot == RECENT_ACTIONS:
            break


def _visible_counts(game, seat, public_events):
    """与 BC 的 RuleBot 示范策略共用公开牌去重逻辑。"""
    from hzmj.bots import seen_counts

    seen = seen_counts(game, seat, public_events=public_events)
    return np.clip(np.asarray(seen, dtype=np.float32), 0, 4)


def encode_observation(game, seat, action_mask, *, version=DEFAULT_VERSION,
                       you_cai_bi_kao=False, public_events=None, base_observation=None,
                       legal_actions=None):
    """适用于训练 Game / 平台 GameView；mask 始终在尾部，返回独立 float32 数组。

    effective[draw, discard] = 该摸牌能降低弃牌后 missing 时的公开未见张数 / 4。
    未见牌包含他家手牌和牌墙，不能解释为真实牌墙数量或精确摸牌概率。
    missing 沿用杭州规则引擎的缺口定义（含财神/七对），不等同于日麻 shanten。
    """
    from mjenv import encode

    layout = observation_layout(version)
    mask = np.asarray(action_mask, dtype=np.float32)
    if mask.shape != (ACTION_N,) or not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError(f"麻将 action_mask 必须是 {ACTION_N} 维 0/1 向量")
    base = np.asarray(base_observation if base_observation is not None else
                      encode.encode(game, seat, you_cai_bi_kao), dtype=np.float32)
    if base.shape != (526,):
        raise ValueError(f"上游基础观测布局已变化: {base.shape}，需要重新适配")
    if len(game.players) != 4:
        raise ValueError("v2 观测仅支持四人麻将")
    result = np.zeros(layout.flat_dim, dtype=np.float32)
    planes = result[:layout.tile_features].reshape(layout.tile_channels, TILE_KINDS)
    # 全局标量先写临时数组，最后广播成常量平面；副露/动作元数据走对齐平面。
    globals_ = np.zeros(GLOBAL_N, dtype=np.float32)
    planes[:15] = base[:510].reshape(15, TILE_KINDS)
    planes[14] = _visible_counts(game, seat, public_events) / 4.0
    globals_[:16] = base[510:]
    globals_[DEALER_SCALAR_START + _relative_seat(game.dealer, seat)] = 1.0
    _encode_melds_and_rivers(game, seat, planes, public_events)
    _encode_public_actions(game, seat, planes, globals_, public_events)
    # A 41-action tile slot can mean self-gang even when discarding that tile is
    # illegal. Candidate-discard features must follow the actual engine actions,
    # not the merged action mask. GameView callers pass their adapter action list.
    if legal_actions is None and hasattr(game, 'legal_actions') and mask[:TILE_KINDS].any():
        legal_actions = game.legal_actions(seat)
    if legal_actions is None:
        legal_discards = mask[:TILE_KINDS]
    else:
        legal_discards = np.zeros(TILE_KINDS, dtype=np.float32)
        for action in legal_actions:
            if action[0] == 'discard':
                legal_discards[action[1]] = 1.0
    if legal_discards.any():
        hero = game.players[seat]
        missing, effective = _discard_analysis(tuple(hero.counts), hero.meld_count)
        # 该牌从手牌移到牌河后，公开已见总数不变，因此使用弃牌前的 seen 即可。
        unseen = np.clip(4.0 - planes[14] * 4.0, 0.0, 4.0)
        remaining = effective * unseen[None, :] * legal_discards[:, None]
        planes[CANDIDATE_START] = legal_discards
        planes[CANDIDATE_START + 1] = missing * legal_discards
        planes[CANDIDATE_START + 2] = np.count_nonzero(remaining, axis=1) / 34.0
        planes[CANDIDATE_START + 3] = remaining.sum(axis=1) / 136.0
        planes[EFFECTIVE_START:STRUCT_PLANES] = remaining.T / 4.0
    # 全局标量广播到最后 22 个平面（无牌位可对齐，故广播）。
    planes[GLOBAL_START:] = globals_[:, None]
    result[-ACTION_N:] = mask
    return result
