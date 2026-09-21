"""公开观测：布局、可见信息边界和派生牌效统一在这里维护。

仅支持 v2 的 6122 维输入，不改变动作编号，也不读取他家手牌或牌墙。
GameView 可传入 public_events（与引擎公开事件同格式）；缺失时明确标记未知，
仍从快照提供各家有序牌河、副露。历史属于当前可见状态，不需要 recurrent 网络。
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

TILE_KINDS = 34
ACTION_N = 74
RECENT_DISCARDS = 18  # 每家最近弃牌，最新的在 slot 0；引擎牌河保留被吃碰的牌
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
DISCARD_START = MELD_START + 4 * MELD_SLOTS
ACTION_START = DISCARD_START + 4 * RECENT_DISCARDS
CANDIDATE_START = ACTION_START + RECENT_ACTIONS
EFFECTIVE_START = CANDIDATE_START + 4
MELD_SCALAR_START = 16 + 5  # 原始 scalars + 庄家相对座次/未知
ACTION_SCALAR_START = MELD_SCALAR_START + 4 * MELD_SLOTS * 11
# v2 planes: 原始15 + 副露16 + 牌河72 + 动作24 + 候选4 + 进张矩阵34。
# v2 scalars: 原始16 + 庄家5 + 副露16×11 + 动作24×10 + 历史可用性1。
LAYOUT = ObservationLayout(
    DEFAULT_VERSION, EFFECTIVE_START + TILE_KINDS,
    ACTION_SCALAR_START + RECENT_ACTIONS * 10 + 1,
)


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


def _encode_melds_and_rivers(game, seat, planes, scalars):
    kinds = {"chi": 0, "peng": 1, "gang_ming": 2, "gang_an": 3, "gang_bu": 4,
             "吃": 0, "碰": 1}
    for relative in range(4):
        player = game.players[(seat + relative) % 4]
        for slot, tile in enumerate(reversed(player.discards[-RECENT_DISCARDS:])):
            planes[DISCARD_START + relative * RECENT_DISCARDS + slot, tile] = 1.0
        for slot, meld in enumerate(player.melds[:MELD_SLOTS]):
            group = relative * MELD_SLOTS + slot
            for tile in meld.tiles:
                planes[MELD_START + group, tile] += 0.25
            metadata = MELD_SCALAR_START + group * 11
            scalars[metadata + kinds.get(meld.kind, 5)] = 1.0  # 未知种类单独编码
            scalars[metadata + 6 + _relative_seat(meld.from_seat, seat)] = 1.0


def _encode_public_actions(game, seat, planes, scalars, public_events):
    # 严格 allowlist；tile_drawn/window_open 可能含私有信息，不能整体序列化。
    events = public_events if public_events is not None else getattr(game, "events", None)
    scalars[-1] = float(events is not None)
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
        # tile 是动作指向的牌；chi 的完整组合已在结构化副露中编码。
        planes[ACTION_START + slot, tile] = 1.0
        metadata = ACTION_SCALAR_START + slot * 10
        scalars[metadata + action] = 1.0
        scalars[metadata + 6 + relative] = 1.0
        slot += 1
        if slot == RECENT_ACTIONS:
            break


def _visible_counts(game, seat, public_events):
    """与 BC 的 RuleBot 示范策略共用公开牌去重逻辑。"""
    from hzmj.bots import seen_counts

    seen = seen_counts(game, seat, public_events=public_events)
    return np.clip(np.asarray(seen, dtype=np.float32), 0, 4)


def encode_observation(game, seat, action_mask, *, version=DEFAULT_VERSION,
                       you_cai_bi_kao=False, public_events=None, base_observation=None):
    """适用于训练 Game / 平台 GameView；mask 始终在尾部，返回独立 float32 数组。

    effective[draw, discard] = 该摸牌能降低弃牌后 missing 时的公开未见张数 / 4。
    未见牌包含他家手牌和牌墙，不能解释为真实牌墙数量或精确摸牌概率。
    missing 沿用杭州规则引擎的缺口定义（含财神/七对），不等同于日麻 shanten。
    """
    from mjenv import encode

    layout = observation_layout(version)
    mask = np.asarray(action_mask, dtype=np.float32)
    if mask.shape != (ACTION_N,) or not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError("麻将 action_mask 必须是 74 维 0/1 向量")
    base = np.asarray(base_observation if base_observation is not None else
                      encode.encode(game, seat, you_cai_bi_kao), dtype=np.float32)
    if base.shape != (526,):
        raise ValueError(f"上游基础观测布局已变化: {base.shape}，需要重新适配")
    if len(game.players) != 4:
        raise ValueError("v2 观测仅支持四人麻将")
    result = np.zeros(layout.flat_dim, dtype=np.float32)
    planes = result[:layout.tile_features].reshape(layout.tile_channels, TILE_KINDS)
    scalars = result[layout.tile_features:-ACTION_N]
    planes[:15] = base[:510].reshape(15, TILE_KINDS)
    planes[14] = _visible_counts(game, seat, public_events) / 4.0
    scalars[:16] = base[510:]
    scalars[16 + _relative_seat(game.dealer, seat)] = 1.0
    _encode_melds_and_rivers(game, seat, planes, scalars)
    _encode_public_actions(game, seat, planes, scalars, public_events)
    legal_discards = mask[:TILE_KINDS]
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
        planes[EFFECTIVE_START:] = remaining.T / 4.0
    result[-ACTION_N:] = mask
    return result
