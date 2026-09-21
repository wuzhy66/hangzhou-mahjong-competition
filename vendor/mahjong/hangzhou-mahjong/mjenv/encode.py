"""观测编码：15 个 34 维平面 + 16 个标量 = 526 维扁平向量。

只使用该座位**合法可见**的信息（自家手牌 + 全场副露/弃牌 + 公开状态），
不泄露他家手牌与牌墙顺序 —— 与平台快照的信息量严格一致。

平面（每个长 34，下标即牌索引，白板=33）：
     0-3   自家手牌 >=1 / >=2 / >=3 / >=4
     4     本轮刚摸的牌
     5     场上最后一张弃牌
     6-9   四家弃牌计数 / 4（相对座次：0 = 自己，1 = 下家 …）
    10-13  四家副露计数 / 4（相对座次）
    14     已见牌计数 / 4（自家手牌 + 全场副露 + 全场弃牌）

标量：
     0  可摸张数 / 63          1  是否庄家           2  爆头态
     3  动作链 / 6             4  已飘 / 6           5  抓打圈受限
     6  有财必拷响             7-9  阶段 onehot（draw / 碰窗 / 吃窗）
    10-13 当前行动者相对座次 onehot
    14  自家副露数 / 4         15  自家缺口数 / 8
"""

from hzmj.tiles import TILE_KINDS

from . import fastrule

N_PLANES = 15
N_SCALARS = 16
OBS_DIM = N_PLANES * TILE_KINDS + N_SCALARS

TOTAL_DRAWABLE = 63.0
_PHASE_SLOT = {"draw": 0, "response_peng": 1, "response_chi": 2}


def encode(game, seat, you_cai_bi_kao=False):
    """game 为 hzmj.engine.Game（或同构对象）。返回长度 526 的 float 列表。"""
    obs = [0.0] * OBS_DIM
    p = game.players[seat]
    n = len(game.players)

    # --- 平面 0-3：自家手牌 ---
    counts = p.counts
    for t in range(TILE_KINDS):
        c = counts[t]
        for k in range(min(c, 4)):
            obs[k * TILE_KINDS + t] = 1.0

    # --- 平面 4：刚摸的牌 ---
    drawn = game.drawn_tile if game.turn == seat else None
    if drawn is not None:
        obs[4 * TILE_KINDS + drawn] = 1.0

    # --- 平面 5：最后弃牌 ---
    if game.last_discard is not None:
        obs[5 * TILE_KINDS + game.last_discard] = 1.0

    # --- 平面 6-13：弃牌 / 副露（相对座次） ---
    seen = [0] * TILE_KINDS
    for t in range(TILE_KINDS):
        seen[t] = counts[t]
    for s in range(n):
        rel = (s - seat) % n
        q = game.players[s]
        dbase = (6 + rel) * TILE_KINDS
        mbase = (10 + rel) * TILE_KINDS
        for t in q.discards:
            obs[dbase + t] += 0.25
            seen[t] += 1
        for m in q.melds:
            for t in m.tiles:
                obs[mbase + t] += 0.25
                seen[t] += 1

    # --- 平面 14：已见牌 ---
    base14 = 14 * TILE_KINDS
    for t in range(TILE_KINDS):
        if seen[t]:
            obs[base14 + t] = min(1.0, seen[t] * 0.25)

    # --- 标量 ---
    s0 = N_PLANES * TILE_KINDS
    obs[s0 + 0] = min(1.0, game.drawable / TOTAL_DRAWABLE)
    obs[s0 + 1] = 1.0 if seat == game.dealer else 0.0
    obs[s0 + 2] = 1.0 if p.baotou else 0.0
    obs[s0 + 3] = min(1.0, p.chain_count / 6.0)
    obs[s0 + 4] = min(1.0, p.piao / 6.0)
    # engine.Game 用 catch_origin（发起者座位），adapter.GameView 只有布尔
    # catch_play（且已是「对本家是否生效」的语义）。两种视图都要能编码，
    # 否则把 NeuralPolicy 挂到 bot_ai 上时这一维会恒为 0。
    if hasattr(game, "catch_origin"):
        catch = game.catch_origin
        obs[s0 + 5] = 1.0 if (catch >= 0 and seat != catch) else 0.0
    else:
        obs[s0 + 5] = 1.0 if getattr(game, "catch_play", False) else 0.0
    obs[s0 + 6] = 1.0 if you_cai_bi_kao else 0.0
    slot = _PHASE_SLOT.get(game.phase)
    if slot is not None:
        obs[s0 + 7 + slot] = 1.0
    if isinstance(game.turn, int) and game.turn >= 0:
        obs[s0 + 10 + ((game.turn - seat) % n)] = 1.0
    obs[s0 + 14] = min(1.0, p.meld_count / 4.0)
    obs[s0 + 15] = min(1.0, fastrule.missing(counts, p.meld_count) / 8.0)
    return obs


def plane_names():
    return [
        "hand>=1", "hand>=2", "hand>=3", "hand>=4", "drawn", "last_discard",
        "discard_rel0", "discard_rel1", "discard_rel2", "discard_rel3",
        "meld_rel0", "meld_rel1", "meld_rel2", "meld_rel3", "seen",
    ]


def scalar_names():
    return [
        "drawable", "is_dealer", "baotou", "chain", "piao", "catch_play",
        "ycbk", "phase_draw", "phase_peng", "phase_chi",
        "turn_rel0", "turn_rel1", "turn_rel2", "turn_rel3",
        "meld_count", "missing",
    ]
