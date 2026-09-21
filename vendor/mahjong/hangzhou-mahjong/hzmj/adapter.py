"""平台快照 <-> 规则内核的适配层。

平台不下发 allowed_actions（v2 起），动作合法性必须客户端自研判定。
本模块把 GET /api/games/{id}/state 的快照还原成与 hzmj.engine 同构的只读视图，
从而让 hzmj.bots 里的策略在真实对局中原样复用。

与引擎的差异：他家手牌不可见，因此 PlayerView.counts 仅本家有值，
其余三家只有副露与弃牌 —— 这正好等于「已见牌」语义，策略侧无需改动。
"""

from .engine import MAX_CHI_PER_PLAYER, RESERVED_TILES
from .hu import can_hu, hand_size, missing
from .tiles import GOD, GOD_IDX, TILE_KINDS, tile_code, tile_index

PHASE_DRAW = "draw"
PHASE_PENG = "response_peng"
PHASE_CHI = "response_chi"

TOTAL_DRAWABLE = 63  # 136 - 52 发牌 - 1 庄家首抽 - 20 保留


class MeldView:
    __slots__ = ("kind", "tile", "tiles", "from_seat")

    def __init__(self, kind="unknown", tile=None, tiles=(), from_seat=-1):
        self.kind = kind
        self.tile = tile
        self.tiles = list(tiles)
        self.from_seat = from_seat

    @property
    def is_gang(self):
        return "gang" in self.kind or self.kind == "杠"


class PlayerView:
    __slots__ = ("counts", "melds", "discards", "chi_count", "baotou",
                 "chain_count", "piao")

    def __init__(self):
        self.counts = [0] * TILE_KINDS
        self.melds = []
        self.discards = []
        self.chi_count = 0
        self.baotou = False
        self.chain_count = 0
        self.piao = 0

    @property
    def meld_count(self):
        return len(self.melds)

    def tiles_in_hand(self):
        out = []
        for i, c in enumerate(self.counts):
            out.extend([i] * c)
        return out


class GameView:
    """鸭子类型兼容 hzmj.engine.Game 中策略所需的只读字段。"""

    def __init__(self):
        self.players = [PlayerView() for _ in range(4)]
        self.phase = PHASE_DRAW
        self.turn = -1
        self.seat = -1
        self.dealer = -1
        self.last_discard = None
        self.last_discarder = -1
        self.responding_seats = []
        self.drawn_tile = None
        self.drawable = TOTAL_DRAWABLE
        self.catch_play = False
        self.you_cai_bi_kao = False
        self.finished = False


def _parse_meld(raw):
    """尽量宽容地解析平台副露结构，无法识别时退化为纯计数。"""
    m = MeldView()
    if isinstance(raw, dict):
        m.kind = str(raw.get("kind") or raw.get("type") or "unknown")
        tile = raw.get("tile") or raw.get("card")
        if isinstance(tile, str):
            m.tile = tile_index(tile)
        raw_tiles = raw.get("tiles") or raw.get("cards") or []
        m.tiles = [tile_index(t) for t in raw_tiles
                   if isinstance(t, str) and tile_index(t) is not None]
        m.from_seat = raw.get("from_seat", raw.get("from", -1))
    elif isinstance(raw, (list, tuple)):
        kinds = [x for x in raw if isinstance(x, str) and tile_index(x) is None]
        tiles = [tile_index(x) for x in raw
                 if isinstance(x, str) and tile_index(x) is not None]
        if kinds:
            m.kind = kinds[0]
        m.tiles = tiles
        if tiles:
            m.tile = tiles[0]
    elif isinstance(raw, str):
        m.kind = raw
    if m.tile is None and m.tiles:
        # 平台副露只下发 kind + tiles，不给代表牌 tile。缺了它，
        # gang_options 的补杠分支（要求 m.tile == 手上那张）永远匹配不上，
        # 线上会整段漏掉补杠。这里按引擎口径回填：
        # 吃取顺子起始牌（engine._do_chi 用 min），碰/杠四张同牌取任意一张。
        m.tile = min(m.tiles) if m.kind in ("chi", "吃") else m.tiles[0]
    if not m.tiles and m.tile is not None:
        n = 4 if m.is_gang else 3
        m.tiles = [m.tile] * n
    return m


def from_snapshot(snap, you_cai_bi_kao=False):
    """把平台快照转成 GameView。返回 (view, seat)。"""
    v = GameView()
    seat = int(snap.get("seat", -1))
    v.seat = seat
    v.phase = snap.get("phase") or PHASE_DRAW
    v.turn = snap.get("turn", -1)
    v.dealer = snap.get("dealer", -1)
    v.you_cai_bi_kao = you_cai_bi_kao
    v.finished = bool(snap.get("finished"))
    v.responding_seats = list(snap.get("responding_seats") or [])

    melds_all = snap.get("melds") or [[], [], [], []]
    for s in range(4):
        raw_list = melds_all[s] if s < len(melds_all) else []
        v.players[s].melds = [_parse_meld(x) for x in (raw_list or [])]
        v.players[s].chi_count = sum(
            1 for m in v.players[s].melds
            if m.kind in ("chi", "吃"))

    discards_all = snap.get("discards") or [[], [], [], []]
    total_discards = 0
    for s in range(4):
        raw = discards_all[s] if s < len(discards_all) else []
        idxs = []
        for t in (raw or []):
            i = tile_index(t) if isinstance(t, str) else t
            if isinstance(i, int) and 0 <= i < TILE_KINDS:
                idxs.append(i)
        v.players[s].discards = idxs
        total_discards += len(idxs)

    if seat >= 0:
        for t in (snap.get("my_hand") or []):
            i = tile_index(t) if isinstance(t, str) else t
            if isinstance(i, int) and 0 <= i < TILE_KINDS:
                v.players[seat].counts[i] += 1

    last = snap.get("last_discard")
    if isinstance(last, str):
        v.last_discard = tile_index(last)
    elif isinstance(last, int):
        v.last_discard = last
    ld = snap.get("last_discarder")
    if isinstance(ld, int):
        v.last_discarder = ld
    elif v.phase in (PHASE_PENG, PHASE_CHI):
        # 实测快照无 last_discarder；响应阶段 turn 仍指向打牌者（座位轮次未推进）。
        # 不可用 min(responding_seats)-1 推导 —— 座位环绕时会算错
        # （打牌者 1 → responding [2,3,0]，min 推导得 3）。
        if isinstance(v.turn, int) and v.turn >= 0:
            v.last_discarder = v.turn
        elif v.responding_seats:
            v.last_discarder = (min(v.responding_seats) - 1) % 4

    drawn = snap.get("drawn_tile")
    if isinstance(drawn, str) and drawn:
        v.drawn_tile = tile_index(drawn)
    elif isinstance(drawn, int):
        v.drawn_tile = drawn

    god = snap.get("god") or {}
    if seat >= 0:
        p = v.players[seat]
        p.chain_count = int(god.get("chain_count") or 0)
        p.piao = min(p.chain_count, int(god.get("piao") or p.chain_count))
        v.catch_play = bool(god.get("catch_play"))
        # 平台给的 baotou 是权威值；缺省时本地推算（须为未摸牌的 13 张态）
        if "baotou" in god:
            p.baotou = bool(god.get("baotou"))
        else:
            probe = list(p.counts)
            if v.drawn_tile is not None and probe[v.drawn_tile] > 0:
                probe[v.drawn_tile] -= 1
            from .hu import is_baotou
            p.baotou = is_baotou(probe, p.meld_count)

    # 实测字段名为 wall_remaining（含末尾 20 张保留牌）：开局 83 = 136-52-1。
    # 可摸张数 = wall_remaining - 20，与引擎 live_end 口径一致。
    wall = snap.get("wall_remaining", snap.get("remaining"))
    if isinstance(wall, int):
        v.drawable = max(0, wall - RESERVED_TILES)
    else:
        v.drawable = max(0, TOTAL_DRAWABLE - total_discards)
    return v, seat


# ---------------- 自研动作判定 ----------------

def chi_options(view, seat, tile):
    p = view.players[seat]
    if tile is None or tile >= 27 or p.chi_count >= MAX_CHI_PER_PLAYER:
        return []
    r = tile % 9
    base = tile - r
    out = []
    for lo in (r - 2, r - 1, r):
        if lo < 0 or lo + 2 > 8:
            continue
        need = [base + lo + k for k in range(3) if base + lo + k != tile]
        if all(p.counts[x] > 0 for x in need):
            if need[0] == need[1] and p.counts[need[0]] < 2:
                continue
            out.append(tuple(need))
    return out


def gang_options(view, seat):
    p = view.players[seat]
    out = []
    if view.drawable <= 0:      # 最后 10 墩之内禁杠
        return out
    for t in range(TILE_KINDS):
        if t == GOD_IDX:
            continue
        if p.counts[t] == 4:
            out.append(("gang", t, "an"))
        elif p.counts[t] == 1:
            for m in p.melds:
                if m.kind in ("peng", "碰") and m.tile == t:
                    out.append(("gang", t, "bu"))
                    break
    return out


def legal_actions(view, seat):
    """依快照自判本家当前可提交的动作（对应平台 §2.1 自研动作判定）。"""
    if view.finished or seat < 0:
        return []
    p = view.players[seat]
    acts = []
    if view.phase == PHASE_DRAW and view.turn == seat:
        expect = hand_size(p.meld_count, drawn=True)
        if sum(p.counts) != expect:
            # 张数与副露数不自洽（多为副露结构解析不全），只做保守出牌
            return [("discard", t) for t in range(TILE_KINDS) if p.counts[t]]
        # 平台 v1 规则：碰/吃/杠后、摸牌前不得胡（必须是刚摸的牌）
        if view.drawn_tile is not None and can_hu(
                p.counts, p.meld_count, view.you_cai_bi_kao, p.baotou):
            acts.append(("hu",))
        if view.catch_play:
            for g in gang_options(view, seat):
                if g[2] == "an":
                    acts.append(g)
            if view.drawn_tile is not None:
                acts.append(("discard", view.drawn_tile))
            return acts
        acts.extend(gang_options(view, seat))
        acts.extend(("discard", t) for t in range(TILE_KINDS) if p.counts[t])
    elif view.phase in (PHASE_PENG, PHASE_CHI):
        if seat not in view.responding_seats:
            return []
        acts.append(("pass",))
        tile = view.last_discard
        if tile is None or tile == GOD_IDX or view.catch_play:
            return acts
        if view.phase == PHASE_PENG:
            if p.counts[tile] >= 2:
                acts.append(("peng",))
            if p.counts[tile] == 3 and view.drawable > 0:
                acts.append(("gang", tile, "ming"))
        else:
            for combo in chi_options(view, seat, tile):
                acts.append(("chi",) + combo)
    return acts


def action_payload(view, action):
    """引擎动作 -> 平台 POST /api/games/{id}/action 请求体。"""
    kind = action[0]
    if kind == "discard":
        return {"action": "discard", "tile": tile_code(action[1])}
    if kind == "hu":
        return {"action": "hu", "tile": ""}
    if kind == "pass":
        return {"action": "pass", "tile": ""}
    if kind == "peng":
        return {"action": "peng",
                "tile": tile_code(view.last_discard)
                if view.last_discard is not None else ""}
    if kind == "chi":
        return {"action": "chi",
                "tile": tile_code(view.last_discard)
                if view.last_discard is not None else "",
                "tiles": [tile_code(action[1]), tile_code(action[2])]}
    if kind == "gang":
        return {"action": "gang", "tile": tile_code(action[1])}
    raise ValueError("未知动作 %r" % (action,))


def describe(action):
    kind = action[0]
    if kind == "discard":
        return "打 " + tile_code(action[1])
    if kind == "gang":
        return "杠 %s(%s)" % (tile_code(action[1]), action[2])
    if kind == "chi":
        return "吃 %s+%s" % (tile_code(action[1]), tile_code(action[2]))
    return {"hu": "胡", "peng": "碰", "pass": "过"}.get(kind, kind)


def fallback_discard(view, seat):
    """兜底出牌：优先打非财神的孤张，保证永远有合法动作可提交。"""
    p = view.players[seat]
    if view.drawn_tile is not None and view.catch_play:
        return ("discard", view.drawn_tile)
    best, best_key = None, None
    for t in range(TILE_KINDS):
        if not p.counts[t]:
            continue
        p.counts[t] -= 1
        m = missing(p.counts, p.meld_count)
        p.counts[t] += 1
        key = (m, 1 if t == GOD_IDX else 0, -t)
        if best_key is None or key < best_key:
            best, best_key = t, key
    return ("discard", best) if best is not None else None
