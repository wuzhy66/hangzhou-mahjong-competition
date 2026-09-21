"""杭州麻将对局引擎（纯标准库，确定性可复现）。

实现接入指南 §1.1-§1.4 的全部平台固有规则：

    136 张无花，白板 = 财神（百搭）
    财神不能被吃/碰/杠，可主动打出；打出财神那一圈为「抓打圈」
    只能自摸，禁止点炮与抢杠胡
    最后 10 墩（20 张）保留不摸，且此区间内禁杠
    吃最多 2 摊，碰/杠不限次数；碰（含明杠）窗口先于吃窗口
    动作链：飘（爆头态打出财神）与杠各 ×2，打出其它牌即断链
    连庄：庄家胡或流局则连庄，闲家胡则庄位下移
"""

import random

from .fan import compute_fan, settle
from .hu import can_hu, hand_size, is_baotou, missing
from .tiles import GOD_IDX, TILE_KINDS, TILES_PER_KIND, tile_code

PHASE_DEAL = "deal"
PHASE_DRAW = "draw"
PHASE_PENG = "response_peng"
PHASE_CHI = "response_chi"
PHASE_SETTLED = "settled"

RESERVED_TILES = 20  # 最后 10 墩保留不摸
MAX_CHI_PER_PLAYER = 2


class Meld:
    __slots__ = ("kind", "tile", "tiles", "from_seat")

    def __init__(self, kind, tile, tiles, from_seat=-1):
        self.kind = kind          # peng / gang_ming / gang_an / gang_bu / chi
        self.tile = tile          # 代表牌（吃为顺子起始牌）
        self.tiles = tiles        # 组成该面子的牌索引
        self.from_seat = from_seat

    @property
    def is_gang(self):
        return self.kind.startswith("gang")

    def __repr__(self):
        return "<%s %s>" % (self.kind, tile_code(self.tile))


class PlayerState:
    def __init__(self):
        self.counts = [0] * TILE_KINDS
        self.melds = []
        self.discards = []
        self.chi_count = 0
        self.baotou = False
        self.chain_count = 0
        self.piao = 0
        self.god_drawn = 0        # 白板获取数（配牌 + 每次摸到）

    @property
    def meld_count(self):
        return len(self.melds)

    def tiles_in_hand(self):
        out = []
        for i, c in enumerate(self.counts):
            out.extend([i] * c)
        return out


class Game:
    """一局（round）杭州麻将。"""

    def __init__(self, dealer=0, base=1, you_cai_bi_kao=False,
                 seed=None, wall=None, n_players=4):
        self.n = n_players
        self.dealer = dealer
        self.base = base
        self.you_cai_bi_kao = you_cai_bi_kao
        self.rng = random.Random(seed)
        self.players = [PlayerState() for _ in range(self.n)]
        self.events = []

        self.wall = list(wall) if wall is not None else self._build_wall()
        self.head = 0                      # 下一张可摸牌下标
        self.tail = len(self.wall)         # 杠补牌自尾部取（开区间）
        self.live_end = len(self.wall) - RESERVED_TILES

        self.phase = PHASE_DEAL
        self.turn = dealer
        self.drawn_tile = None
        self.last_discard = None
        self.last_discarder = -1
        self.responding_seats = []
        self.responded = set()
        self.catch_origin = -1             # 抓打圈发起者，-1 表示未激活
        self.finished = False
        self.result = None
        self.gang_supplement = False       # 当前摸牌是否为杠补牌

        self._deal()

    # ---------------- 初始化 ----------------

    def _build_wall(self):
        wall = [i for i in range(TILE_KINDS) for _ in range(TILES_PER_KIND)]
        self.rng.shuffle(wall)
        return wall

    def _deal(self):
        for seat in range(self.n):
            p = self.players[seat]
            for _ in range(13):
                t = self.wall[self.head]
                self.head += 1
                p.counts[t] += 1
                if t == GOD_IDX:
                    p.god_drawn += 1
        self._emit("deal", {"dealer": self.dealer})
        for seat in range(self.n):
            self._refresh_baotou(seat)
        self.phase = PHASE_DRAW
        self.turn = self.dealer
        self._draw(self.dealer)

    # ---------------- 事件 ----------------

    def _emit(self, kind, data):
        self.events.append({"seq": len(self.events), "kind": kind, "data": data})

    # ---------------- 牌墙 ----------------

    @property
    def drawable(self):
        return max(0, self.live_end - self.head)

    def _refresh_baotou(self, seat):
        p = self.players[seat]
        p.baotou = is_baotou(p.counts, p.meld_count)

    def _draw(self, seat, from_tail=False):
        p = self.players[seat]
        if from_tail:
            self.tail -= 1
            t = self.wall[self.tail]
            self.live_end -= 1
            self.gang_supplement = True
        else:
            if self.drawable <= 0:
                self._exhaust()
                return None
            t = self.wall[self.head]
            self.head += 1
            self.gang_supplement = False
        p.counts[t] += 1
        if t == GOD_IDX:
            p.god_drawn += 1
        self.drawn_tile = t
        self.turn = seat
        self.phase = PHASE_DRAW
        self.responded.clear()
        self._emit("tile_drawn", {"seat": seat, "tile": t,
                                  "gang": self.gang_supplement})
        return t

    def _exhaust(self):
        self.finished = True
        self.phase = PHASE_SETTLED
        self.result = {"draw": True, "winner": -1, "fan": 0,
                       "scores": [0] * self.n, "next_dealer": self.dealer,
                       "detail": []}
        self._emit("round_ended", {"draw": True})

    # ---------------- 合法动作 ----------------

    def legal_actions(self, seat):
        """返回该座位当前可提交的动作列表。"""
        if self.finished:
            return []
        p = self.players[seat]
        acts = []
        if self.phase == PHASE_DRAW and self.turn == seat:
            # 平台 v1 规则：碰/吃/杠后、摸牌前不得胡（必须是刚摸的牌）
            if self.drawn_tile is not None and self._can_hu(seat):
                acts.append(("hu",))
            catch = self.catch_origin >= 0 and seat != self.catch_origin
            if catch:
                # 抓打圈：出牌只能打刚摸到的牌，四种副露里仅暗杠可用
                for g in self._gang_options(seat, self_draw=True):
                    if g[2] == "an":
                        acts.append(g)
                if self.drawn_tile is not None:
                    acts.append(("discard", self.drawn_tile))
                return acts
            for g in self._gang_options(seat, self_draw=True):
                acts.append(g)
            for t in range(TILE_KINDS):
                if p.counts[t]:
                    acts.append(("discard", t))
        elif self.phase in (PHASE_PENG, PHASE_CHI):
            if seat not in self.responding_seats or seat in self.responded:
                return []
            acts.append(("pass",))
            tile = self.last_discard
            if tile is None or tile == GOD_IDX:
                return acts
            if self.phase == PHASE_PENG:
                if p.counts[tile] >= 2:
                    acts.append(("peng",))
                if p.counts[tile] == 3 and self._gang_allowed():
                    acts.append(("gang", tile, "ming"))
            else:
                for combo in self._chi_options(seat, tile):
                    acts.append(("chi",) + combo)
        return acts

    def _gang_allowed(self):
        """最后 10 墩之内禁止杠牌。"""
        return self.drawable > 0

    def _gang_options(self, seat, self_draw):
        p = self.players[seat]
        out = []
        if not self._gang_allowed():
            return out
        for t in range(TILE_KINDS):
            if t == GOD_IDX:
                continue          # 财神不能杠
            if p.counts[t] == 4:
                out.append(("gang", t, "an"))
            elif p.counts[t] == 1:
                for m in p.melds:
                    if m.kind == "peng" and m.tile == t:
                        out.append(("gang", t, "bu"))
                        break
        return out

    def _chi_options(self, seat, tile):
        """返回可吃组合 [(a, b), ...]，a/b 为手牌中用于吃的两张牌索引。"""
        p = self.players[seat]
        if tile >= 27 or p.chi_count >= MAX_CHI_PER_PLAYER:
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

    def _can_hu(self, seat):
        p = self.players[seat]
        if sum(p.counts) != hand_size(p.meld_count, drawn=True):
            return False
        return can_hu(p.counts, p.meld_count, self.you_cai_bi_kao, p.baotou)

    # ---------------- 动作执行 ----------------

    def step(self, seat, action):
        """执行动作。非法动作抛 ValueError（对应平台 409）。"""
        if self.finished:
            raise ValueError("对局已结束")
        legal = self.legal_actions(seat)
        if action not in legal:
            raise ValueError("非法动作 %r（座位 %d，阶段 %s）"
                             % (action, seat, self.phase))
        kind = action[0]
        if kind == "hu":
            self._do_hu(seat)
        elif kind == "discard":
            self._do_discard(seat, action[1])
        elif kind == "peng":
            self._do_peng(seat)
        elif kind == "chi":
            self._do_chi(seat, action[1], action[2])
        elif kind == "gang":
            self._do_gang(seat, action[1], action[2])
        elif kind == "pass":
            self._do_pass(seat)
        return self.events

    # -- 胡

    def _do_hu(self, seat):
        p = self.players[seat]
        res = compute_fan(p.counts, p.meld_count, p.chain_count, p.piao,
                          p.baotou)
        fan = res["fan"]
        is_dealer = seat == self.dealer
        total, losses = settle(fan, self.base, is_dealer, self.n)
        scores = [0] * self.n
        scores[seat] = total
        others = [s for s in range(self.n) if s != seat]
        if is_dealer:
            for s in others:
                scores[s] = -self.base * fan * 8
        else:
            for s in others:
                scores[s] = -(self.base * fan * 8 if s == self.dealer
                              else self.base * fan)
        self.finished = True
        self.phase = PHASE_SETTLED
        self.result = {
            "draw": False, "winner": seat, "fan": fan, "scores": scores,
            "detail": res["detail"], "baotou": res["baotou"],
            "gods": res["gods"],
            "next_dealer": self.dealer if is_dealer
            else (self.dealer + 1) % self.n,
        }
        self._emit("round_ended", {"draw": False, "seat": seat, "fan": fan,
                                   "detail": res["detail"]})

    # -- 打牌

    def _do_discard(self, seat, tile):
        p = self.players[seat]
        was_baotou = p.baotou
        p.counts[tile] -= 1
        p.discards.append(tile)

        if tile == GOD_IDX and was_baotou:
            p.chain_count += 1        # 财飘：爆头态打出财神，链 +1
            p.piao += 1
        else:
            p.chain_count = 0         # 打出其它牌 -> 断链
            p.piao = 0

        self._refresh_baotou(seat)
        self.last_discard = tile
        self.last_discarder = seat
        self.drawn_tile = None
        self._emit("tile_discarded", {"seat": seat, "tile": tile})

        if tile == GOD_IDX:
            self.catch_origin = seat  # 抓打圈开启

        self._open_window()

    def _open_window(self):
        """打牌后依次开碰窗口 -> 吃窗口 -> 下家摸牌。"""
        catch = self.catch_origin >= 0
        tile = self.last_discard
        if tile == GOD_IDX or catch:
            # 财神不可被吃碰杠；抓打圈内其余玩家亦不可吃碰明杠
            self._advance_turn()
            return
        seats = []
        for s in range(self.n):
            if s == self.last_discarder:
                continue
            p = self.players[s]
            if p.counts[tile] >= 2:
                seats.append(s)
        seats.sort(key=lambda s: (s - self.last_discarder) % self.n)
        if seats:
            self.phase = PHASE_PENG
            self.responding_seats = seats
            self.responded = set()
            self._emit("window_open", {"phase": PHASE_PENG, "seats": seats})
            return
        self._open_chi_window()

    def _open_chi_window(self):
        tile = self.last_discard
        nxt = (self.last_discarder + 1) % self.n
        if self._chi_options(nxt, tile):
            self.phase = PHASE_CHI
            self.responding_seats = [nxt]
            self.responded = set()
            self._emit("window_open", {"phase": PHASE_CHI, "seats": [nxt]})
            return
        self._advance_turn()

    def _advance_turn(self):
        """无人响应：下家摸牌。"""
        nxt = (self.last_discarder + 1) % self.n
        if self.catch_origin >= 0 and nxt == self.catch_origin:
            self.catch_origin = -1    # 一圈走完，解除抓打圈
        self.responding_seats = []
        self.responded = set()
        if self.drawable <= 0:
            self._exhaust()
            return
        self._draw(nxt)

    def _do_pass(self, seat):
        self.responded.add(seat)
        if set(self.responding_seats) <= self.responded:
            if self.phase == PHASE_PENG:
                self._open_chi_window()
            else:
                self._advance_turn()

    # -- 碰 / 吃 / 杠

    def _do_peng(self, seat):
        tile = self.last_discard
        p = self.players[seat]
        p.counts[tile] -= 2
        p.melds.append(Meld("peng", tile, [tile] * 3, self.last_discarder))
        p.chain_count = 0
        p.piao = 0
        self._refresh_baotou(seat)
        self._emit("peng", {"seat": seat, "tile": tile})
        self.responding_seats = []
        self.responded = set()
        self.phase = PHASE_DRAW
        self.turn = seat
        self.drawn_tile = None        # 碰后直接打牌，不摸牌

    def _do_chi(self, seat, a, b):
        tile = self.last_discard
        p = self.players[seat]
        p.counts[a] -= 1
        p.counts[b] -= 1
        p.chi_count += 1
        start = min(tile, a, b)
        p.melds.append(Meld("chi", start, sorted([tile, a, b]),
                            self.last_discarder))
        p.chain_count = 0
        p.piao = 0
        self._refresh_baotou(seat)
        self._emit("chi", {"seat": seat, "tile": tile, "used": [a, b]})
        self.responding_seats = []
        self.responded = set()
        self.phase = PHASE_DRAW
        self.turn = seat
        self.drawn_tile = None

    def _do_gang(self, seat, tile, mode):
        p = self.players[seat]
        if mode == "ming":
            p.counts[tile] -= 3
            p.melds.append(Meld("gang_ming", tile, [tile] * 4,
                                self.last_discarder))
        elif mode == "an":
            p.counts[tile] -= 4
            p.melds.append(Meld("gang_an", tile, [tile] * 4, seat))
        else:  # bu：碰后加杠
            p.counts[tile] -= 1
            for m in p.melds:
                if m.kind == "peng" and m.tile == tile:
                    m.kind = "gang_bu"
                    m.tiles = [tile] * 4
                    break
        p.chain_count += 1            # 杠：链 +1（杠开 ×2）
        self._emit("gang", {"seat": seat, "tile": tile, "mode": mode})
        self.responding_seats = []
        self.responded = set()
        self._refresh_baotou(seat)
        if self.tail <= self.live_end or self.tail <= self.head:
            self._exhaust()
            return
        self._draw(seat, from_tail=True)

    # ---------------- 观察 ----------------

    def snapshot(self, seat):
        """与平台快照同构的本人视角局面。"""
        p = self.players[seat]
        return {
            "seat": seat,
            "phase": self.phase,
            "turn": self.turn,
            "dealer": self.dealer,
            "my_hand": p.tiles_in_hand(),
            "melds": [[(m.kind, m.tile) for m in q.melds]
                      for q in self.players],
            "discards": [list(q.discards) for q in self.players],
            "drawn_tile": self.drawn_tile if self.turn == seat else None,
            "last_discard": self.last_discard,
            "responding_seats": list(self.responding_seats),
            "remaining": self.drawable,
            "god": {"baotou": p.baotou, "chain_count": p.chain_count,
                    "piao": p.piao,
                    "catch_play": self.catch_origin >= 0
                    and seat != self.catch_origin},
            "finished": self.finished,
        }

    def shanten_of(self, seat):
        p = self.players[seat]
        return missing(p.counts, p.meld_count) - 1


def play_round(bots, dealer=0, base=1, you_cai_bi_kao=False, seed=None,
               wall=None, max_steps=4000):
    """用 bots（4 个可调用对象）跑完一局，返回 game.result。

    bot 接口：bot(game, seat, legal_actions) -> action
    """
    g = Game(dealer=dealer, base=base, you_cai_bi_kao=you_cai_bi_kao,
             seed=seed, wall=wall)
    steps = 0
    while not g.finished and steps < max_steps:
        steps += 1
        if g.phase == PHASE_DRAW:
            seat = g.turn
            acts = g.legal_actions(seat)
            if not acts:
                g._exhaust()
                break
            g.step(seat, bots[seat](g, seat, acts))
        elif g.phase in (PHASE_PENG, PHASE_CHI):
            pending = [s for s in g.responding_seats if s not in g.responded]
            if not pending:
                g._do_pass(g.responding_seats[0])
                continue
            seat = pending[0]
            acts = g.legal_actions(seat)
            if not acts:
                g.responded.add(seat)
                continue
            g.step(seat, bots[seat](g, seat, acts))
        else:
            break
    if not g.finished:
        g._exhaust()
    return g.result
