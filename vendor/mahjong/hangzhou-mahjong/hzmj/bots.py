"""决策策略（纯标准库）。

    RandomBot   合法动作均匀随机，仅作对照基线
    GreedyBot   能胡就胡、向听数最小化出牌，不做链博弈
    RuleBot     向听 + 加权进张 + 爆头/财飘链博弈 + 副露收益判定（主力）

RuleBot 的核心权衡（杭麻特有）：
    1. 无点炮 -> 防守价值为 0，全部算力用于进攻速度与番型
    2. 爆头态摸到财神时，弃胡打财神（财飘）期望是直接胡的 2 倍
    3. 杠与飘同属动作链，每次 ×2，牌墙足够时值得博
    4. 吃碰破坏七对分支且吃只有 2 摊，需比较副露前后的向听收益

策略开关（S1~S9）
----------------
所有增强策略都挂在 `hzmj.flags` 的开关上，**默认全关**；全关时行为与
未引入开关前逐动作等价（52 项单测 + 配对复式基线均以此为准）。
开关文件 `config/strategy.json` 改完存盘即生效，无需重启 bot。
消融实验通过 `RuleBot(flags=...)` 显式绑定，不受文件影响。
"""

import random

from . import flags as flagmod
from .hu import (is_baotou, missing, qidui_missing, standard_missing,
                 useful_tiles)
from .prior import p_hu
from .tiles import GOD_IDX, TILE_KINDS, TILES_PER_KIND

FULL_WALL = 63.0         # 开局可摸张数

# S2 番期望目标函数的可调系数（`RuleBot(params=...)` 可逐项覆盖，供扫参用）
DEFAULT_PARAMS = {
    "fan_gamma": 0.65,        # 番的阻尼指数：番是 2 的幂，直接线性用会过度冒进
    "qidui_discount": 0.55,   # 七对分支的可达性折价（实测七对仅占胡牌 1.7%）
    "slack_margin": 1.35,     # 容许缺口劣化 1 时，EV 须超过最优缺口候选的倍数
    "god_hold_wall": 20,      # S5：牌墙剩这么多以上才肯为留财神让缺口
    "god_hold_min": 2,        # S5：手上至少这么多财神才值得让缺口
}

FAN_GAMMA = DEFAULT_PARAMS["fan_gamma"]
QIDUI_DISCOUNT = DEFAULT_PARAMS["qidui_discount"]
SLACK_MARGIN = DEFAULT_PARAMS["slack_margin"]


def seen_counts(game, seat, *, public_events=None):
    """本家手牌 + 全场副露 + 牌河，扣除副露引用的来源弃牌。

    引擎牌河保留所有弃牌。整局公开事件能确定吃牌实际引用的牌，
    碰转补杠只扣一次；暗杠不引用弃牌。public_events 须覆盖当前整局。
    仅快照无历史时，碰/明杠/补杠可从副露确定来源牌；吃无法确定，
    保留保守计数，不把顺子起点误当来源牌。
    """
    seen = [0] * TILE_KINDS
    for i, c in enumerate(game.players[seat].counts):
        seen[i] += c
    for p in game.players:
        for m in p.melds:
            for t in m.tiles:
                seen[t] += 1
        for t in p.discards:
            seen[t] += 1
    events = public_events if public_events is not None else getattr(game, "events", None)
    if events is not None:
        for event in events:
            kind, data = event.get("kind"), event.get("data", {})
            if kind in ("chi", "peng") or (kind == "gang" and data.get("mode") == "ming"):
                seen[data["tile"]] -= 1
    else:
        for p in game.players:
            for meld in p.melds:
                if meld.kind in ("peng", "gang_ming", "gang_bu", "碰"):
                    seen[meld.tile] -= 1
    return seen


def remaining(seen, tile):
    return max(0, TILES_PER_KIND - seen[tile])


def hand_value(counts, meld_count, seen):
    """(缺口数, 加权进张数)：缺口越小越好，进张越多越好。"""
    m = missing(counts, meld_count)
    gain = sum(remaining(seen, t) for t in useful_tiles(counts, meld_count))
    return m, gain


class RandomBot:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, game, seat, actions):
        return self.rng.choice(actions)


class GreedyBot:
    """能胡即胡；出牌取缺口最小、进张最多者。不参与链博弈。"""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, game, seat, actions):
        for a in actions:
            if a[0] == "hu":
                return a
        discards = [a for a in actions if a[0] == "discard"]
        if discards:
            return best_discard(game, seat, discards)
        return next((a for a in actions if a[0] == "pass"), actions[0])


def best_discard(game, seat, discards, keep_god_bonus=True):
    """出牌选择：先按缺口数筛选，再在最优组内比加权进张（省去无谓的进张枚举）。"""
    p = game.players[seat]
    scored = []
    best_m = None
    for a in discards:
        t = a[1]
        p.counts[t] -= 1
        m = missing(p.counts, p.meld_count)
        p.counts[t] += 1
        scored.append((m, a))
        if best_m is None or m < best_m:
            best_m = m
    finalists = [a for m, a in scored if m == best_m]
    if len(finalists) == 1:
        return finalists[0]

    seen = seen_counts(game, seat)
    best, best_key = None, None
    for a in finalists:
        t = a[1]
        p.counts[t] -= 1
        gain = sum(remaining(seen, x)
                   for x in useful_tiles(p.counts, p.meld_count))
        baotou_bonus = 1 if is_baotou(p.counts, p.meld_count) else 0
        p.counts[t] += 1
        god_penalty = 1 if (keep_god_bonus and t == GOD_IDX) else 0
        key = (-baotou_bonus, god_penalty, -gain, -t)
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


class RuleBot:
    """主力策略：向听 + 加权进张 + 动作链博弈。

    style: balanced（默认） / aggressive（更愿弃胡博链） / safe（能胡即胡）
    flags: 显式绑定策略开关（消融用）；None 表示跟随 config/strategy.json
    """

    def __init__(self, seed=None, style="balanced", epsilon=0.0, flags=None,
                 params=None):
        self.rng = random.Random(seed)
        self.style = style
        self.epsilon = epsilon
        self._bound = dict(flags) if flags is not None else None
        self._f = self._bound if self._bound is not None else flagmod.DEFAULTS
        self._p = dict(DEFAULT_PARAMS)
        if params:
            unknown = set(params) - set(DEFAULT_PARAMS)
            if unknown:
                raise KeyError("未知策略参数 %s" % sorted(unknown))
            self._p.update(params)

    @property
    def flags(self):
        return self._bound if self._bound is not None else flagmod.current()

    def __call__(self, game, seat, actions):
        # 每次决策取一次开关并缓存，避免决策链里反复查文件 mtime
        self._f = self._bound if self._bound is not None else flagmod.current()
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        if game.phase in ("response_peng", "response_chi"):
            return self._respond(game, seat, actions)
        return self._on_turn(game, seat, actions)

    # ---------------- 自己回合 ----------------

    def _on_turn(self, game, seat, actions):
        hu_act = next((a for a in actions if a[0] == "hu"), None)
        gangs = [a for a in actions if a[0] == "gang"]
        discards = [a for a in actions if a[0] == "discard"]

        if hu_act is not None:
            piao = self._piao_option(game, seat, discards)
            if piao is not None and self._should_piao(game, seat):
                return piao
            switch = self._baotou_switch(game, seat, discards)
            if switch is not None:
                return switch
            g = self._gang_before_hu(game, seat, gangs)
            if g is not None:
                return g
            return hu_act

        if gangs:
            g = self._pick_gang(game, seat, gangs)
            if g is not None:
                return g
        if discards:
            dump = self._ycbk_dump(game, seat, discards)
            if dump is not None:
                return dump
            catch = self._tactical_catch(game, seat, discards)
            if catch is not None:
                return catch
            hold = self._god_hold(game, seat, discards)
            if hold is not None:
                return hold
            return self._choose_discard(game, seat, discards)
        return actions[0]

    def _missing_map(self, game, seat, discards):
        """{动作: 打出后的缺口数}，顺带返回最小缺口。"""
        p = game.players[seat]
        out, best = [], None
        for a in discards:
            t = a[1]
            p.counts[t] -= 1
            m = missing(p.counts, p.meld_count)
            p.counts[t] += 1
            out.append((m, a))
            if best is None or m < best:
                best = m
        return out, best

    def _ycbk_dump(self, game, seat, discards):
        """S9：「有财必拷响」下手持财神禁平胡 —— 抱着不能赢的牌等于自动流局。

        `hu.can_hu` 在 `you_cai_bi_kao ∧ 手留财神 ∧ 非爆头` 时直接返回 False，
        所以这种局面**必须**二选一：切成爆头，或者把财神清掉。基线两件都不做
        （财神永远排最后），于是听牌了也胡不了。

        优先切爆头（保住万能牌 + 番 ×2）；不行才弃财神，且只在弃后缺口不劣化
        超过 1 时弃 —— 否则等于为解封拆掉整手牌。
        """
        f = self._f
        if not (f["S9_ycbk_mode"] and game.you_cai_bi_kao):
            return None
        p = game.players[seat]
        if p.counts[GOD_IDX] <= 0 or p.baotou:
            return None
        scored, base = self._missing_map(game, seat, discards)
        if base > 1:                      # 还没听牌，财神的万能性仍然更值钱
            return None

        # ① 切成爆头：财神留着也能胡，且番 ×2
        for m, a in scored:
            if a[1] == GOD_IDX or m > base:
                continue
            t = a[1]
            p.counts[t] -= 1
            ok = is_baotou(p.counts, p.meld_count)
            p.counts[t] += 1
            if ok:
                return a

        # ② 清财神解封。只在**零代价**时做 —— 早期版本容许劣化 1，实测
        #    −3.454 ± 1.799（n=240，ycbk）：为解封拆掉一步向听远不如干等爆头。
        god = next(((m, a) for m, a in scored if a[1] == GOD_IDX), None)
        if god is not None and god[0] <= base:
            return god[1]
        return None

    def _god_hold(self, game, seat, discards):
        """S5：留财神让缺口 —— **实测为不可达分支，保留仅作反证**。

        原设想：当「唯一的最小缺口选择就是打财神」时改打次优的非财神牌。
        但财神是万能牌，拿掉它只会让缺口不变或变大，而拿掉一张废牌一定让缺口
        不变，故 `缺口(打财神) ≥ 缺口(打废牌)` 恒成立 —— 前提永远不满足，
        触发率恒为 0.0%（n=240 实测）。

        结论：**「财神定价」在出牌阶段不存在决策面**。S5 唯一真实的落点是
        `_should_piao` 里「飘完手留白+已飘=4」那个平局判定。
        """
        f = self._f
        if not f["S5_god_nonlinear"] or game.you_cai_bi_kao:
            return None
        p = game.players[seat]
        if p.counts[GOD_IDX] < self._p["god_hold_min"]:
            return None
        if game.drawable < self._p["god_hold_wall"]:
            return None
        scored, base = self._missing_map(game, seat, discards)
        mins = [a for m, a in scored if m == base]
        if len(mins) != 1 or mins[0][1] != GOD_IDX:
            return None
        alts = [a for m, a in scored if m == base + 1 and a[1] != GOD_IDX]
        if not alts:
            return None
        return best_discard(game, seat, alts)

    def _piao_option(self, game, seat, discards):
        """爆头态且手上有财神 -> 打出财神即「飘」，链 +1。"""
        p = game.players[seat]
        if not p.baotou or p.counts[GOD_IDX] <= 0:
            return None
        return next((a for a in discards if a[1] == GOD_IDX), None)

    def _baotou_switch(self, game, seat, discards):
        """杭麻核心 alpha：可胡但非爆头时，若打出某张即成爆头态，则弃胡改听爆头。

        爆头保证下一轮摸任意牌都能胡，代价仅是让出一圈；番型直接 ×2，
        且若下轮摸进财神还能续飘 ×4。剩余牌墙不足或已是爆头态时不做。
        """
        f = self._f
        p = game.players[seat]
        if p.baotou or self.style == "safe" or game.drawable < 2:
            return None
        # S4：末盘弃胡博爆头的风险陡增，牌墙薄了直接落袋
        if f["S4_wall_timeline"] and game.drawable < 10:
            return None
        # S3：庄家「胡 vs 不胡」的分差是闲家的 2.4 倍，落袋价值更高
        if f["S3_dealer_asymmetry"] and game.dealer == seat \
                and game.drawable < 16:
            return None
        # S7：有人明显要胡大牌时，抢先落袋优于博 ×2
        if f["S7_opponent_threat"] and self._threat(game, seat) >= 1.2:
            return None
        cands = []
        for a in discards:
            t = a[1]
            p.counts[t] -= 1
            ok = is_baotou(p.counts, p.meld_count)
            p.counts[t] += 1
            if ok:
                cands.append(a)
        if not cands:
            return None
        # 优先打非财神（留财神续飘）；其次打大牌号（信息价值低）
        cands.sort(key=lambda a: (a[1] == GOD_IDX, -a[1]))
        return cands[0]

    def _should_piao(self, game, seat):
        """弃胡打财神博财飘：飘出后必须仍爆头，且牌墙还摸得到。"""
        if self.style == "safe":
            return False
        f = self._f
        p = game.players[seat]
        if game.drawable < 2 or p.chain_count >= 5:
            return False
        if f["S4_wall_timeline"] and game.drawable < 8:
            return False
        p.counts[GOD_IDX] -= 1
        still = is_baotou(p.counts, p.meld_count)
        p.counts[GOD_IDX] += 1
        if not still:
            return False
        # v23 四白板与爆头可叠加，不再因四白板排除财飘。
        return True

    def _gang_before_hu(self, game, seat, gangs):
        """已可胡时先暗杠博杠开 ×2：要求杠后仍保持可胡或听牌。"""
        if not gangs or self.style == "safe" or game.drawable <= 0:
            return None
        loose = self._f["S1_loose_gang"]
        p = game.players[seat]
        for a in gangs:
            mode = a[2]
            if mode != "an" and not loose:
                continue
            tile = a[1]
            take = 4 if mode == "an" else 1
            p.counts[tile] -= take
            mc = p.meld_count + (0 if mode == "bu" else 1)
            ok = missing(p.counts, mc) <= 1
            p.counts[tile] += take
            if ok:
                return a
        return None

    def _pick_gang(self, game, seat, gangs):
        """未听牌时的杠：不劣化缺口即杠（链 ×2 且多摸一张）。"""
        f = self._f
        loose = f["S1_loose_gang"]
        floor = 0 if loose else 1
        if game.drawable <= floor:
            return None
        if f["S4_wall_timeline"] and game.drawable < 4:
            return None
        p = game.players[seat]
        base = missing(p.counts, p.meld_count)
        # 已接近七对成型时不杠（杠即副露，七对分支作废）
        if qidui_missing(p.counts, p.meld_count) <= 1 < base:
            return None
        # S1：杠 = 动作链 ×2 + 多摸一张，杭麻无点炮故零风险，
        #     容许缺口暂时劣化 1；补杠只丢一张手牌，最便宜，优先。
        slack = 1 if loose else 0
        best, best_key = None, None
        for a in gangs:
            tile, mode = a[1], a[2]
            take = 4 if mode == "an" else 1
            p.counts[tile] -= take
            after = missing(p.counts, p.meld_count + (0 if mode == "bu" else 1))
            p.counts[tile] += take
            if after > base + slack:
                continue
            if not loose:
                return a
            key = (after, 0 if mode == "bu" else 1, -tile)
            if best_key is None or key < best_key:
                best, best_key = a, key
        return best

    # ---------------- 出牌 ----------------

    def _choose_discard(self, game, seat, discards):
        """出牌路由。

        S2 是「换目标函数」（EV 排序 + 容许缺口劣化），S3/S4/S7 是它的**权重
        调制项**，单独开无处生效 —— 所以它们只在 S2 打开时才有意义，
        评测须走 `S2+Sx` 组合。
        S5/S9 是**独立**的加项，故走 `_tweaked_discard`：键结构与基线逐位相同，
        只替换财神项，从而把它们的效应与「切换到 EV 排序」解耦。
        """
        f = self._f
        if f["S2_fan_ev_discard"]:
            return self._ev_discard(game, seat, discards)
        if f["S5_god_nonlinear"] or f["S9_ycbk_mode"]:
            return self._tweaked_discard(game, seat, discards)
        return best_discard(game, seat, discards)

    def _tweaked_discard(self, game, seat, discards):
        """基线出牌 + S5 财神凸权重 / S9 有财必拷响修正。

        与 `best_discard` 的唯一差别是把二元的 `god_penalty` 换成连续代价，
        其余键（爆头优先、进张、牌号）逐位保持一致。
        """
        f = self._f
        p = game.players[seat]
        scored = []
        best_m = None
        for a in discards:
            t = a[1]
            p.counts[t] -= 1
            m = missing(p.counts, p.meld_count)
            p.counts[t] += 1
            scored.append((m, a))
            if best_m is None or m < best_m:
                best_m = m
        finalists = [a for m, a in scored if m == best_m]
        if len(finalists) == 1:
            return finalists[0]

        seen = seen_counts(game, seat)
        held_god = p.counts[GOD_IDX]
        ycbk = f["S9_ycbk_mode"] and game.you_cai_bi_kao
        best, best_key = None, None
        for a in finalists:
            t = a[1]
            p.counts[t] -= 1
            gain = sum(remaining(seen, x)
                       for x in useful_tiles(p.counts, p.meld_count))
            bt = is_baotou(p.counts, p.meld_count)
            p.counts[t] += 1
            # 基线把财神当「永远最后打」（等价于代价 1.0）。S5 改成按持有数定价：
            # 持 3 张时第 4 张能凑 4 白板 ×2，几乎不该打；只持 1 张且帮不上
            # 牌形时，硬留着反而占位。
            score = (2.0 if bt else 1.0) * (1.0 + gain / 12.0)
            if t == GOD_IDX:
                cost = (self._god_cost(held_god, game.drawable)
                        if f["S5_god_nonlinear"] else 1.0)
                score *= max(0.0, 1.0 - cost)
            if ycbk:
                score *= self._ycbk_factor(t, bt, held_god)
            key = (-score, 1 if t == GOD_IDX else 0, -t)
            if best_key is None or key < best_key:
                best, best_key = a, key
        return best

    def _ev_discard(self, game, seat, discards):
        """S2：按 EV = P(胡) × E[番] 选牌，容许缺口劣化 1 换更高番型。

        P(胡) 取自 hzmj.prior 的经验先验表（自带「被对手抢先」的竞争因素），
        E[番] 由分支因子 / 动作链 / 爆头可达性 / 白板张数合成。

        缺口劣化的候选须以 SLACK_MARGIN 倍优势胜出才被采纳 —— 否则
        「用速度换番」会退化成无脑追大牌（实测未加余量时掉 2.6 分/局）。
        """
        f = self._f
        p = game.players[seat]
        drawable = game.drawable
        seen = seen_counts(game, seat)
        unseen = max(1, sum(TILES_PER_KIND - seen[t]
                            for t in range(TILE_KINDS)))

        scored = []
        best_m = None
        for a in discards:
            t = a[1]
            p.counts[t] -= 1
            m = missing(p.counts, p.meld_count)
            p.counts[t] += 1
            scored.append((m, a))
            if best_m is None or m < best_m:
                best_m = m
        cands = [(m, a) for m, a in scored if m <= best_m + 1]
        if len(cands) == 1:
            return cands[0][1]

        fan_w = self._fan_weight(game, seat) * self._p["fan_gamma"]
        held_god = p.counts[GOD_IDX]
        tiers = {}                       # 缺口数 -> (最佳分, 动作)
        for m, a in cands:
            t = a[1]
            p.counts[t] -= 1
            gain = sum(remaining(seen, x)
                       for x in useful_tiles(p.counts, p.meld_count))
            bt = is_baotou(p.counts, p.meld_count)
            fan = self._expected_fan(game, seat, m, bt, drawable, unseen)
            p.counts[t] += 1

            score = p_hu(m, drawable) * ((max(gain, 1) / 20.0) ** 0.35) \
                * (fan ** fan_w)
            if f["S5_god_nonlinear"] and t == GOD_IDX:
                score *= 1.0 - self._god_cost(held_god, drawable)
            if f["S9_ycbk_mode"] and game.you_cai_bi_kao:
                score *= self._ycbk_factor(t, bt, held_god)
            cur = tiers.get(m)
            if cur is None or score > cur[0] or (
                    score == cur[0] and (t != GOD_IDX, -t) > cur[2]):
                tiers[m] = (score, a, (t != GOD_IDX, -t))

        top = tiers.get(best_m)
        alt = tiers.get(best_m + 1)
        if top is None:
            return alt[1]
        if alt is not None and alt[0] > top[0] * self._slack_margin(game, seat):
            return alt[1]
        return top[1]

    def _slack_margin(self, game, seat):
        """「用速度换番」的准入门槛倍数。越大越保守（越倾向保持最小缺口）。

        S3/S4/S7 的真正落点在这里 —— 它们改的是**决策边界**，而不只是 `fan_w`
        指数（指数对两档候选近似等比缩放，几乎翻不动 argmax，实测 0 触发）。
        """
        f = self._f
        m = self._p["slack_margin"]
        if f["S3_dealer_asymmetry"]:
            # 庄家「胡 vs 不胡」分差 32、闲家 13.3 -> 庄家先落袋，闲家可博
            m *= 1.45 if game.dealer == seat else 0.85
        if f["S4_wall_timeline"]:
            # 开局摸牌次数多，博番的时间成本便宜；末盘只求落袋
            m *= 1.55 - 0.75 * min(1.0, game.drawable / FULL_WALL)
        if f["S7_opponent_threat"]:
            m *= 1.0 + 0.25 * self._threat(game, seat)
        return m

    def _expected_fan(self, game, seat, m, baotou, drawable, unseen):
        """当前牌形的期望番（不含底分与庄闲系数 —— 它们对本家各选项等比）。"""
        p = game.players[seat]
        fan = 2.0 ** min(p.chain_count, 6)
        if p.meld_count == 0:
            q = qidui_missing(p.counts, 0)
            s = standard_missing(p.counts, 0)
            quads = min(3, sum(1 for c in p.counts if c == 4))
            # 七对分支名义 ×2^(1+quads)，但实测仅占胡牌 1.7% —— 按可达性折价，
            # 否则 EV 会把每手牌都往七对上拐。
            qfan = 1.0 + (2.0 * (2 ** quads) - 1.0) * self._p["qidui_discount"]
            if q < s:
                fan *= qfan
            elif q == s:
                fan *= (1.0 + qfan) / 2.0
        if baotou:
            fan *= 2.0
        elif m == 1:
            fan *= 1.25          # 听牌未爆头，后续仍有切成爆头的机会
        gods = p.counts[GOD_IDX] + p.piao
        if gods >= 4:
            fan *= 2.0
        elif gods == 3 and self._f["S5_god_nonlinear"]:
            # 差一张凑 4 白板 ×2：按「还剩几次摸牌 × 白板剩余密度」折现
            draws = max(0.0, drawable / 4.0)
            fan *= 1.0 + min(0.6, draws * 4.0 / unseen)
        return fan

    def _fan_weight(self, game, seat):
        """番在 EV 里的指数权重。>1 更博番，<1 更求速。"""
        f = self._f
        w = 1.0
        if f["S3_dealer_asymmetry"]:
            # 庄家「胡 vs 不胡」分差 32，闲家仅 13.3 -> 庄家更该先把胡拿到手
            w *= 0.72 if game.dealer == seat else 1.15
        if f["S4_wall_timeline"]:
            # 开局有的是摸牌次数，值得博番；末盘只求落袋
            w *= 0.45 + 0.85 * min(1.0, game.drawable / FULL_WALL)
        if f["S7_opponent_threat"]:
            w *= max(0.6, 1.0 - 0.30 * self._threat(game, seat))
        return w

    def _god_cost(self, held, drawable):
        """S5：打出财神的相对代价（0~1），随手上张数凸变化。

        基线等价于「代价恒为 1.0」= 财神永远最后打。实测这过度保守：只持 1 张
        且帮不上牌形时，硬留着等于少一个进张位。持 3 张时反过来 —— 第 4 张就是
        4 白板 ×2，代价接近 1。持满 4 张不算爆头，必须打掉一张，代价回落。
        """
        if held <= 0:
            return 0.0
        base = {1: 0.35, 2: 0.55, 3: 0.88, 4: 0.45}.get(held, 0.35)
        return base * (0.55 + 0.45 * min(1.0, drawable / FULL_WALL))

    def _ycbk_factor(self, tile, baotou, held_god):
        """S9：有财必拷响 —— 手持财神禁平胡，须清财神或转爆头。"""
        if held_god <= 0 or baotou:
            return 1.0
        if tile == GOD_IDX:
            return 1.6                       # 打掉财神即解除封锁
        return 0.75

    def _tactical_catch(self, game, seat, discards):
        """S8：非爆头态主动打财神开抓打圈，冻结三家一圈（只能打刚摸的牌）。

        代价是丢掉一张万能牌且断链，所以只在「自己本来就没戏 + 对手威胁高 +
        牌墙还长」三条同时成立时才用。
        """
        if not self._f["S8_catch_play_tactic"]:
            return None
        p = game.players[seat]
        if p.baotou or p.counts[GOD_IDX] <= 0 or p.chain_count:
            return None
        if game.drawable < 8:
            return None
        if missing(p.counts, p.meld_count) <= 2:
            return None
        if self._threat(game, seat) < 1.0:
            return None
        return next((a for a in discards if a[1] == GOD_IDX), None)

    def _threat(self, game, seat):
        """S7：从公开信息估计对手离胡的近度，取全场最高者。"""
        worst = 0.0
        for s in range(4):
            if s == seat:
                continue
            p = game.players[s]
            v = 0.34 * p.meld_count
            v += 0.30 * sum(1 for m in p.melds if m.is_gang)
            d = p.discards
            if len(d) >= 5:
                # 早巡就把字牌打光 = 手牌整齐，进度快
                v += 0.12 * sum(1 for t in d[:5] if t >= 27)
            if v > worst:
                worst = v
        return worst

    # ---------------- 响应窗口 ----------------

    def _respond(self, game, seat, actions):
        f = self._f
        pass_act = next((a for a in actions if a[0] == "pass"), None)
        p = game.players[seat]
        tile = game.last_discard
        base = missing(p.counts, p.meld_count)
        qidui = qidui_missing(p.counts, p.meld_count)

        # 仅在七对已接近成型且优于普通胡时放弃副露
        if qidui < base and qidui <= 2:
            return pass_act or actions[0]

        best, best_key = None, None
        best_gang, best_gang_after = None, None
        for a in actions:
            if a[0] == "pass":
                continue
            after = self._meld_missing(game, seat, a, tile)
            if after is None:
                continue
            gain = 1 if a[0] == "gang" else 0
            key = (after, -gain)
            if best_key is None or key < best_key:
                best, best_key = a, key
            if a[0] == "gang" and (best_gang_after is None
                                   or after < best_gang_after):
                best_gang, best_gang_after = a, after

        # S1：明杠自带动作链 ×2 + 补摸一张，不该跟碰吃挤同一条准入通道
        if f["S1_loose_gang"] and best_gang is not None \
                and best_gang_after <= base and game.drawable > 0:
            return best_gang

        if best is None:
            return pass_act or actions[0]
        after = best_key[0]
        if after < base:
            return best
        if f["S6_meld_cost"] and self._meld_worth(game, seat, best, base,
                                                  after):
            return best
        if after == base and best[0] in ("peng", "gang") and base <= 2:
            return best
        return pass_act or actions[0]

    def _meld_worth(self, game, seat, action, base, after):
        """S6：副露的两笔漏账。

        支出：碰/吃后 `drawn_tile = None`，本轮**不摸牌** ——
              等于拿一张确定的牌换掉一次随机自摸。
        收入：碰/杠跳过中间座位的自摸，直接削弱对手；跳过庄家尤其值钱
              （庄家胡我付 8，其他闲家胡我只付 1）。
        原实现两笔都没记，只比副露前后的缺口数。
        """
        p = game.players[seat]
        seen = seen_counts(game, seat)
        unseen = max(1, sum(TILES_PER_KIND - seen[t]
                            for t in range(TILE_KINDS)))
        useful = useful_tiles(p.counts, p.meld_count)
        # 一次随机自摸的期望缺口下降
        draw_ev = sum(remaining(seen, t) * (base - m)
                      for t, m in useful.items()) / unseen
        certain = base - after

        skipped = ((seat - game.last_discarder) % 4) - 1
        bonus = 0.10 * max(0, skipped)
        if game.dealer != seat and skipped > 0:
            step = (game.dealer - game.last_discarder) % 4
            if 1 <= step <= skipped:
                bonus += 0.16          # 被跳过的正好是庄家
        if action[0] == "gang":
            bonus += 0.30              # 杠额外补摸一张，支出被抵消
        return certain + bonus > draw_ev

    def _meld_missing(self, game, seat, action, tile):
        """试算副露后的缺口数；不可行返回 None。"""
        p = game.players[seat]
        kind = action[0]
        if kind == "peng":
            if p.counts[tile] < 2:
                return None
            p.counts[tile] -= 2
            m = missing(p.counts, p.meld_count + 1)
            p.counts[tile] += 2
            return m
        if kind == "gang":
            if p.counts[tile] < 3:
                return None
            p.counts[tile] -= 3
            m = missing(p.counts, p.meld_count + 1)
            p.counts[tile] += 3
            return m
        if kind == "chi":
            a, b = action[1], action[2]
            if p.counts[a] < 1 or p.counts[b] < 1:
                return None
            p.counts[a] -= 1
            p.counts[b] -= 1
            m = missing(p.counts, p.meld_count + 1)
            p.counts[a] += 1
            p.counts[b] += 1
            return m
        return None


BOT_REGISTRY = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "rule": RuleBot,
}


def make_bot(name, seed=None, **kw):
    cls = BOT_REGISTRY[name]
    if cls is RuleBot:
        return RuleBot(seed=seed, **kw)
    return cls(seed=seed)
