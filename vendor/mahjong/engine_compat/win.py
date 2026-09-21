"""有财必拷的杠开豁免独立于爆头倍率，环境与平台共用。"""
import copy
from functools import wraps


def install():
    from hzmj import adapter, hu
    from hzmj.engine import Game

    if getattr(Game._can_hu, '_gang_hu_compat', False):
        return

    original_hu = Game._can_hu
    original_snapshot = adapter.from_snapshot
    original_legal = adapter.legal_actions

    @wraps(original_hu)
    def can_hu(self, seat):
        p = self.players[seat]
        # 胡的刚摸牌门禁仍由legal_actions负责；杠开不等于爆头。
        return hu.can_hu(p.counts, p.meld_count,
                         self.you_cai_bi_kao and not self.gang_supplement,
                         p.baotou)

    @wraps(original_snapshot)
    def from_snapshot(snapshot, you_cai_bi_kao=False):
        view, seat = original_snapshot(snapshot, you_cai_bi_kao)
        god = snapshot.get('god') or {}
        view.catch_origin = god.get('god_discarder_seat', -1)
        view.god_discarder_seat = view.catch_origin
        view.catch_play = bool(god.get('catch_play')) and view.catch_origin != seat
        view.gang_supplement = bool(snapshot.get('gang_supplement', False))
        if seat >= 0 and 'piao' in god:
            p = view.players[seat]
            # 显式0表示纯杠链，不能用“or chain_count”回填成财飘。
            p.piao = max(0, min(p.chain_count, int(god['piao'] or 0)))
        return view, seat

    @wraps(original_legal)
    def legal_actions(view, seat):
        if view.you_cai_bi_kao and getattr(view, 'gang_supplement', False):
            # 原函数仍检查阶段、轮次、刚摸牌与手牌张数；不修改调用者视图。
            view = copy.copy(view)
            view.you_cai_bi_kao = False
        return original_legal(view, seat)

    can_hu._gang_hu_compat = True
    Game._can_hu = can_hu
    adapter.from_snapshot = from_snapshot
    adapter.legal_actions = legal_actions
