"""固定开放响应窗口；抓打圈内仅发起者豁免，不按暗手筛选窗口。"""
from functools import wraps


def install():
    from hzmj.engine import Game, PHASE_PENG, PHASE_CHI
    from hzmj.tiles import GOD_IDX

    original_peng = Game._open_window
    original_chi = Game._open_chi_window
    if getattr(original_peng, '_catch_exemption_compat', False):
        return

    @wraps(original_peng)
    def open_window(self):
        origin, tile = self.catch_origin, self.last_discard
        if tile == GOD_IDX:
            return original_peng(self)
        if self.last_discarder == origin:
            # 发起者吃碰后打普通牌：链由原弃牌方法清零，抓打圈同时解除。
            self.catch_origin = -1
            origin = -1
        # 普通窗口固定问其余三家；抓打圈只问豁免方，有无可碰牌都可 pass。
        seats = ([origin] if origin >= 0 else
                 [(self.last_discarder + offset) % self.n for offset in range(1, self.n)])
        self.phase = PHASE_PENG
        self.responding_seats = seats
        self.responded = set()
        self._emit('window_open', {'phase': PHASE_PENG, 'seats': seats[:]})

    @wraps(original_chi)
    def open_chi_window(self):
        nxt = (self.last_discarder + 1) % self.n
        if self.catch_origin >= 0 and nxt != self.catch_origin:
            return self._advance_turn()
        self.phase = PHASE_CHI
        self.responding_seats = [nxt]
        self.responded = set()
        self._emit('window_open', {'phase': PHASE_CHI, 'seats': self.responding_seats[:]})

    open_window._catch_exemption_compat = True
    Game._open_window = open_window
    Game._open_chi_window = open_chi_window
