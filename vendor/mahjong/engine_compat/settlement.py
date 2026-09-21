"""平台赢家坐庄规则；保留上游单局番数与得分计算。"""
from functools import wraps


def install():
    from hzmj.engine import Game

    original = Game._do_hu
    if getattr(original, '_winner_dealer_compat', False):
        return

    @wraps(original)
    def hu(self, seat):
        result = original(self, seat)
        # 已结束线上对局逐局验证：谁胡谁坐庄；流局沿用 _exhaust 的庄位。
        self.result['next_dealer'] = seat
        return result

    hu._winner_dealer_compat = True
    Game._do_hu = hu
