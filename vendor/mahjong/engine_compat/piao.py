"""平台吃碰后打白的财飘语义，在 submodule 外适配。"""
from functools import wraps


def claim_piao(remaining_counts, meld_count, drawn_tile):
    """吃碰后没有摸牌；打白后仍听任意牌，平台将这次打白计为财飘。"""
    from hzmj import hu

    return drawn_tile in (None, '') and hu.is_baotou(remaining_counts, meld_count)


def discard_chain(player, remaining_counts, tile, drawn_tile):
    """环境和DP3使用相同的财飘判断；remaining_counts为弃牌后的手牌。"""
    piao = tile == 33 and (player.baotou or claim_piao(
        remaining_counts, player.meld_count, drawn_tile))
    return (player.chain_count + 1, player.piao + 1) if piao else (0, 0)


def install():
    from hzmj.engine import Game
    from hzmj.tiles import GOD_IDX

    original = Game._do_discard
    if getattr(original, '_claim_piao_compat', False):
        return

    @wraps(original)
    def discard(self, seat, tile):
        player = self.players[seat]
        if tile == GOD_IDX and not player.baotou and self.drawn_tile is None:
            remaining = list(player.counts)
            remaining[tile] -= 1
            if discard_chain(player, remaining, tile, self.drawn_tile)[0]:
                # 原实现负责链计数、弃牌、重新计算 baotou 和开窗。
                player.baotou = True
        return original(self, seat, tile)

    discard._claim_piao_compat = True
    Game._do_discard = discard

    def preserve_claim_state(original):
        @wraps(original)
        def claim(self, seat, *args):
            player = self.players[seat]
            previous = (player.chain_count, player.piao)
            previous_baotou = player.baotou
            exempt = self.catch_origin == seat
            result = original(self, seat, *args)
            # 吃碰后没有摸牌，平台保留旧爆头状态，等弃牌后再刷新。
            player.baotou = previous_baotou
            if exempt:
                # 圈内吃碰不打断旧链；之后打普通牌仍由_do_discard清零。
                player.chain_count, player.piao = previous
            return result
        return claim

    Game._do_peng = preserve_claim_state(Game._do_peng)
    Game._do_chi = preserve_claim_state(Game._do_chi)
