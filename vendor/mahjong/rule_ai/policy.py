"""当前规则 AI：修复版 score_dp3，三步无放回摸牌纯得分估值。"""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import os
import tempfile

from hzmj import bots as BASE
from mahjong.engine_compat.piao import discard_chain

PACKAGE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def native():
    source = PACKAGE / 'search_native.cpp'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = PACKAGE / '.build' / f'search_{digest}.so'
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        # 每个进程编译独立临时文件，原子发布，避免多 worker 首次加载半成品。
        fd, temporary = tempfile.mkstemp(prefix='search_', suffix='.so', dir=target.parent)
        os.close(fd)
        try:
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-pthread',
                            str(source), '-o', temporary], check=True, capture_output=True)
            os.replace(temporary, target)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('score_dp3 native 编译失败；请在 Linux/WSL 安装 g++，不回退旧策略') from exc
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.rb_missing.argtypes = [ptr, ctypes.c_int]
    lib.rb_missing.restype = ctypes.c_int
    lib.rb_discard.argtypes = [ptr,ptr,ctypes.c_int,ptr]
    lib.rb_discard.restype = ctypes.c_int
    lib.rb_value.argtypes = [ptr, ptr] + [ctypes.c_int]*4 + [ctypes.c_double]*3 + [ctypes.c_int]*2 + [ctypes.POINTER(ctypes.c_double)]
    lib.rb_value.restype = None
    return lib


def native_missing(counts, melds):
    return native().rb_missing((ctypes.c_int*34)(*counts), melds)


def native_discard(game,seat,discards):
    p=game.players[seat];allowed=[0]*34
    for a in discards:allowed[a[1]]=1
    remain=[max(0,4-c) for c in BASE.seen_counts(game,seat)]
    tile=native().rb_discard((ctypes.c_int*34)(*p.counts),(ctypes.c_int*34)(*remain),
                             p.meld_count,(ctypes.c_int*34)(*allowed))
    if tile<0:raise ValueError('没有合法弃牌')
    return ('discard',tile)


@dataclass(frozen=True)
class Estimate:
    score: float
    win: float
    nodes: int


def value(counts, remain, melds=0, depth=2, chain=0, piao=0,
          unit=10., bonus=12., survival=1., ycbk=False, improvements=True):
    if len(counts) != 34 or len(remain) != 34:
        raise ValueError('需要 34 种牌计数')
    if any(c < 0 or r < 0 or c+r > 4 for c,r in zip(counts,remain)):
        raise ValueError('手牌和未知牌必须满足每种牌最多四张')
    if sum(counts) != 13-3*melds or not 0 <= melds <= 4 or not 0 <= depth <= 4:
        raise ValueError('DP 入口必须为待摸牌状态，horizon 不超过四次摸牌')
    out = (ctypes.c_double*3)()
    native().rb_value((ctypes.c_int*34)(*counts), (ctypes.c_int*34)(*remain),
                     melds, depth, chain, piao, unit, bonus, survival,
                     int(ycbk), int(improvements), out)
    return Estimate(out[0],out[1],int(out[2]))


class RuleBot(BASE.RuleBot):
    teacher_version = "score_dp3_rules_v30"
    depth = 3
    bonus = 0.
    improvements = True
    response_search = False

    def _should_piao(self, game, seat):
        # 可摸张数是全桌剩余量，必须留够其余三家后本家的一次摸牌。
        return game.drawable >= len(game.players) and super()._should_piao(game, seat)

    def _baotou_switch(self, game, seat, discards):
        if game.drawable < len(game.players):
            return None
        return super()._baotou_switch(game, seat, discards)

    def evaluate(self, game, seat, counts, melds, depth, chain=0, piao=0):
        remain = [4-c for c in BASE.seen_counts(game,seat)]
        return value(counts,remain,melds,depth,chain,piao,
                     24. if seat==game.dealer else 10.,self.bonus,
                     ycbk=game.you_cai_bi_kao, improvements=self.improvements)

    def utility(self, estimate):
        return estimate.score+self.bonus*estimate.win

    def _choose_discard(self, game, seat, discards):
        default = super()._choose_discard(game,seat,discards)
        if len(discards)==1:
            return default
        p=game.players[seat]
        horizon=min(self.depth,game.drawable//4)
        best, best_value = default, -1.
        # 基线先参与，完全相等时保留基线；不预先排除打财神或向听劣化。
        for action in [default]+[a for a in discards if a!=default]:
            counts=list(p.counts);counts[action[1]]-=1
            chain,piao=discard_chain(p,counts,action[1],game.drawn_tile)
            estimate=self.evaluate(game,seat,counts,p.meld_count,horizon,chain,piao)
            score=self.utility(estimate)
            if score>best_value+1e-10:
                best,best_value=action,score
        return best

