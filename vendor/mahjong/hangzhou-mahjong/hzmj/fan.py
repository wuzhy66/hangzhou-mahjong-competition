"""番型倍率与计分（杭州麻将）。

平台口径（接入指南 §1.3 / §1.4）：

    总番 = 1 × 分支因子 × 2^动作链次数 × (4 白板 ×2) × (爆头 ×2)
    计算顺序：分支 -> 动作链 -> 4 白板 -> 爆头（最后统一加）

    分支因子：平胡 ×1 / 七对 ×2 / 豪华七对 ×4 / 双豪华 ×8 / 三豪华 ×16
    动作链：每个飘或杠动作 ×2（打出非飘非杠的牌即断链清零）
    4 白板：胡牌时手牌留存 + 链内飘出的白板 = 恰好 4 张 -> ×2
    爆头：听牌态摸任意牌即胡 -> ×2

    ⚠️ 指南字面把「4 白板」与「爆头」写成互斥（爆头栏注「正好 4 白板除外」），
    实测 `fan-calc` 二者**叠加**：`白白白白 1w..9w + 5b` -> fan=4
    （平胡 ×1 × 4白板 ×2 × 爆头 ×2）。×512 上限也只有叠加才成立。

    得分 = 底分 × 倍率 ×（庄家 ×8 / 闲家 ×1），三家分别结算
"""

from .hu import missing, qidui_missing, standard_missing
from .tiles import GOD_IDX

QIDUI_FACTORS = {0: 2, 1: 4, 2: 8, 3: 16}
QIDUI_LABELS = {0: "七对子", 1: "豪华七对", 2: "双豪华七对", 3: "三豪华七对"}
MAX_CHAIN = 6


def qidui_quads(counts):
    """七对牌型里的「组四张」数（豪华组）。

    平台口径（`fan-calc` 实测）：
      * 真四张 = 4 张**物理相同的真牌**，算 1 组
      * 4 张真白板也算 1 组 —— **但仅当它们在实际分解里作为 2 个对子留存**
      * 财神补出来的四张（如 3 张 1w + 1 财神）**不算**

    所以不能直接数 `counts[t] == 4`。财神要先去补落单牌（七对分解的唯一
    合法走法），补完剩下的白板才可能凑成那 1 组四张：

        白白白白 + 1w1w 2w2w 3w3w 4w4w 5w5w  -> 无落单，4 白板留存 -> 豪华×1
        白白白白 + 1w1w 2w2w 3w3w 4w4w 5w 6w -> 2 白补单，只剩 2 张 -> 七对（非豪华）
        白白白白 + 1w1w1w 2w2w2w 3w3w3w 9w   -> 4 白全去补单           -> 七对（非豪华）
        白白白白 + 1w1w1w1w 2w2w 3w3w 4w4w   -> 无落单                 -> 豪华×2

    曾直接数 `c == 4`，第 2、3 例多算一组豪华（×2 番）。
    """
    god = counts[GOD_IDX]
    singles = sum(c % 2 for i, c in enumerate(counts) if i != GOD_IDX)
    gods_left = god - min(god, singles)
    real = sum(1 for i, c in enumerate(counts) if i != GOD_IDX and c == 4)
    return min(3, real + (1 if gods_left == 4 else 0))


def branch_factor(counts, meld_count=0):
    """返回 (分支因子, 分支名)。同时满足多种牌型时取倍率最大者。"""
    best = (1, "平胡") if standard_missing(counts, meld_count) == 0 else None
    if qidui_missing(counts, meld_count) == 0:
        quads = qidui_quads(counts)
        cand = (QIDUI_FACTORS[quads], QIDUI_LABELS[quads])
        if best is None or cand[0] > best[0]:
            best = cand
    return best if best else (0, "未胡")


def god_count_total(counts, piao=0):
    """胡牌时的白板计数 = 手牌留存 + 链内飘出。"""
    return counts[GOD_IDX] + piao


def compute_fan(counts, meld_count=0, chain_count=0, piao=0, baotou=False):
    """计算总番。counts 为胡牌时的完整手牌（含刚摸的那张）。

    chain_count：本人动作链次数（飘 + 杠累计，断链清零）
    piao：链内飘出的白板张数（piao <= chain_count）
    baotou：爆头态

    返回 dict：{hu, fan, branch, detail, gods, baotou}
    """
    factor, branch = branch_factor(counts, meld_count)
    if factor == 0:
        return {"hu": False, "fan": 0, "branch": "未胡", "detail": [],
                "gods": god_count_total(counts, piao), "baotou": False}

    chain_count = max(0, min(MAX_CHAIN, int(chain_count)))
    piao = max(0, min(chain_count, int(piao)))

    detail = [branch]
    fan = factor
    if chain_count:
        fan *= 2 ** chain_count
        detail.append("动作链×%d" % chain_count)

    gods = god_count_total(counts, piao)
    if gods == 4:
        # 「4 个白板」= 手留 + 链内飘出恰好 4 张，与爆头因子可叠加
        # （×512 上限 = 三豪华七对 16 × 三财飘 8 × 4 白板 2 × 爆头 2）
        fan *= 2
        detail.append("4个白板")
    if baotou:
        fan *= 2
        detail.append("爆头")

    return {"hu": True, "fan": fan, "branch": branch, "detail": detail,
            "gods": gods, "baotou": baotou}


def settle(fan, base=1, winner_is_dealer=False, n_players=4):
    """结算：返回 (胡家净得, [其余三家各自净失])，顺序与传入无关。

    庄家胡：三家闲家各付 base × fan × 8
    闲家胡：庄家付 base × fan × 8，另两闲家各付 base × fan × 1
    """
    unit = base * fan
    if winner_is_dealer:
        losses = [unit * 8] * (n_players - 1)
    else:
        losses = [unit * 8] + [unit] * (n_players - 2)
    return sum(losses), losses


def score_table(counts, meld_count=0, chain_count=0, piao=0, baotou=False,
                base=1):
    """与平台 POST /portal/api/tools/fan-calc 同构的结果，便于对拍。"""
    res = compute_fan(counts, meld_count, chain_count, piao, baotou)
    if not res["hu"]:
        return {"hu": False, "baotou": False, "fan": 0, "detail": [],
                "scores": {}}
    dealer_win, dealer_lose = settle(res["fan"], base, True)
    other_win, other_lose = settle(res["fan"], base, False)
    return {
        "hu": True,
        "baotou": res["baotou"],
        "fan": res["fan"],
        "detail": res["detail"],
        "scores": {
            "dealer_hu": {"win": dealer_win, "lose": dealer_lose},
            "nondealer_hu": {"win": other_win, "lose": other_lose},
        },
    }


def is_hu_shape(counts, meld_count=0):
    return missing(counts, meld_count) == 0
