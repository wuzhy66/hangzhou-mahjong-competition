"""仅供 ONNX 打榜推理使用的弃牌后处理，不改观测或训练策略。"""
import numpy as np


def discard_mask(counts, legal_mask, seen_counts):
    """字牌集合加上剩余同种牌更少、没有同花色 ±1 邻牌的单张数牌。"""
    hand = np.asarray(counts)
    seen = np.asarray(seen_counts)
    original = np.asarray(legal_mask, dtype=np.float32)
    if hand.shape != (34,) or seen.shape != (34,) or original.shape != (74,):
        raise ValueError('需要 34 维手牌、34 维可见张数和 74 维 legal mask')
    if not (np.all(np.isfinite(hand)) and np.all(np.isfinite(seen))
            and np.all(hand == np.rint(hand)) and np.all(seen == np.rint(seen))
            and np.all((0 <= hand) & (hand <= seen) & (seen <= 4))
            and np.all((original == 0) | (original == 1))):
        raise ValueError('手牌、去重可见计数或 legal mask 不合法')
    remaining = 4 - seen.astype(int)
    honors = [t for t in range(27, 33) if hand[t] == 1 and original[t] == 1]
    result = original.copy()
    info = dict(honors=honors, added_suited=[], threshold=None, remaining=remaining.tolist())
    if not honors:
        return result, info
    threshold = min(int(remaining[t]) for t in honors)
    added = []
    for t in range(27):
        if hand[t] != 1 or original[t] != 1 or remaining[t] >= threshold:
            continue
        rank = t % 9
        if rank > 0 and hand[t - 1] > 0:
            continue
        if rank < 8 and hand[t + 1] > 0:
            continue
        added.append(t)
    result[:34] = 0
    result[honors + added] = 1
    info.update(added_suited=added, threshold=threshold)
    return result, info


def postprocess_honor_discard(action_id, logits, observation, *, history_available=True):
    """保留特殊决策；仅从原始 logits 重选普通弃牌，不再次调用模型。"""
    obs = np.asarray(observation)
    scores = np.asarray(logits)
    aid = int(action_id)
    if obs.shape != (6122,) or scores.shape != (74,):
        raise ValueError('后处理需要 observation v2（6122 维）和 74 维 logits')
    legal = obs[-74:]
    if not 0 <= aid < 74 or legal[aid] != 1:
        raise ValueError('原模型输出了不合法动作')
    info = dict(name='honor_remaining', original_action=aid, final_action=aid, changed=False)
    reason = ('white_discard' if aid == 33 else
              'non_discard_action' if aid >= 34 else
              'declined_hu' if legal[72] == 1 else
              'public_history_unavailable' if not history_available else None)
    if reason:
        info['reason'] = reason
        return aid, info
    counts = obs[:4 * 34].reshape(4, 34).sum(axis=0)
    seen = obs[14 * 34:15 * 34] * 4
    mask, details = discard_mask(counts, legal, seen)
    info.update(details)
    if not details['honors']:
        info['reason'] = 'no_single_honor'
    elif mask[aid] == 1:
        info['reason'] = 'already_allowed'
    else:
        allowed = np.flatnonzero(mask[:34])
        if not np.isfinite(scores[allowed]).all():
            raise ValueError('候选弃牌 logits 不是有限值')
        aid = int(allowed[np.argmax(scores[allowed])])
        info.update(final_action=aid, changed=True, reason='reselected_discard')
    return aid, info


DEFAULT_METHOD = 'shape_lookahead'
TOP_K = 4
TOP_P = 0.9


def shortlist(scores, legal, k=TOP_K, p=TOP_P):
    """达到 Top-p 的最短前缀，再以 Top-k 封顶；同分沿用动作编号。"""
    ids = np.flatnonzero(np.asarray(legal)[:33])
    if not len(ids):
        return [], []
    values = np.asarray(scores, dtype=float)[ids]
    if not np.isfinite(values).all():
        raise ValueError('候选弃牌 logits 不是有限值')
    order = np.lexsort((ids, -values))
    ids, values = ids[order], values[order]
    probs = np.exp(values-values.max())
    probs /= probs.sum()
    n = min(len(ids), int(np.searchsorted(np.cumsum(probs), p)) + 1, k)
    return ids[:n].tolist(), [dict(action=int(a), prob=float(v)) for a, v in zip(ids, probs)]


def postprocess_discard(action_id, logits, observation, *, view=None, seat=None,
                        history_available=True):
    """默认方法：一层进张搜索；仅后处理原模型的普通弃牌，不修改网络输入。"""
    obs = np.asarray(observation)
    scores = np.asarray(logits)
    aid = int(action_id)
    if obs.shape != (6122,) or scores.shape != (74,):
        raise ValueError('后处理需要 observation v2（6122 维）和 74 维 logits')
    legal = obs[-74:]
    if not 0 <= aid < 74 or legal[aid] != 1:
        raise ValueError('原模型输出了不合法动作')
    info = dict(name=DEFAULT_METHOD, original_action=aid, final_action=aid, changed=False,
                top_k=TOP_K, top_p=TOP_P)
    reason = ('white_discard' if aid == 33 else
              'non_discard_action' if aid >= 34 else
              'declined_hu' if legal[72] == 1 else
              'public_history_unavailable' if not history_available else
              'state_unavailable' if view is None or seat not in range(4) else None)
    if reason is None:
        hero = view.players[seat]
        catch = max(getattr(view, 'catch_origin', -1),
                    getattr(view, 'postprocess_catch_origin', -1))
        reason = ('gang_available' if np.any(legal[34:68]) else
                  'white_or_own_chain' if hero.counts[33] or hero.baotou or hero.chain_count or hero.piao else
                  'special_phase' if view.phase != 'draw' or view.turn != seat or view.drawn_tile is None
                  or getattr(view, 'gang_supplement', False) or catch >= 0
                  or getattr(view, 'catch_play', False) else
                  'wall_end' if view.drawable < 4 else None)
    if reason:
        info['reason'] = reason
        return aid, info
    choices, probabilities = shortlist(scores, legal)
    # 客户端使用原始 argmax；若调用方显式给了另一合法动作，则保留它。
    if not choices or choices[0] != aid:
        info['reason'] = 'non_argmax_action'
        return aid, info
    info.update(choices=choices, probabilities=probabilities)
    if len(choices) == 1:
        info['reason'] = 'single_candidate'
        return aid, info
    counts = obs[:136].reshape(4, 34).sum(axis=0)
    seen = obs[14*34:15*34] * 4
    if (not np.isfinite(counts).all() or not np.isfinite(seen).all()
            or not np.array_equal(counts, np.rint(counts)) or not np.array_equal(seen, np.rint(seen))
            or not np.all((0 <= counts) & (counts <= seen) & (seen <= 4))
            or not np.array_equal(counts, hero.counts)):
        raise ValueError('后处理自手与去重公开计数不一致')
    counts = counts.astype(int).tolist()
    remaining = (4-seen.astype(int)).tolist()
    depth = min(2, view.drawable//4)
    unit = 24. if seat == view.dealer else 10.
    from .discard_search import evaluate
    result = evaluate((counts, remaining, hero.meld_count, choices, depth, unit, view.you_cai_bi_kao))
    aid = result['final_action']
    if aid not in choices or legal[aid] != 1:
        raise ValueError('弃牌搜索返回了候选以外的动作')
    info.update(result, reason=result['status'], counts=counts, remaining=remaining,
                melds=hero.meld_count, depth=depth, unit=unit, ycbk=view.you_cai_bi_kao)
    return aid, info


def warmup_postprocess():
    """仅显式开启时预热，不让首次编译占用出牌窗口。"""
    from . import decision  # 安装与打榜观测相同的规则适配及 submodule 路径。
    from .discard_search import warmup
    warmup()
