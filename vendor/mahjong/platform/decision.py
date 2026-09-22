"""平台快照到 BC 观测和动作的适配。"""
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1] / 'hangzhou-mahjong'
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from mahjong.engine_compat import install
install()
from hzmj import adapter
from .river import canonical_snapshot

from mjenv import spaces, fastrule
from mahjong.observation import encode_observation

fastrule.patch()


def observation(snapshot, you_cai_bi_kao=False, public_events=None):
    snapshot = canonical_snapshot(snapshot, public_events)
    view, seat = adapter.from_snapshot(snapshot, you_cai_bi_kao)
    if not any(view.players[s].discards for s in range(4)):
        view.last_discard = None  # 平台开局可能返回 0w 或上一局最后弃牌。
    # 上游 adapter 尚未读取 v27 文档中的豁免座位；只在外层补齐。
    god = snapshot.get('god') or {}
    view.god_discarder_seat = god.get('god_discarder_seat', -1)
    if view.god_discarder_seat == seat:
        view.catch_play = False
    if public_events is not None:
        # 从当前局公开动作恢复快照省略的副露来源；绝不使用未来事件。
        meld_sources = [[] for _ in range(4)]
        last_discarder = -1
        hero_chain = []
        catch_origin = -1
        last_hero_action = None
        for event in public_events:
            kind, data = event['kind'], event['data']
            actor = data['seat']
            mode = data.get('mode', data.get('kind'))
            if kind == 'tile_discarded':
                last_discarder = actor
                if data['tile'] == 33:
                    catch_origin = actor
                elif actor == catch_origin:
                    catch_origin = -1
            elif kind in ('chi', 'peng') or kind == 'gang' and mode in ('an', 'ming'):
                meld_sources[actor].append(actor if mode == 'an' else last_discarder)
            if actor == seat:
                if kind in ('chi', 'peng', 'gang', 'tile_discarded'):
                    last_hero_action = kind
                if (kind in ('chi', 'peng') and actor != catch_origin
                        or kind == 'tile_discarded' and data['tile'] != 33):
                    hero_chain.clear()
                elif kind == 'gang':
                    hero_chain.append('gang')
                elif kind == 'tile_discarded' and data['tile'] == 33:
                    hero_chain.append('piao')
        for actor, player in enumerate(view.players):
            if len(meld_sources[actor]) != len(player.melds):
                raise ValueError('完整公开历史与副露数量不一致')
            for meld, origin in zip(player.melds, meld_sources[actor]):
                meld.from_seat = origin
        # 当前快照的显式 -1 表示圈已结束，不能被上圈公开弃牌历史重新激活。
        # 独立保存后处理状态，不改变 encode 对 GameView.catch_play 的既有语义。
        view.postprocess_catch_origin = god.get('god_discarder_seat', catch_origin)
        # chain 同时含杠与飘；上游缺省 piao=chain 会把杠开错误编码为财飘。
        chain = view.players[seat].chain_count
        view.players[seat].piao = hero_chain[-chain:].count('piao') if chain else 0
        # 完整公开历史中，本家上一动作是杠且此时刚摸牌，才是杠补。
        # 不能用chain_count>0代替：财飘摸牌不享有杠开豁免。
        view.gang_supplement = (last_hero_action == 'gang' and chain > 0
                                and view.phase == 'draw' and view.turn == seat
                                and view.drawn_tile is not None)
        if ('god_discarder_seat' not in god and view.phase == 'draw'
                and view.turn == catch_origin and view.drawn_tile is not None
                and not view.gang_supplement):
            # 旧快照只能从历史还原；自然摸牌回到发起者时已走完一圈。
            # 杠补不推进轮次，不能据此解除仍在进行中的抓打圈。
            view.postprocess_catch_origin = -1
    actions = adapter.legal_actions(view, seat)
    if not actions:
        return None, view, actions
    obs = encode_observation(view, seat, spaces.action_mask(actions),
                             you_cai_bi_kao=you_cai_bi_kao, public_events=public_events)
    return obs, view, actions


def decide(policy, snapshot, you_cai_bi_kao=False, public_events=None, *, with_observation=False,
           honor_discard_postprocess=False, postprocess_info=None):
    obs, view, actions = observation(snapshot, you_cai_bi_kao, public_events)
    if obs is None:
        return (None, None) if with_observation else None
    if len(actions) == 1:
        aid = spaces.to_id(actions[0], actions)
    elif honor_discard_postprocess:
        from .postprocess import postprocess_discard
        # 与 ONNX actions 一样取原始 logits 的 argmax，仅执行一次网络推理。
        logits = policy.scores(obs)[0]
        aid, info = postprocess_discard(int(logits.argmax()), logits, obs,
                                       view=view, seat=view.seat,
                                       history_available=public_events is not None)
        if postprocess_info is not None:
            postprocess_info.update(info)
    else:
        aid = int(policy.predict(obs)[0])
    action = spaces.from_id(aid, actions)
    if action is None:
        raise ValueError('ONNX 输出了不在 action mask 中的动作')
    payload = adapter.action_payload(view, action)
    return (payload, obs) if with_observation else payload
