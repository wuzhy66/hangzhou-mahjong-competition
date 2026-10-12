"""兼容客户端旧导入名；实现完全来自原样提取的 WSL modules。"""
import inference_runtime
from hzmj.adapter import *
from hzmj.tiles import *
from hzmj.hu import *
from hzmj.bots import seen_counts
from mjenv.spaces import *
from mjenv.encode import encode as encode_base
from mahjong.observation import *
from mahjong.engine_compat.piao import claim_piao
from mahjong.platform.river import canonical_snapshot, public_events_from_raw
from mahjong.platform.decision import observation, decide


_legacy_decide = decide

def decide(policy, snapshot, you_cai_bi_kao=False, public_events=None, *,
           with_observation=False, honor_discard_postprocess=False, postprocess_info=None):
    if getattr(policy, 'strategy', None) != 'historical297':
        return _legacy_decide(policy, snapshot, you_cai_bi_kao, public_events,
            with_observation=with_observation, honor_discard_postprocess=honor_discard_postprocess,
            postprocess_info=postprocess_info)
    if honor_discard_postprocess:
        raise ValueError('historical297 不允许添加旧版弃牌后处理')
    # 复用已对账的公开历史和 GameView，独立编码7113维，保留真实模型输入。
    _, view, actions = observation(snapshot, you_cai_bi_kao, public_events)
    if not actions:
        return (None, None) if with_observation else None
    from historical297_source.observation import encode_observation as encode_public
    from historical297_source.action_scheme import action_mask, from_id
    obs = encode_public(view, view.seat, action_mask(actions), public_events=public_events,
                        you_cai_bi_kao=you_cai_bi_kao, legal_actions=actions)
    action = from_id(int(policy.predict(obs)[0]), actions)
    if action is None:
        raise ValueError('historical297 输出非法动作')
    payload = action_payload(view, action)
    return (payload, obs) if with_observation else payload
