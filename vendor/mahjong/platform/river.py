"""兼容完整历史牌河与移除吃碰明杠来源牌后的牌河，保持模型观测语义。"""
from hzmj.tiles import tile_code, tile_index


def public_events_from_raw(raw):
    """将已按快照水位截取的原始事件转换为观测使用的公开事件。"""
    result = []
    for event in raw:
        kind = event['type']
        if kind not in ('tile_discarded', 'chi', 'peng', 'gang'):
            continue
        tile = tile_index(event['tile'])
        if tile is None:
            raise ValueError('公开事件牌码无效')
        data = dict(event.get('data') or {}, seat=event['seat'], tile=tile)
        if kind == 'gang':
            data['mode'] = data.get('mode', data.get('kind'))
        result.append(dict(kind=kind, data=data))
    return result


def canonical_snapshot(snapshot, public_events):
    """严格对账后返回完整历史牌河；不修改平台快照，不凭副露数量猜缺牌。"""
    if public_events is None:
        return snapshot
    full, remaining = [[] for _ in range(4)], [[] for _ in range(4)]
    pending = None
    for event in public_events:
        kind, data = event['kind'], event['data']
        actor, tile = data['seat'], data['tile']
        if actor not in range(4) or tile not in range(34):
            raise ValueError('公开事件座位或牌码无效')
        if kind == 'tile_discarded':
            full[actor].append(tile)
            remaining[actor].append(tile)
            pending = (actor, tile)
        elif kind in ('chi', 'peng') or kind == 'gang' and data.get('mode', data.get('kind')) == 'ming':
            if (pending is None or pending[0] == actor or pending[1] != tile
                    or kind == 'chi' and actor != (pending[0] + 1) % 4):
                raise ValueError('吃碰明杠缺少一致的来源弃牌')
            source = pending[0]
            if not remaining[source] or remaining[source][-1] != tile:
                raise ValueError('吃碰明杠来源弃牌已被取走')
            remaining[source].pop()
            pending = None
        elif kind == 'gang':
            if data.get('mode', data.get('kind')) not in ('an', 'bu'):
                raise ValueError('公开杠事件类型无效')
            # 暗杠、补杠不再从牌河取牌，也结束此前弃牌的响应阶段。
            pending = None
    actual = [[tile_index(tile) for tile in river] for river in snapshot['discards']]
    if actual != full and actual != remaining:
        raise ValueError('公开历史与快照牌河不一致')
    if actual == full:
        return snapshot
    return dict(snapshot, discards=[[tile_code(tile) for tile in river] for river in full])
