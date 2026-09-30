"""在新房匹配前错开本机账号；只读取对端状态，不影响在途 Game。"""
import json
import os
from pathlib import Path
import time


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        # Windows 的 os.kill(pid, 0) 不能作为无副作用的存活检查。
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _peer_room(peer, previous_room, *, now, max_age):
    try:
        model = _read(peer / 'active_model.json')
        if model.get('state') != 'running' or not _pid_alive(model.get('pid')):
            return None, 'peer_not_running'
        status_path = peer / 'status.json'
        age = now - status_path.stat().st_mtime
        if age < -5 or age > max_age:
            return None, 'peer_status_stale'
        status = _read(status_path)
        room = status.get('tournament_id')
        games = status.get('my_games')
        if status.get('status') != 'running':
            return None, 'peer_waiting_for_room'
        if (not isinstance(room, str) or not room.startswith('a_')
                or not isinstance(games, list) or not games
                or any(not isinstance(gid, str) or not gid.startswith(room + '_r') for gid in games)):
            return None, 'peer_status_incomplete'
        if room == previous_room:
            return None, 'peer_still_in_previous_room'
        return room, 'peer_room_running'
    except (OSError, ValueError, TypeError, AttributeError):
        return None, 'peer_status_unavailable'


def wait_for_peer_room(client, stop, *, clock=time.monotonic, wall_clock=time.time):
    """可选的 follower 门禁；每次匹配读取配置，Astra 无配置时保持原流程。"""
    output = Path(client.output)
    control = output / 'match-coordination.json'
    try:
        config = _read(control)
    except FileNotFoundError:
        return not stop.is_set()
    if config.get('enabled') is False:
        return not stop.is_set()
    if config.get('version') != 1 or config.get('enabled') is not True:
        raise ValueError('match-coordination 需要 version=1、enabled=true/false')
    peer_name = config.get('peer_output')
    if not isinstance(peer_name, str) or not peer_name:
        raise ValueError('match-coordination 缺少 peer_output')
    peer = Path(peer_name)
    if not peer.is_absolute():
        peer = output.parent / peer
    peer = peer.resolve()
    if peer == output.resolve():
        raise ValueError('match-coordination 不能等待自身')
    limits = {'min_delay_seconds': (15, 0, 60), 'settle_seconds': (5, 1, 30),
              'peer_stale_seconds': (10, 3, 60), 'offline_timeout_seconds': (120, 30, 600)}
    settings = {}
    for key, (default, low, high) in limits.items():
        value = config.get(key, default)
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f'match-coordination {key} 超出允许范围')
        settings[key] = value
    try:
        previous_room = _read(output / 'status.json').get('tournament_id')
    except (OSError, ValueError, AttributeError):
        previous_room = None
    started = clock()
    stable_room, stable_since = None, None
    unavailable_since = None
    last_reason, last_log = None, -float('inf')
    unavailable = {'peer_not_running', 'peer_status_stale', 'peer_status_unavailable'}
    while not stop.is_set():
        room, reason = _peer_room(peer, previous_room, now=wall_clock(),
                                  max_age=settings['peer_stale_seconds'])
        elapsed = clock() - started
        if reason in unavailable:
            if unavailable_since is None:
                unavailable_since = clock()
        else:
            unavailable_since = None
        if room != stable_room:
            stable_room, stable_since = room, clock() if room else None
        if (room and elapsed >= settings['min_delay_seconds']
                and clock() - stable_since >= settings['settle_seconds']):
            client.record('ranking_match_released', peer_room=room,
                          previous_room=previous_room, waited_seconds=round(elapsed, 3))
            return not stop.is_set()
        # 对端离线/状态不可读持续一段时间后允许独立运行；活着且等待新房时保留 Astra 优先。
        if (unavailable_since is not None
                and clock() - unavailable_since >= settings['offline_timeout_seconds']):
            client.record('ranking_match_peer_unavailable', reason=reason,
                          previous_room=previous_room, waited_seconds=round(elapsed, 3))
            return not stop.is_set()
        if reason != last_reason or clock() - last_log >= 30:
            client.record('ranking_match_wait', reason=reason, peer_room=room,
                          previous_room=previous_room, waited_seconds=round(elapsed, 3))
            last_reason, last_log = reason, clock()
        if stop.wait(.5):
            return False
    return False
