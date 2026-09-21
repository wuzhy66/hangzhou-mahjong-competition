"""官方比赛和打榜客户端：HTTP/SSE、限流、状态恢复和多桌调度。"""
import os
import uuid
import json
import time
from pathlib import Path
import observation as adapter
from observation import (observation, decide, canonical_snapshot, public_events_from_raw,
                         ALL_TILES, tile_index, is_baotou, claim_piao)


# 原子状态保存

def write_json(path, value):
    """同目录临时文件 + rename，读者不会看到只写了一半的请求。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        # Windows 读者/文件系统过滤器可能短暂阻止替换；保留原子写入并有界重试。
        replace_deadline = time.perf_counter() + .25
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                if (os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33)
                        or time.perf_counter() >= replace_deadline):
                    raise
                time.sleep(.005)
    finally:
        temporary.unlink(missing_ok=True)


# 时钟

import time
import uuid


# 防止恢复旧日志/状态时把另一进程的 monotonic 时间当成本进程快照。
CLOCK_ID = uuid.uuid4().hex


def response_age(response, *, default=0):
    """读取快照的实际等待时间；旧响应保留 wall-clock 回退语义。"""
    meta = response.get('_client') or {}
    if meta.get('clock_id') == CLOCK_ID and meta.get('received_monotonic') is not None:
        return time.monotonic() - meta['received_monotonic']
    if meta.get('received_at') is not None:
        return time.time() - meta['received_at']
    return default


# 请求调度

import heapq
import itertools
import threading
import time


class RequestGate:
    def __init__(self, rate=14):
        # Python 3.12/Windows monotonic为15.625ms粗时钟；高精度单调计时保持原发送上限。
        # 保留精确计时，最高速率收回至粗时钟版本实测容量，避免未验证的提速。
        self.max_rate = 12.8
        self.interval = 1 / min(rate, self.max_rate)
        self.recover_at = time.perf_counter() + 30
        self.next_at = 0.0
        self.condition = threading.Condition()
        self.waiters = []
        self.counter = itertools.count()
        # 首跳 SSE 增量及普通读取均可能携带动作窗口；等待350ms后禁止第二跳继续插队。
        self.max_priority_wait = .35
        self.last_decision_refresh = False

    def send(self, function, *, urgent=False, decision_refresh=False):
        # 已确认存在决策机会的后续快照优先完成；最多连续一项，保留其他桌进度。
        ticket = (0 if decision_refresh else 1 if urgent else 2,
                  next(self.counter), time.perf_counter())
        with self.condition:
            heapq.heappush(self.waiters, ticket)
            try:
                # 所有类型等待过久均按入队顺序晋升；否则持续首跳流量会饿死第二跳。
                # 晋升不跳过 next_at，因此共享速率和 429 退避仍约束实际发送。
                while True:
                    now = time.perf_counter()
                    aged = [item for item in self.waiters
                            if now - item[2] >= self.max_priority_wait]
                    others = [item for item in self.waiters if item[0] != 0]
                    if aged:
                        head = min(aged, key=lambda item: item[1])
                    elif self.last_decision_refresh and others:
                        head = min(others)
                    else:
                        head = self.waiters[0]
                    if head == ticket and now >= self.next_at:
                        break
                    # 没有入队/出队时，也须在普通请求晋升后重新选择队首。
                    self.condition.wait(timeout=max(.001, min(.05, self.next_at - now))
                                        if head == ticket else .05)
                self.last_decision_refresh = bool(decision_refresh)
                return function()
            finally:
                self.waiters.remove(ticket)
                heapq.heapify(self.waiters)
                self.next_at = time.perf_counter() + self.interval
                self.condition.notify_all()

    def backoff(self):
        with self.condition:
            self.next_at = max(self.next_at, time.perf_counter() + .2)
            self.interval = min(1 / 6, self.interval * 1.05)
            self.recover_at = time.perf_counter() + 30
            self.condition.notify_all()

    def success(self):
        """无 429 至少 30 秒后，每 20 秒缓慢恢复 5%，始终低于文档的 16/s。"""
        with self.condition:
            now = time.perf_counter()
            if now < self.recover_at:
                return False
            self.recover_at = now + 20
            previous = self.interval
            self.interval = max(1 / self.max_rate, self.interval / 1.05)
            return self.interval < previous


# HTTP

import http.client
from collections import deque
import json
from pathlib import Path
import ssl
import threading
import time
import urllib.parse


"""更高SSE水位到达时，有界备用增量读取；旧响应不进入GameState。"""
from collections import deque
from types import SimpleNamespace
import threading
import time


class StateReadCancelled(Exception):
    """另一读取已成功，尚未发送的同游标 GET 不再发送。"""


def read_state_delta(client, gid, cursor, notifications, *, urgent=False):
    path = f'/api/games/{gid}/state?seq={cursor}'
    owner = client.connections
    pending = getattr(owner, 'delta_pending', None)
    if cursor <= 0 or (pending is not None and not pending.done.is_set()):
        return client.api(path, urgent=urgent)
    owner.delta_pending = None
    group = SimpleNamespace(lock=threading.Lock(), done=threading.Event(),
                            selected=threading.Event(), calls=[], winner=None,
                            fatal=None, slot=False, hedged=False)
    connection = getattr(owner, 'connection', None)
    owner.connection = None

    def launch(connection, backup):
        call = SimpleNamespace(done=False, completed_at=None, result=None, error=None, connection=None,
                               reads=[], dispatched=None, watermark=None, backup=backup)
        group.calls.append(call)

        def dispatched():
            with group.lock:
                call.dispatched = time.perf_counter()
                call.watermark = notifications.seq

        def run():
            client.connections.connection = connection
            client.connections.state_reads = deque(maxlen=32)
            try:
                call.result = client.api(path, urgent=True if backup else urgent,
                                         _state_dispatch=dispatched,
                                         _state_cancel=group.selected.is_set)
            except BaseException as error:
                call.error = error
            finally:
                conn = getattr(client.connections, 'connection', None)
                call.reads = list(getattr(client.connections, 'state_reads', ()))
                with group.lock:
                    call.completed_at = time.perf_counter()
                    call.done = True
                    if not group.selected.is_set():
                        if call.error is None:
                            group.winner = call
                            group.selected.set()
                        elif getattr(call.error, 'status', None) in (401, 403):
                            group.fatal = call.error
                            group.selected.set()
                    accepted = group.winner is call
                    if accepted:
                        call.connection = conn
                    if all(c.done for c in group.calls):
                        if group.slot:
                            client._delta_slots.release()
                            group.slot = False
                        group.done.set()
                if not accepted and conn is not None:
                    conn.close()
                with notifications.condition:
                    notifications.condition.notify_all()
                if group.hedged:
                    cancelled = isinstance(call.error, StateReadCancelled)
                    client.record('state_read_selected' if accepted else
                                  'state_read_cancelled' if cancelled else 'state_read_discarded',
                                  game_id=gid, cursor=cursor, source='backup' if backup else 'primary',
                                  status=getattr(call.error, 'status', 200 if call.error is None else None),
                                  error_type=type(call.error).__name__ if call.error else None,
                                  request_failure=getattr(call.error, 'request_failure', None),
                                  state_reads=call.reads)

        worker = threading.Thread(target=run, name='state-delta-backup' if backup else 'state-delta-read',
                                  daemon=True)
        try:
            worker.start()
        except BaseException:
            group.calls.remove(call)
            raise
        return call

    # 每桌最多一个未结束的额外 GET，跨桌最多16个；连同16桌主读取最多32个。
    with group.lock:
        try:
            primary = launch(connection, False)
        except BaseException:
            owner.connection = connection
            raise
    noticed = None
    while True:
        with notifications.condition:
            with group.lock:
                if group.winner is not None or group.fatal is not None:
                    owner.delta_pending = group
                    if group.fatal is not None:
                        raise group.fatal
                    winner = group.winner
                    owner.connection = winner.connection
                    if not hasattr(owner, 'state_reads'):
                        owner.state_reads = deque(maxlen=32)
                    owner.state_reads.extend(winner.reads)
                    return winner.result
                if group.done.is_set():
                    raise next(c.error for c in group.calls if c.error is not None)
                now = time.perf_counter()
                failed = [c for c in group.calls if c.done and c.error is not None]
                if failed and now - min(c.completed_at for c in failed) >= .05:
                    # 一边报错后最多短等50ms；不能为等另一个成功结果挂到35秒。
                    # 原有429退避/增量重试仍生效，剩余读取只清理连接，不修改状态。
                    group.selected.set()
                    owner.delta_pending = group
                    raise failed[0].error
                if (not group.hedged and primary.dispatched is not None
                        and notifications.seq > primary.watermark):
                    if noticed is None:
                        noticed = now
                    if now - noticed >= .25 and now - primary.dispatched >= .35:
                        if client._delta_slots.acquire(blocking=False):
                            group.slot = True
                            group.hedged = True
                            try:
                                launch(None, True)
                            except BaseException:
                                client._delta_slots.release()
                                group.slot = False
                                # 主读取仍在运行，备用线程创建失败不能丢掉主结果。
                                client.record('state_read_backup_unavailable', game_id=gid, cursor=cursor)
                            else:
                                client.record('state_read_hedged', game_id=gid, cursor=cursor,
                                              notified_seq=notifications.seq, retry_limit=1,
                                              waiting_ms=(now-primary.dispatched)*1000)
            notifications.condition.wait(timeout=.05)


class ApiError(RuntimeError):
    def __init__(self, status, endpoint=None, action=None, detail=None, timing=None):
        self.status = status
        self.endpoint = endpoint
        self.action = action
        self.detail = detail
        self.timing = timing
        super().__init__(f'平台 HTTP {status}')


class Client:
    def __init__(self, server, token, output, insecure=False, *, ca_file=None):
        self.server = server.rstrip('/')
        self.token = token
        self.context = ssl._create_unverified_context() if insecure else ssl.create_default_context(cafile=ca_file)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.budget_path = self.output / 'request_budget.json'
        budget = json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {}
        self.gate = RequestGate(rate=min(14, budget.get('state_rate', 14)))
        self.connections = threading.local()
        self._delta_slots = threading.BoundedSemaphore(16)

    def record(self, kind, **data):
        row = {'unix': time.time(), 'kind': kind, **data}
        with self.lock:
            with (self.output / 'events.jsonl').open('a') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
        if kind != 'action':
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def state_delta(self, gid, cursor, notifications, *, urgent=False):
        # 线上备用同游标 GET 未带来可测收益，反而触发额外 429；恢复单请求。
        # 保留共享 gate 和原游标重试，不能靠重复长轮询绕过平台等待。
        return self.api(f'/api/games/{gid}/state?seq={cursor}', urgent=urgent)

    def api(self, path, payload=None, *, urgent=False, decision_refresh=False, _state_dispatch=None, _state_cancel=None):
        """发送平台请求。

        ``urgent`` 适用于由新 SSE 水位触发的 state 读取。
        ``decision_refresh`` 用于增量确认决策机会后仍需获取权威快照的后续读取。
        两者均不提高请求总速率，允许该读取在同一身份的普通 state 队列前发送，避免
        在尚未得知窗口类型的第一跳就耗尽 1 秒吃碰或 3 秒弃牌窗口。
        """
        if decision_refresh and (payload is not None or '/state?' not in path or not path.endswith('seq=0')):
            raise ValueError('decision_refresh 仅用于已确认决策机会的权威快照读取')
        if (_state_dispatch is not None or _state_cancel is not None) and (payload is not None or '/state?' not in path):
            raise ValueError('读取派发回调仅用于state GET')
        connection = getattr(self.connections, 'connection', None)
        if connection is None:
            url = urllib.parse.urlsplit(self.server)
            connection = (http.client.HTTPSConnection(url.hostname, url.port, timeout=35, context=self.context)
                          if url.scheme == 'https' else http.client.HTTPConnection(url.hostname, url.port, timeout=35))
            self.connections.connection = connection
        body = None if payload is None else json.dumps(payload).encode()
        headers = {'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'}
        connection.timeout = 1  # 建连/发送不能阻塞其他桌的 state 调度。
        if connection.sock is not None:
            connection.sock.settimeout(connection.timeout)
        request_phase = 'prepare'
        try:
            requested_at = time.time()
            requested_monotonic = time.monotonic()
            ready_at = requested_at
            dispatched_at = requested_at
            sent_at = requested_at
            ready_monotonic = dispatched_monotonic = requested_monotonic
            if '/state?' in path:
                # 建连只属于本桌，不能持有全身份 gate 等待 DNS/TCP/TLS。
                # 实际 GET 仍在 gate 内发送，握手并行不会放宽 state 请求预算。
                if _state_cancel is not None and _state_cancel():
                    raise StateReadCancelled()
                if connection.sock is None:
                    request_phase = 'connect'
                    connection.connect()
                ready_at = time.time()
                ready_monotonic = time.monotonic()
                # 按实际发送时刻节流，连接复用避免每次 TLS 握手抖动后成批抵达。
                def send_state():
                    nonlocal request_phase
                    if _state_cancel is not None and _state_cancel():
                        raise StateReadCancelled()
                    request_phase = 'send'
                    dispatched_at = time.time()
                    dispatched_monotonic = time.monotonic()
                    connection.request('GET', path, headers=headers)
                    if _state_dispatch is not None:
                        _state_dispatch()
                    return dispatched_at, dispatched_monotonic
                request_phase = 'queue'
                dispatched_at, dispatched_monotonic = self.gate.send(
                    send_state, urgent=urgent or path.endswith('seq=0'),
                    decision_refresh=decision_refresh)
            else:
                request_phase = 'send'
                connection.request('GET' if payload is None else 'POST', path, body=body, headers=headers)
            sent_at = time.time()
            sent_monotonic = time.monotonic()
            if connection.sock is not None:
                # 匹配不受出牌窗口约束；保持非幂等请求不自动重发。
                connection.sock.settimeout(35 if path == "/api/match" or payload is None else 1)
            request_phase = 'response_headers'
            response = connection.getresponse()
            request_phase = 'response_body'
            content = response.read()
            received_at = time.time()
            received_monotonic = time.monotonic()
            meta = dict(requested_at=requested_at, received_at=received_at,
                        dispatched_at=dispatched_at,
                        connect_ms=(ready_at-requested_at)*1000,
                        queue_ms=(dispatched_at-ready_at)*1000,
                        send_ms=(sent_at-dispatched_at)*1000,
                        http_ms=(received_at-dispatched_at)*1000,
                        server_date=response.getheader('Date'),
                        urgent=bool(urgent or path.endswith('seq=0')),
                        decision_refresh=bool(decision_refresh))
            # 保留原 wall-clock 字段作对照，不把校时跳变当作 HTTP/队列处理耗时。
            meta.update(clock_id=CLOCK_ID, requested_monotonic=requested_monotonic,
                        dispatched_monotonic=dispatched_monotonic,
                        received_monotonic=received_monotonic,
                        connect_monotonic_ms=(ready_monotonic-requested_monotonic)*1000,
                        queue_monotonic_ms=(dispatched_monotonic-ready_monotonic)*1000,
                        send_monotonic_ms=(sent_monotonic-dispatched_monotonic)*1000,
                        http_monotonic_ms=(received_monotonic-dispatched_monotonic)*1000,
                        request_monotonic_ms=(received_monotonic-requested_monotonic)*1000,
                        clock_shift_ms=((received_at-requested_at)-
                                        (received_monotonic-requested_monotonic))*1000)
            if '/state?' in path:
                if not hasattr(self.connections, 'state_reads'):
                    self.connections.state_reads = deque(maxlen=32)
                self.connections.state_reads.append(dict(meta, status=response.status,
                    query_seq=int(urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)['seq'][0])))
            if response.status < 400:
                result = json.loads(content)
                if '/state?' in path:
                    result['_client'] = meta
                    if self.gate.success():
                        write_json(self.budget_path, {'state_rate': 1/self.gate.interval})
                        self.record('request_budget_changed', state_rate=1/self.gate.interval,
                                    reason='限流恢复期内成功读取，缓慢恢复预算')
                return result
            # 只保存服务端错误码和短消息，不记录 Authorization 或响应中的其他数据。
            try:
                body = json.loads(content[:8192])
                detail = {key: str(body[key]).replace(self.token, '[redacted]')[:300]
                          for key in ('code', 'error', 'message') if key in body}
            except (ValueError, TypeError):
                detail = {'message': 'non-json error'}
            if response.status == 429 and '/state?' in path:
                # 所有线程共享退避，避免各线程各自重试形成下一轮限流。
                self.gate.backoff()
                write_json(self.budget_path, {'state_rate': 1/self.gate.interval})
            raise ApiError(response.status, path.split('?')[0].rsplit('/', 1)[-1],
                           payload.get('action') if payload else None, detail, meta) from None
        except ssl.SSLCertVerificationError:
            connection.close()
            self.connections.connection = None
            raise ValueError('平台 TLS 证书未受信任。请使用 --ca-file 官方CA文件；'
                             '确认是自签名平台且接受不验证证书时，可显式加 --insecure。') from None
        except (OSError, http.client.HTTPException) as error:
            # 仅记录失败阶段和计时，不保存请求正文、Authorization 或底层异常文本。
            failure = dict(endpoint=path.split('?')[0].rsplit('/', 1)[-1],
                           phase=request_phase, error_type=type(error).__name__,
                           errno=getattr(error, 'errno', None), clock_id=CLOCK_ID,
                           requested_at=requested_at,
                           request_monotonic_ms=(time.monotonic()-requested_monotonic)*1000)
            if '/state?' in path:
                failure['query_seq'] = int(urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)['seq'][0])
            connection.close()
            self.connections.connection = None
            # POST 不盲重试：响应丢失时动作可能已经执行，交给调用方重建快照。
            wrapped = ConnectionError('平台连接中断，需要重新同步状态')
            wrapped.request_failure = failure
            raise wrapped from None

    def recent_state_reads(self, since, *, since_monotonic=None):
        """慢动作附带本线程最近的 state 时序；不保存令牌、牌面或其他桌信息。"""
        rows = getattr(self.connections, 'state_reads', ())
        if since_monotonic is not None:
            return [row for row in rows if row['received_monotonic'] >= since_monotonic]
        return [row for row in rows if since is not None and row['received_at'] >= since]


# SSE

import json
import http.client
from collections import deque
import threading
import time
import urllib.error
import urllib.request


class Notifications:
    def __init__(self, client, gid):
        self.client, self.gid = client, gid
        self.condition = threading.Condition()
        self.seq = -1
        self.connected = False
        self.closed = False
        self.stop = threading.Event()
        self.frames = deque(maxlen=256)

    def noticed_at(self, seq):
        return self.noticed_times(seq)[0]

    def noticed_times(self, seq):
        """同一通知同时保留 Unix/monotonic 时间，缺帧时不伪造延迟。"""
        with self.condition:
            return next(((wall, mono) for watermark, wall, mono in self.frames
                         if watermark >= seq), (None, None))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, name='notify-'+self.gid, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        # 不从另一线程关闭正在 readline 的对象；下一帧/keepalive/timeout 会退出。
        self.stop.set()

    def wait(self, cursor):
        with self.condition:
            self.condition.wait_for(
                lambda: self.seq > cursor or self.closed or not self.connected or self.stop.is_set(),
                timeout=2)
            if self.stop.is_set():
                return False
            if self.seq > cursor or self.closed:
                return True
            if self.connected:
                # SSE 半开或遗漏通知时，每2秒经原有 gate 探测 state。
                # connected 只表示连接尚未报错，不能作为牌局没有进展的依据。
                return True
        # 断流时保留有限频率的恢复轮询，不因通知链路故障停止出牌。
        self.stop.wait(.2)
        return True

    def _run(self):
        while not self.stop.is_set():
            try:
                request = urllib.request.Request(self.client.server+f'/api/games/{self.gid}/notify',
                    headers={'Authorization': 'Bearer '+self.client.token, 'Accept': 'text/event-stream'})
                with urllib.request.urlopen(request, timeout=35, context=self.client.context) as response:
                    with self.condition:
                        self.connected = True
                        self.condition.notify_all()
                    while not self.stop.is_set():
                        line = response.readline()
                        if not line:
                            break
                        if not line.startswith(b'data:'):
                            continue
                        message = json.loads(line[5:])
                        with self.condition:
                            # 初始帧为包含式水位，绝不能推进 GameState 的消费游标。
                            watermark = int(message.get('seq', -1))
                            if watermark > self.seq:
                                self.frames.append((watermark, time.time(), time.monotonic()))
                            self.seq = max(self.seq, watermark)
                            self.closed = bool(message.get('closed'))
                            self.condition.notify_all()
                        if self.closed:
                            return
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
                self.client.record('notify_reconnect', game_id=self.gid, error_type=type(error).__name__)
            finally:
                with self.condition:
                    self.connected = False
                    self.condition.notify_all()
            self.stop.wait(1)


# 公开事件增量状态

import copy


class Projection:
    def __init__(self, snapshot, seq):
        self.snapshot = copy.deepcopy(snapshot)
        self.seq = seq
        self.valid = True

    def apply(self, events):
        for event in events:
            if event['seq'] <= self.seq:
                continue
            if not self.valid or event['seq'] != self.seq + 1:
                self.valid = False
                return
            self.seq = event['seq']
            s=self.snapshot;hero=s['seat'];actor=event['seat'];kind=event['type'];tile=event.get('tile')
            data=event.get('data') or {};god=s['god']
            if kind in ('round_ended','game_ended'):
                self.valid=False
                return
            if kind=='tile_drawn':
                if actor==hero:
                    if tile_index(tile) is None:
                        self.valid=False
                        return
                    s['my_hand'].append(tile)
                supplement = s.pop('_gang_supplement_seat', None) == actor
                s['gang_supplement'] = supplement
                s['wall_remaining']-=1
                s['turn']=actor;s['phase']='draw';s['drawn_tile']=tile if actor==hero else ''
                s['responding_seats']=[];s['hand_counts'][actor]+=1
                if actor==god.get('god_discarder_seat',-1) and not supplement:
                    god['catch_play']=False;god['god_discarder_seat']=-1
                s.pop('window_deadline_ms',None)
            elif kind=='tile_discarded':
                s['gang_supplement']=False
                if actor==hero:
                    s['my_hand'].remove(tile)
                    piao = tile=='白' and god['baotou']
                    if tile=='白' and not piao and not s['drawn_tile']:
                        remaining=[0]*34
                        for t in s['my_hand']:remaining[tile_index(t)]+=1
                        piao=claim_piao(remaining,len(s['melds'][hero]),s['drawn_tile'])
                    god['chain_count']=god.get('chain_count',0)+1 if piao else 0
                s['hand_counts'][actor]-=1;s['discards'][actor].append(tile)
                s['last_discard']=tile;s['last_discarder']=actor;s['drawn_tile']='';s['turn']=actor
                if tile=='白':god['god_discarder_seat']=actor
                god['catch_play']=bool(data.get('catch_play',god.get('catch_play',False)))
                if not god['catch_play']:god['god_discarder_seat']=-1
                origin=god.get('god_discarder_seat',-1)
                s['phase']='response_peng' if tile!='白' else 'await_draw'
                s['responding_seats']=[p for p in range(4) if p!=actor and (not god['catch_play'] or p==origin)] if tile!='白' else []
                s.pop('window_deadline_ms',None)
            elif kind in ('chi','peng','gang'):
                if kind=='peng':used=[tile]*2;tiles=[tile]*3;meld_kind='peng'
                elif kind=='chi':
                    tiles=list(data['tiles']);used=tiles.copy();used.remove(tile);meld_kind='chi'
                else:
                    mode=data.get('mode',data.get('kind'))
                    if mode not in ('an','ming','bu'):
                        self.valid=False
                        return
                    used=[tile]*{'an':4,'ming':3,'bu':1}[mode];tiles=[tile]*4;meld_kind='gang_'+mode
                if actor==hero:
                    for t in used:s['my_hand'].remove(t)
                    if kind=='gang':
                        god['chain_count']=god.get('chain_count',0)+1
                    elif not (god.get('catch_play') and god.get('god_discarder_seat')==hero):
                        god['chain_count']=0
                s['hand_counts'][actor]-=len(used)
                if meld_kind=='gang_bu':
                    meld=next(m for m in s['melds'][actor] if m['kind']=='peng' and tile in m['tiles'])
                    meld.update(kind=meld_kind,tiles=tiles)
                else:
                    s['melds'][actor].append(dict(kind=meld_kind,tiles=tiles))
                s['gang_supplement']=False
                if kind=='gang':s['_gang_supplement_seat']=actor
                s['turn']=actor;s['drawn_tile']='';s['responding_seats']=[]
                s['phase']='await_draw' if kind=='gang' else 'draw'
                s.pop('window_deadline_ms',None)
            elif kind=='timeout' and data.get('kind')=='response':
                # 一个批次可能只含部分窗口关闭事件；吃窗口一律用权威快照核实。
                s['phase']='await_window'
            elif kind=='pass':
                s['responding_seats']=[p for p in s.get('responding_seats',[]) if p!=actor]
            # 平台吃碰后保留旧 baotou；此时多一张待弃牌，不能按待摸牌手牌重算。
            if actor==hero and kind in ('tile_drawn','tile_discarded','gang'):
                counts=[0]*34
                for t in s['my_hand']:counts[tile_index(t)]+=1
                if s['drawn_tile'] and s['turn']==hero:counts[tile_index(s['drawn_tile'])]-=1
                god['baotou']=bool(is_baotou(counts,len(s['melds'][hero])))
            # piao 在观测层由完整公开历史与 chain 恢复，不能保留旧快照的值。
            god.pop('piao',None)

    def actionable(self):
        if not self.valid or self.snapshot['phase'] not in ('draw','response_peng'):
            return False
        view,seat=adapter.from_snapshot(self.snapshot)
        return (view.phase=='draw' and view.turn==seat or
                view.phase=='response_peng' and seat in view.responding_seats)


# 状态与历史恢复

from collections import Counter
import copy
import json
from pathlib import Path



class GameState:
    def __init__(self, path, record=None):
        self.path = Path(path)
        self.record = record
        self.snapshot = None
        self.round_no = None
        self.base = None
        self.cursor = 0
        self.raw = []
        self.projection = None
        self.projected_uses = 0
        self.projection_disabled = False
        self.pending_delta = None
        self._projection_diagnostics = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.__dict__.update({k: data[k] for k in ('snapshot', 'round_no', 'base', 'cursor', 'raw')})

    def save(self):
        write_json(self.path, {k: getattr(self, k) for k in ('snapshot', 'round_no', 'base', 'cursor', 'raw')})

    def append(self, response):
        events = response.get('events') or []
        selected = [e for e in events if e['seq'] > self.cursor]
        if selected and [e['seq'] for e in selected] != list(range(self.cursor + 1, selected[-1]['seq'] + 1)):
            raise ValueError('事件游标出现缺口')
        self.raw.extend(selected)
        if self.projection is not None:
            try:
                self.projection.apply(selected)
            except (ValueError, KeyError, StopIteration, IndexError):
                self.projection.valid = False
        if self.record is not None and self.snapshot is not None:
            for event in selected:
                if (event['type'] == 'timeout' and event['seat'] == self.snapshot['seat']
                        and (event.get('data') or {}).get('kind') == 'discard'):
                    self.record('server_auto_discard', game_id=self.snapshot['game_id'],
                                round_no=self.round_no, seq=event['seq'], server_ts=event.get('ts'),
                                state_read=response.get('_client'),
                                trigger=next((dict(seq=e['seq'], type=e['type'], server_ts=e.get('ts'))
                                              for e in reversed(self.raw) if e['seq'] < event['seq']
                                              and e['seat'] == event['seat']
                                              and e['type'] in ('tile_drawn', 'peng', 'chi')), None))
        if selected:
            self.cursor = selected[-1]['seq']
        return selected

    def sync(self, client, gid, response):
        """快照为真相；历史只用 <= 快照水位的事件，允许缓存更晚事件。"""
        self.pending_delta = None
        snap, stop = response['snapshot'], response['seq']
        if self.round_no != snap['round_no']:
            previous_round, previous_cursor = self.round_no, self.cursor
            boundary = next((e['seq'] for e in reversed(self.raw) if e['type'] == 'round_ended'
                             and (e.get('data') or {}).get('round_no') == snap['round_no'] - 1), None)
            self.round_no, self.raw, self.base = snap['round_no'], [], None
            self.cursor = stop
            if not any(snap['discards']) and not any(snap['melds']):
                self.base = stop
            elif boundary is not None:
                self.base = boundary
                self.cursor = boundary
            elif self.round_no == 1 and stop <= 256:
                # 新对局从 seq=0 开始；首次连接略晚仍可恢复完整首局，
                # 仅限服务端保留的增量范围，首条事件仍须与牌河严格对账。
                self.base = 0
                self.cursor = 0
            elif previous_round == self.round_no - 1 and 0 < previous_cursor < stop:
                # gap 快照可能已含庄家首弃。用协议的局边界判定定位 base，
                # 不能把此快照的 seq 当开局，更不能整局放弃已经公开的历史。
                self.base = self._find_boundary(client, gid, previous_cursor, stop, self.round_no)
                self.cursor = self.base
        if self.base is None and stop > 0:
            # 冷启动或旧缓存缺少局起点时，也必须补齐当前局公开历史。
            # 原实现仅恢复相邻跨局，接管第 2 局以后会整局使用缺失历史。
            # 仍只请求本人 state 增量；首事件沿用严格对账，无法恢复则不做推理。
            self.base = self._find_boundary(client, gid, 0, stop, self.round_no)
            self.cursor = self.base
        if self.base is not None and self.cursor < stop:
            query = self.cursor
            first = query == self.base
            # 平台新局边界游标返回 gap；首条公开事件由快照和余下事件严格对账恢复。
            if first:
                query += 1
            delta = client.api(f'/api/games/{gid}/state?seq={query}') if query < stop else {'events': []}
            if delta.get('snapshot'):
                if delta['snapshot']['round_no'] != self.round_no:
                    return self.sync(client, gid, delta)
                raise ValueError('历史游标仍返回同局 gap，不能冒充完整历史')
            if first:
                self.raw.append(self._first_event(snap, query, delta.get('events') or [], stop))
                self.cursor = query
            # 后续事件可能包含“轮到我摸牌”。这里只补齐当前快照，
            # 不能提前推进游标，否则下一轮会跳过这些尚未决策的事件。
            self.append({'events': [e for e in delta.get('events') or [] if e['seq'] <= stop],
                         '_client': delta.get('_client')})
            later = [e for e in delta.get('events') or [] if e['seq'] > stop]
            if later:
                # 暂存本次已收到的事件，不推进游标，也不混入旧快照的观测。
                self.pending_delta = dict(events=later, _client=delta.get('_client') or {})
            elif (delta.get('seq') == stop and self.cursor == stop
                  and not delta.get('finished') and snap.get('phase') == 'draw'
                  and snap.get('turn') == snap.get('seat')
                  and self.public_events(response) is not None
                  and (delta.get('_client') or {}).get('received_at') is not None):
                # 补历史的完整增量确认了相同水位；弃牌状态变化必有公开事件。
                # 只复用我方 draw，不能据同 seq 推断吃碰窗口或局间阶段未变化。
                # 用新增量的真实接收时间，同时保留原快照时间，禁止改写旧时间戳。
                self.pending_delta = dict(events=[], _client=delta['_client'],
                                          _confirmed_seq=stop,
                                          _snapshot_client=response.get('_client') or {})
        if response.get('_source') != 'events':
            if (self.projection is not None and self.projection.actionable()
                    and self.projection.seq == stop and self.base is not None
                    and self.projection.snapshot['phase'] == snap['phase']):
                import numpy as np
                history = self.public_events(response)
                actual, _, _ = observation(snap, public_events=history)
                projected, _, _ = observation(self.projection.snapshot, public_events=history)
                matched = (actual is None and projected is None or
                           actual is not None and projected is not None and np.array_equal(actual, projected))
                if not matched:
                    self.projection_disabled = True
                    if self.record is not None:
                        comparable = (actual is not None and projected is not None
                                      and actual.shape == projected.shape)
                        indices = np.flatnonzero(actual != projected) if comparable else []
                        # 保存失败现场供赛后核对；仅包含当前用户快照和公开历史。
                        # 写日志交给 play loop 在本次决策之后处理，不阻塞动作提交。
                        self._projection_diagnostics.append(dict(
                            game_id=snap['game_id'], round_no=snap['round_no'], seq=stop,
                            response=copy.deepcopy(response),
                            projected_snapshot=copy.deepcopy(self.projection.snapshot),
                            public_events=copy.deepcopy(history),
                            feature_difference_count=len(indices) if comparable else None,
                            feature_differences=[dict(index=int(i), actual=float(actual[i]),
                                                      projected=float(projected[i])) for i in indices[:32]]))
                if self.record is not None:
                    self.record('projection_check', game_id=snap['game_id'], seq=stop, matched=bool(matched))
            self.projected_uses = 0
        else:
            self.projected_uses += 1
        # 增量 Projection 沿用训练时的完整历史牌河，避免从剩余牌河继续 append。
        canonical = canonical_snapshot(snap, self.public_events(response))
        self.projection = Projection(canonical, stop)
        self.snapshot = copy.deepcopy(snap)
        if self.base is None:
            self.cursor = max(self.cursor, stop)
        self.save()
        return snap

    def take_projection_diagnostics(self):
        """取走本轮校验失败现场；不持久化到恢复状态，避免重启后重复写入。"""
        pending, self._projection_diagnostics = self._projection_diagnostics, []
        return pending

    def advance_from_history(self):
        """旧快照过期时，尝试用补历史响应生成新决策视图；不确定则交回权威刷新。"""
        delta, self.pending_delta = self.pending_delta, None
        if delta is None:
            return None
        self.append(delta)
        response = self.immediate_response(delta)
        if response is not None:
            response['_history_fast_forward'] = True
            if delta.get('_confirmed_seq') == self.cursor:
                response['_history_confirmation'] = dict(
                    seq=self.cursor, snapshot_client=delta['_snapshot_client'])
                if self.record is not None:
                    self.record('history_watermark_confirmed',
                                game_id=response['snapshot']['game_id'], seq=self.cursor,
                                snapshot_client=delta['_snapshot_client'],
                                confirmation_client=delta['_client'])
        return response

    def immediate_response(self, delta):
        if (self.base is not None and not self.projection_disabled and self.projected_uses < 10
                and self.projection is not None and self.projection.actionable()
                and self.projection.seq == self.cursor):
            return dict(snapshot=copy.deepcopy(self.projection.snapshot), seq=self.cursor,
                        _source='events', _client=delta.get('_client', {}))
        return None

    @staticmethod
    def _find_boundary(client, gid, previous, stop, round_no):
        low, high = previous, stop - 1
        while low < high:
            middle = (low + high + 1) // 2
            result = client.api(f'/api/games/{gid}/state?seq={middle}')
            if result.get('snapshot'):
                if result['snapshot']['round_no'] != round_no:
                    raise ValueError('查找局边界时对局再次推进，需要重新同步')
                low = middle
            else:
                high = middle - 1
        return low

    def trigger_seq(self, snap):
        hero = snap['seat']
        for event in reversed(self.raw):
            kind, data = event['type'], event.get('data') or {}
            if snap['phase'] == 'draw' and event['seat'] == hero and kind in ('tile_drawn', 'peng', 'chi'):
                return event['seq']
            if snap['phase'] == 'response_peng' and kind == 'tile_discarded':
                return event['seq']
            if snap['phase'] == 'response_chi' and kind == 'timeout' and data.get('window') == 'peng':
                return event['seq']
        return self.base

    def _first_event(self, snap, seq, later, stop):
        # 新快照可能已移除首弃（被吃碰明杠取走）；逐个候选严格回放，不能仅做 Counter 相减。
        known = [e for e in later if e['seq'] <= stop]
        concealed = Counter(m['tiles'][0] for m in snap['melds'][snap['dealer']]
                            if m['kind'] == 'gang_an')
        concealed.subtract(e['tile'] for e in known
                           if e['type'] == 'gang' and e['seat'] == snap['dealer']
                           and (e.get('data') or {}).get('kind', (e.get('data') or {}).get('mode')) == 'an')
        missing = list(concealed.elements())
        if len(missing) == 1 and all(v >= 0 for v in concealed.values()):
            # 快照存在尚未被后续事件解释的暗杠，首条不能再猜为弃牌。
            candidates = [dict(seq=seq, type='gang', seat=snap['dealer'],
                               tile=missing[0], data={'kind': 'an'})]
        elif all(v == 0 for v in concealed.values()):
            candidates = [dict(seq=seq, type='tile_discarded', seat=snap['dealer'], tile=tile, data={})
                          for tile in ALL_TILES]
        else:
            raise ValueError('首条事件无法与公开快照唯一对账')
        matched = []
        for candidate in candidates:
            try:
                canonical_snapshot(snap, public_events_from_raw([candidate, *known]))
            except ValueError:
                continue
            matched.append(candidate)
        if len(matched) != 1:
            raise ValueError('首条事件无法与公开快照唯一对账')
        return matched[0]

    def public_events(self, response):
        if self.base is None:
            return None
        stop = response['seq']
        raw = [e for e in self.raw if e['seq'] <= stop]
        if [e['seq'] for e in raw] != list(range(self.base + 1, stop + 1)):
            return None
        result = public_events_from_raw(raw)
        canonical_snapshot(response['snapshot'], result)
        return result

    def awaiting_next_round(self):
        """局间 settled 或结算事件均须直接 poll；新局发牌可能不增加 seq。"""
        # 重连可能只拿到上一局终态，缺少 round_ended 历史；仍按原游标等新局。
        if self.snapshot is not None and self.snapshot.get('phase') == 'settled':
            return True
        return self.round_no is not None and any(
            event['type'] == 'round_ended'
            and (event.get('data') or {}).get('round_no') == self.round_no
            for event in reversed(self.raw)
        )

    def needs_snapshot(self, events):
        """只省略确定无须我方决策的变化；决策前仍以服务端快照核实阶段。"""
        if self.snapshot is None or self.base is None:
            return True
        snap = self.snapshot
        hero = snap['seat']
        counts = Counter(snap['my_hand'])
        view, _ = adapter.from_snapshot(snap)
        last = None
        discarder = None
        for event in self.raw:
            if event['type'] == 'tile_discarded':
                last, discarder = event['tile'], event['seat']
        for event in events:
            kind, seat = event['type'], event['seat']
            if kind in ('round_ended', 'game_ended') or (seat == hero and kind in ('tile_drawn', 'peng', 'chi', 'gang')):
                return True
            if kind == 'tile_discarded' and seat != hero and event['tile'] != '白':
                if counts[event['tile']] >= 2:
                    return True
            if (kind == 'timeout' and (event.get('data') or {}).get('window') == 'peng'
                    and discarder is not None and (discarder + 1) % 4 == hero):
                if adapter.chi_options(view, hero, tile_index(last)):
                    return True
        return False


# 单桌决策提交

import json
from pathlib import Path
import time
import urllib.error



# 结算/局间等待可能没有新的 SSE seq；保留直接 state 探测，但不能以 200ms
# 频率让多桌同时轮询，超过共享 state gate 后会把动作窗口一起排掉。
ROUND_BOUNDARY_POLL_INTERVAL = 1.0


def action_window_remaining_ms(snapshot, *, now_ms=None):
    """返回服务端动作窗口剩余时间；协议未提供 deadline 时返回 ``None``。"""
    deadline_ms = snapshot.get('window_deadline_ms')
    if not isinstance(deadline_ms, (int, float)):
        return None
    return deadline_ms - (time.time() * 1000 if now_ms is None else now_ms)


def play_game(client, policy, gid, config, deadline, *, stop_event=None):
    with Notifications(client, gid) as notifications:
        return _play_game(client, policy, gid, config, deadline, notifications,
                          stop_event=stop_event)


def _play_game(client, policy, gid, config, deadline, notifications, *, stop_event=None):
    previous = None
    saved_round = None
    state = GameState(client.output / 'states' / (gid + '.json'), record=client.record)
    refresh = True
    response = None
    while (deadline is None or time.time() < deadline) and not (stop_event and stop_event.is_set()):
        pending_round_snapshot = None
        notified_at = notified_monotonic = None
        reading_delta = False
        try:
            if refresh:
                response = client.api(f'/api/games/{gid}/state?seq=0')
                refresh = False
            else:
                waiting_for_round = state.awaiting_next_round()
                if not waiting_for_round and not notifications.wait(state.cursor):
                    continue
                reading_delta = state.cursor > 0
                # 新 SSE 水位后的首个增量读取决定是否存在动作窗口；若先按
                # 普通轮询排队，1 秒吃碰和 3 秒弃牌会在“第一跳”就被耗尽。
                # urgent 仍走同一 RequestGate，既不提高速率也不绕过退避。
                delta = client.state_delta(gid, state.cursor, notifications,
                                           urgent=not waiting_for_round and notifications.seq > state.cursor)
                reading_delta = False
                if delta.get('snapshot'):
                    response = delta
                else:
                    events = state.append(delta)
                    state.save()
                    if not state.needs_snapshot(events):
                        if state.awaiting_next_round():
                            # 新局可能不增加 seq；保持直接探测，但按 1 秒节奏限流，
                            # 避免多桌 settled 轮询把共享 state gate 排队打满。
                            time.sleep(ROUND_BOUNDARY_POLL_INTERVAL)
                        continue
                    response = state.immediate_response(delta)
                    if response is None:
                        # 已确认可能需要我方决策；吃窗口仍读取权威快照，第二跳优先完成。
                        # gate 限制连续优先次数并保留普通请求晋升，不挤死其他桌第一跳。
                        response = client.api(f'/api/games/{gid}/state?seq=0',
                                              decision_refresh=not state.awaiting_next_round()
                                              and not any(e.get('type') == 'game_ended' for e in events))
            snap = response.get('snapshot')
            if not snap:
                time.sleep(.2)
                continue
            if response.get('finished') or snap.get('phase') == 'finished':
                # 终局 API 只返回最终快照，不再提供增量；无需为已结束的对局补历史。
                client.record('game_finished', game_id=gid, response=response)
                return
            snap = state.sync(client, gid, response)
            # sync 可能发现已进入下一局，以返回的权威新快照重新开始。
            if snap['round_no'] != response['snapshot']['round_no']:
                refresh = True
                continue
            if state.awaiting_next_round():
                # 带快照的 settled 响应也必须限流，与空增量走相同节奏。
                time.sleep(ROUND_BOUNDARY_POLL_INTERVAL)
                continue
            snapshot_age = response_age(response)
            if snapshot_age > (.2 if snap['phase'].startswith('response_') else .75):
                # 补历史可能已读到更新事件：同步推进观测与游标，避免再排一次 GET。
                # 无法确定最新阶段时仍刷新，绝不能用旧快照配上已前进的游标。
                advanced = state.advance_from_history()
                if advanced is None:
                    refresh = True
                    continue
                response = advanced
                snap = state.sync(client, gid, response)
                snapshot_age = response_age(response, default=float('inf'))
                if snapshot_age > (.2 if snap['phase'].startswith('response_') else .75):
                    refresh = True
                    continue
            snapshot_ready_at = time.monotonic()
            received_at = response.get('_client', {}).get('received_at')
            snapshot_age_wall_ms = (time.time()-received_at)*1000 if received_at is not None else None
            if snap.get('round_no') != saved_round:
                saved_round = snap.get('round_no')
                pending_round_snapshot = response
            key = json.dumps([snap.get(k) for k in ('round_no', 'phase', 'turn', 'my_hand',
                'discards', 'melds', 'last_discard', 'drawn_tile', 'responding_seats',
                'window_deadline_ms', 'god')], sort_keys=True)
            if key != previous:
                tick = time.monotonic()
                pre_decision_ms = (tick-snapshot_ready_at)*1000
                ycbk = bool(config.get('YouCaiBiKao'))
                events = state.public_events(response)
                trigger_seq = state.trigger_seq(snap)
                if trigger_seq is not None:
                    notified_at, notified_monotonic = notifications.noticed_times(trigger_seq)
                postprocess_info = {}
                with policy.decision() as model:
                    payload, obs = decide(model.policy, snap, ycbk, events, with_observation=True, honor_discard_postprocess=getattr(policy, "discard_postprocess", False), postprocess_info=postprocess_info)
                    model_sha256 = model.sha256
                observation_policy_ms = (time.monotonic()-tick)*1000
                if stop_event and stop_event.is_set():
                    return
                if payload:
                    observation_file = None
                    action_request_ms = None
                    if payload['action'] != 'pass':
                        deadline_ms = snap.get('window_deadline_ms')
                        remaining_ms = action_window_remaining_ms(snap)
                        if remaining_ms is not None and remaining_ms <= 0:
                            # 服务端窗口已经结束时，POST 必然失效且可能掩盖真正的
                            # 超时链路。重新取权威状态，不把这次旧快照标记为已处理。
                            client.record('stale_action_window', game_id=gid,
                                          round_no=snap.get('round_no'), snapshot_seq=response['seq'],
                                          phase=snap.get('phase'), action=payload['action'],
                                          window_deadline_ms=deadline_ms,
                                          remaining_ms=remaining_ms,
                                          snapshot_age_ms=snapshot_age * 1000,
                                          state_queue_ms=response.get('_client', {}).get('queue_ms'),
                                          state_http_ms=response.get('_client', {}).get('http_ms'))
                            refresh = True
                            continue
                        action_tick = time.monotonic()
                        try:
                            client.api(f'/api/games/{gid}/action', payload)
                        finally:
                            action_request_ms = (time.monotonic()-action_tick)*1000
                    submitted_at = time.time()
                    submitted_monotonic = time.monotonic()
                    notify_ms = (submitted_at-notified_at)*1000 if notified_at is not None else None
                    notify_monotonic_ms = ((submitted_monotonic-notified_monotonic)*1000
                                           if notified_monotonic is not None else None)
                    previous = key
                    if obs[-74:].sum() > 1:
                        # 保存真正送入模型的输入，便于按 seq 回放核对；不包含他家暗牌。
                        import numpy as np
                        directory = client.output / 'observations'
                        directory.mkdir(exist_ok=True)
                        observation_file = f'{gid}-r{snap["round_no"]}-s{response["seq"]}-{snap["phase"]}.npz'
                        np.savez_compressed(directory / observation_file, obs=obs)
                    client.record('action', game_id=gid, round_no=snap.get('round_no'),
                                  payload=payload, model_sha256=model_sha256,
                                  honor_discard_postprocess=getattr(policy, "discard_postprocess", False),
                                  postprocess=postprocess_info or None,
                                  public_history_available=events is not None,
                                  snapshot_seq=response['seq'], public_event_count=len(events) if events is not None else None,
                                  trigger_seq=trigger_seq,
                                  snapshot_source=response.get('_source', 'snapshot'),
                                  history_fast_forward=response.get('_history_fast_forward', False),
                                  observation_file=observation_file, snapshot_age_ms=snapshot_age*1000,
                                  snapshot_age_wall_ms=snapshot_age_wall_ms,
                                  submitted_at=submitted_at,
                                  submitted_monotonic=submitted_monotonic, clock_id=CLOCK_ID,
                                  pre_decision_ms=pre_decision_ms,
                                  observation_policy_ms=observation_policy_ms,
                                  action_request_ms=action_request_ms,
                                  post_submit_ms=(time.time()-submitted_at)*1000,
                                  post_submit_monotonic_ms=(time.monotonic()-submitted_monotonic)*1000,
                                  notify_to_submit_ms=notify_ms,
                                  notify_to_submit_monotonic_ms=notify_monotonic_ms,
                                  notify_clock_shift_ms=notify_ms-notify_monotonic_ms
                                      if notify_ms is not None and notify_monotonic_ms is not None else None,
                                  state_request_ms=(response.get('_client', {}).get('received_at',0)-response.get('_client', {}).get('requested_at',0))*1000,
                                  state_urgent=response.get('_client', {}).get('urgent'),
                                  state_decision_refresh=response.get('_client', {}).get('decision_refresh'),
                                  state_queue_ms=response.get('_client', {}).get('queue_ms'),
                                  state_connect_ms=response.get('_client', {}).get('connect_ms'),
                                  state_http_ms=response.get('_client', {}).get('http_ms'),
                                  state_request_monotonic_ms=response.get('_client', {}).get('request_monotonic_ms'),
                                  state_queue_monotonic_ms=response.get('_client', {}).get('queue_monotonic_ms'),
                                  state_connect_monotonic_ms=response.get('_client', {}).get('connect_monotonic_ms'),
                                  state_http_monotonic_ms=response.get('_client', {}).get('http_monotonic_ms'),
                                  state_reads=client.recent_state_reads(notified_at, since_monotonic=notified_monotonic)
                                      if any(value is not None and value > 750
                                             for value in (notify_ms, notify_monotonic_ms)) else None,
                                  decision_ms=(time.monotonic()-tick)*1000)
            # 游标长轮询在下一事件前等待，不再固定频率反复获取同一快照。
        except ApiError as error:
            # 增量 GET 被限流时尚未消费事件，原游标仍有效；保留共享退避后重试。
            # 仅限这个读取阶段，补历史/快照失败或其他错误仍走权威刷新。
            retry_state_delta = reading_delta and error.status == 429 and error.endpoint == 'state'
            client.record('game_api_error', game_id=gid, status=error.status,
                          retry_state_delta=retry_state_delta,
                          endpoint=error.endpoint, action=error.action, detail=error.detail,
                          request_timing=error.timing,
                          notify_to_submit_ms=(time.time()-notified_at)*1000 if error.endpoint == 'action' and notified_at is not None else None,
                          notify_to_submit_monotonic_ms=(time.monotonic()-notified_monotonic)*1000
                              if error.endpoint == 'action' and notified_monotonic is not None else None,
                          clock_id=CLOCK_ID,
                          snapshot=response if error.endpoint == 'action' else None,
                          state_reads=client.recent_state_reads(notified_at, since_monotonic=notified_monotonic)
                              if error.endpoint == 'action' else None,
                          attempted_payload=payload if error.endpoint == 'action' else None,
                          observation_policy_ms=observation_policy_ms if error.endpoint == 'action' else None,
                          action_request_ms=action_request_ms if error.endpoint == 'action' else None,
                          decision_ms=(time.monotonic()-tick)*1000 if error.endpoint == 'action' else None)
            if error.status in (401, 403):
                raise
            if error.status == 404:
                return
            if error.status == 409:
                previous = key
            refresh = not retry_state_delta
            # 409 后重取快照；同一失效动作不重复提交。
            # state 的 429 已由共享 gate 退避；额外停整秒会错过下一次吃碰窗口。
            time.sleep(.02 if error.status == 429 else .05)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            # 增量 GET 失败尚未消费事件；和 429 一样保留原游标，避免快照/补历史放大故障。
            # POST、快照及 sync 中的网络错误仍刷新，绝不盲重发可能已经执行的动作。
            client.record('network_retry', game_id=gid, retry_state_delta=reading_delta,
                          cursor=state.cursor, error_type=type(error).__name__,
                          request_failure=getattr(error, 'request_failure', None))
            refresh = not reading_delta
            time.sleep(.05)
        except ValueError as error:
            client.record('state_resync', game_id=gid, reason=str(error), cursor=state.cursor,
                          base_seq=state.base, phase=snap.get('phase') if response else None)
            # 缺口显式降为未知，下一局从首个快照完整恢复；不能伪造历史。
            state.base = None
            state.raw.clear()
            refresh = True
        finally:
            for diagnostic in state.take_projection_diagnostics():
                client.record('projection_mismatch', **diagnostic)
            if pending_round_snapshot is not None:
                # 完整快照仍保存；写文件/输出延后到本次决策和POST之后。
                # 即使动作被拒绝也保留该局面，不能为提速丢弃诊断证据。
                log_tick = time.monotonic()
                client.record('round_snapshot', game_id=gid, response=pending_round_snapshot)
                client.record('round_snapshot_written', game_id=gid, round_no=saved_round,
                              log_ms=(time.monotonic()-log_tick)*1000)


# 比赛与打榜生命周期

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import urllib.error


TERMINAL = {"finished", "closed", "void"}
NETWORK = (TimeoutError, ConnectionError, urllib.error.URLError)


def game_id(value):
    return value.get("game_id") if isinstance(value, dict) else value


@contextmanager
def workers(max_workers, stop):
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="game")
    try:
        yield pool
    finally:
        # 主线程退出时先通知其他桌；不等待所有桌自然结束才退出。
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)


def retryable(error):
    return isinstance(error, NETWORK) or isinstance(error, ApiError) and error.status in (429, 500, 502, 503, 504)


def request(client, path, stop, payload=None):
    """仅用于 GET 和文档声明幂等的 register/ready；绝不重发 action/match。"""
    delay = .5
    while not stop.is_set():
        try:
            return client.api(path, payload)
        except (ApiError, *NETWORK) as error:
            if not retryable(error):
                raise
            client.record("controller_retry", path=path, error_type=type(error).__name__,
                          status=getattr(error, "status", None), delay_seconds=delay)
            stop.wait(delay)
            delay = min(5, delay * 2)
    raise InterruptedError("运行已停止")


def attendance(client, tid, endpoint, stop):
    try:
        request(client, f"/api/tournaments/{tid}/{endpoint}", stop, {})
    except ApiError as error:
        if error.status != 409:
            raise
        # 开赛与请求竞争时重新读状态；不能把拒绝记作确认成功。
        client.record("attendance_race", endpoint=endpoint, status=409)
        return False
    client.record("attendance_accepted", endpoint=endpoint, tournament_id=tid)
    return True


def reap(futures, completed):
    for gid, future in list(futures.items()):
        if future.done():
            future.result()  # 实现错误/身份错误必须向调用方传播。
            completed.add(gid)
            del futures[gid]


def tournament(client, policy, me, tournament_id=None, *, stop=None, play=play_game, poll=1):
    stop = stop or threading.Event()
    bound = me.get("tournament_id")
    if bound and tournament_id and tournament_id != bound:
        raise ValueError("配置 tournament_id 与参赛令牌绑定赛事不同")
    tid = bound or tournament_id
    if not tid:
        raise ValueError("全局令牌必须明确提供 tournament_id；也可使用门户派发的参赛令牌")
    if bound:
        request(client, "/api/tournaments/me/rules", stop)
    else:
        # 全局身份未报名时不能先读赛事详情。
        attendance(client, tid, "register", stop)
    futures, completed, ready = {}, set(), set()
    last_marker = None
    with workers(16, stop) as pool:
        while not stop.is_set():
            reap(futures, completed)
            state = request(client, f"/api/tournaments/{tid}", stop)
            write_json(client.output / "status.json", state)
            status = state.get("status")
            stage = state.get("stage") or {}
            key = (stage.get("no"), stage.get("name"), status)
            marker = (key, state.get("qualified"), state.get("stage_crashed"))
            if marker != last_marker:
                client.record("tournament_status", tournament_id=tid, status=status,
                              stage=stage, qualified=state.get("qualified"),
                              qualify_role=state.get("qualify_role"), stage_crashed=state.get("stage_crashed"))
                last_marker = marker
            if status not in ("registering", "stage_open"):
                # 中断后同一个阶段重新开放时，也需要再次确认。
                ready.clear()
            if status in TERMINAL:
                client.record("tournament_finished", tournament_id=tid, status=status, result=state)
                return
            if status == "registering" and key not in ready:
                registered = attendance(client, tid, "register", stop)
                if registered and attendance(client, tid, "ready", stop):
                    ready.add(key)
            elif status == "stage_open" and state.get("qualified") and key not in ready:
                if attendance(client, tid, "ready", stop):
                    ready.add(key)
            elif status == "running":
                known = {game_id(value) for value in state.get("my_games") or []}
                active = request(client, "/api/me", stop).get("active_games") or []
                # 全局令牌可能还有别的房间；空 my_games 绝不扩大范围。
                for gid in dict.fromkeys(game_id(value) for value in active):
                    if gid and gid in known and gid not in completed and gid not in futures:
                        if len(futures) >= 16:
                            break
                        futures[gid] = pool.submit(play, client, policy, gid,
                                                   state.get("config") or {}, None, stop_event=stop)
                        client.record("tournament_game_started", game_id=gid, tournament_id=tid)
            # stage_done、暂无桌、未获得本轮资格均持续等待，直到整场终态。
            stop.wait(min(.25, poll) if status == "running" and not futures else poll)


def ranking(client, policy, me, resume_room=None, *, stop=None, play=play_game, poll=2):
    if me.get("tournament_id"):
        raise ValueError("打榜需要全局身份令牌，不能使用参赛专用令牌")
    if me.get("active_games") and not resume_room:
        raise ValueError("账号仍有进行中的对局，请指定 --resume-room 接管原房间")
    stop = stop or threading.Event()
    room, futures, completed = resume_room, {}, set()
    with workers(10, stop) as pool:
        while not stop.is_set():
            reap(futures, completed)
            if room is None:
                # 非幂等 match 出现不确定失败就退出；先核查账号再显式接管。
                try:
                    room = client.api("/api/match", {})["room_id"]
                except NETWORK as error:
                    # 保留失败阶段用于定位；不重发可能已执行的匹配请求。
                    client.record("match_network_failure", request_failure=getattr(error, "request_failure", None))
                    raise
                client.record("matched", room_id=room)
            try:
                state = request(client, f"/api/tournaments/{room}", stop)
            except ApiError as error:
                if error.status != 404:
                    raise
                # 房间短暂不可见不等于匹配失败；先核对本人仍在该房间。
                # 只重读GET，不重发match，也不接管其他房间。
                # match 已成功，但 me 的 active 索引也可能尚未可见。
                # 最多补查十次身份；只有确认同房本人 active 才重读房间。
                # 全程不重发 match，也不因其他房间 active 而接管。
                client.record("room_visibility_check", room_id=room, status=404)
                for attempt in range(10):
                    try:
                        identity = request(client, "/api/me", stop)
                    except ApiError as identity_error:
                        client.record("room_identity_error", room_id=room,
                                      status=identity_error.status)
                        raise
                    active = identity.get("active_games") or []
                    owned = [game_id(g) for g in active
                             if isinstance(game_id(g), str) and game_id(g).startswith(room + "_r")]
                    if owned or active:
                        break
                    client.record("room_identity_pending", room_id=room, status=404,
                                  attempt=attempt + 1)
                    if attempt < 9 and stop.wait(.5):
                        raise InterruptedError("运行已停止")
                if not owned:
                    client.record("room_identity_unconfirmed", room_id=room, status=404)
                    raise
                client.record("room_visibility_retry", room_id=room, status=404,
                              owned_active_games=len(owned))
                stop.wait(2)
                continue
            write_json(client.output / "status.json", state)
            if state.get("status") in TERMINAL:
                client.record("room_finished", room_id=room, result=state)
                for future in futures.values():
                    future.result(timeout=40)
                futures.clear()
                completed.clear()
                room = None
                stop.wait(7)
                continue
            for value in state.get("my_games") or []:
                gid = game_id(value)
                if gid and gid not in futures and gid not in completed and len(futures) < 10:
                    futures[gid] = pool.submit(play, client, policy, gid,
                                               state.get("config") or {}, None, stop_event=stop)
            stop.wait(min(.25, poll) if not futures else poll)
