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
