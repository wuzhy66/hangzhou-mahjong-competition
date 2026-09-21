"""RuleBot 策略开关：全关基线、显式消融配置及每 2 秒检查一次的文件热重载。"""

import json
from pathlib import Path
import time


KEYS = (
    "S1_loose_gang",
    "S2_fan_ev_discard",
    "S3_dealer_asymmetry",
    "S4_wall_timeline",
    "S5_god_nonlinear",
    "S6_meld_cost",
    "S7_opponent_threat",
    "S8_catch_play_tactic",
    "S9_ycbk_mode",
)
DEFAULTS = dict.fromkeys(KEYS, False)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "mahjong" / "rulebot_strategy.json"
_cached = None
_checked_at = float("-inf")


def normalize(values):
    """补齐未指定的开关；拼写错误和非 bool 值直接报错。"""
    if not isinstance(values, dict):
        raise TypeError("策略开关必须是 JSON object / dict")
    unknown = set(values) - set(KEYS)
    if unknown:
        raise ValueError("未知策略开关：%s" % sorted(unknown))
    if any(type(value) is not bool for value in values.values()):
        raise TypeError("策略开关值必须为 bool")
    return {**DEFAULTS, **values}


def make(*keys):
    """构造全关基线上的显式组合，保留消融实验测试 S2×S9 的能力。"""
    return normalize(dict.fromkeys(keys, True))


def _sanitize(values):
    result = normalize(values)
    if result["S2_fan_ev_discard"] and result["S9_ycbk_mode"]:
        raise ValueError("strategy.json 禁止同时启用 S2_fan_ev_discard 与 S9_ycbk_mode")
    return result


def current():
    """相对模块位置读取配置，与工作目录无关；缺失文件时使用全关基线。"""
    global _cached, _checked_at
    now = time.monotonic()
    if _cached is None or now - _checked_at >= 2.0:
        try:
            raw = CONFIG_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            values = {}
        else:
            values = json.loads(raw)
        # 无效配置直接报错，不静默保留旧策略或改成其他对手。
        _cached = _sanitize(values)
        _checked_at = now
    return dict(_cached)


def active():
    values = current()
    return [key for key in KEYS if values[key]]


if __name__ == "__main__":
    print(json.dumps(current(), ensure_ascii=False, indent=2))
