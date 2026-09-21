"""旧入口兼容层；后处理算法使用 WSL 原文件。"""
import inference_runtime
from mahjong.platform.postprocess import *
from mahjong.platform.discard_search import feature, choose, evaluate, native_search
from mahjong.rule_ai.policy import value, Estimate
