"""生产规则策略入口；环境初始化时安装，不依赖研究目录。"""
import sys


def install():
    from hzmj import bots
    from mjenv import fastrule
    from .policy import RuleBot

    fastrule.patch()
    bots.RuleBot = RuleBot
    bots.BOT_REGISTRY['rule'] = RuleBot
    # 兼容先加载 opponents 再初始化环境的调用顺序。
    opponents = sys.modules.get('mjenv.opponents')
    if opponents is not None:
        opponents.RuleBot = RuleBot
