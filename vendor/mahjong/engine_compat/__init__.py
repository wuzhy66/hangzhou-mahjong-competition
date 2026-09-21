"""为不完整的上游发布提供训练所需依赖，优先使用 submodule 自带实现。"""

from pathlib import Path


def install():
    """在导入 mjenv 前调用；补齐依赖与平台规则兼容，不改写上游文件。"""
    import hzmj

    fallback = str(Path(__file__).resolve().parent)
    if fallback not in hzmj.__path__:
        hzmj.__path__.append(fallback)
    from .piao import install as install_piao
    install_piao()
    from .settlement import install as install_settlement
    install_settlement()
    from .catch_play import install as install_catch_play
    install_catch_play()
    from .win import install as install_win
    install_win()
    from mahjong.rule_ai import install as install_rule_ai
    install_rule_ai()
