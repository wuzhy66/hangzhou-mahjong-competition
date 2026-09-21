"""上游遗漏 S2 经验先验时的依赖适配；默认 S1+S6 不使用该模块。"""


def p_hu(missing, drawable):
    """拒绝用未经校准的概率替代上游经验数据。"""
    raise RuntimeError(
        "S2_fan_ev_discard 需要上游 hzmj/prior.py 的经验先验；"
        "请补齐该文件或关闭 S2。"
    )
