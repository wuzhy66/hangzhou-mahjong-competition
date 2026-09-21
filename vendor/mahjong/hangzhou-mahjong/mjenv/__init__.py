"""杭州麻将本地环境模拟（AEC 多智能体 + API 无感替换）。

不修改 hzmj / bot_ai 任何一行，全部能力以「运行期注入」方式提供：

    fastrule   hu.py 热点的批量+分组缓存加速版，patch() 后 RuleBot 直接提速
    core       把 hzmj.engine.Game 包成 AEC 序列化状态机
    spaces     离散动作空间（74 维）与合法动作掩码
    encode     观测编码（15 平面 × 34 + 16 标量 = 526 维）
    aec_env    PettingZoo AEC 接口（不依赖 pettingzoo，纯标准库）
    opponents  对手池
    http_mock  拦截 bot_ai.api()，用本地引擎伪装远程平台
    run_bot    装配入口
    tests      正确性/等价性/性能验证
"""

__all__ = ["fastrule", "core", "spaces", "encode", "aec_env",
           "opponents", "http_mock"]
