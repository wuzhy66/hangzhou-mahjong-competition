# 原样提取的推理依赖

`SOURCES.json` 记录 WSL 来源路径、父仓库 HEAD 和每个文件实际 SHA256。HEAD 不代替文件 hash：来源可包含用户尚未提交的修改。31 个文件均逐字节复制，没有删除 import、拼接函数或内联兼容补丁。

`mahjong/platform/decision.py` 执行原来的 engine_compat.install / fastrule.patch。观测、动作空间、river、规则兼容、OnnxPolicy 和后处理继续按原 package 关系调用。hzmj / mjenv 仅提取该调用链需要的完整模块；没有训练入口及训练依赖。

顶层 observation.py / postprocess.py 只兼容客户端原 import 名。inference_runtime.py 设置独立路径并拒绝已加载的外部同名模块。Windows 仅替换 native 库加载函数；native_runtime.py 直接编译这里的原始 C++，通过 linker 导出 ABI。Linux 使用原始加载函数。

此提取范围是推理核心及依赖。platform_client.py 的网络、状态恢复、比赛调度仍是独立客户端实现，不能据此声称整个客户端已逐文件等同 WSL。
