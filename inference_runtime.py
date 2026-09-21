"""独立运行时入口：原样源码优先，Windows 仅替换动态库加载。"""
import os
from pathlib import Path
import sys

VENDOR = Path(__file__).parent / 'vendor'
# 拒绝复用已加载的外部同名模块，防止验证误用 WSL 仓库。
for prefix in ('mahjong', 'onnx_tools', 'hzmj', 'mjenv'):
    for name, module in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + '.'):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(VENDOR.resolve()):
                raise RuntimeError(f'外部模块已加载，需使用独立 Python 进程：{name}')
sys.path.insert(0, str(VENDOR))
# decision 原样执行 engine_compat.install 和 fastrule.patch。
from mahjong.platform import decision
if os.name == 'nt':
    from native_runtime import native_search
    from mahjong.rule_ai import policy
    from mahjong.platform import discard_search
    policy.native = native_search
    discard_search.native = native_search
    discard_search.native_search = native_search
