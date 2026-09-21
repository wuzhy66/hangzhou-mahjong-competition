"""Windows 动态库编译与 ABI 适配；直接编译原始 C++ 文件。"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

@lru_cache(maxsize=1)
def native_search():
    source = Path(__file__).parent / 'vendor/mahjong/platform/discard_search_native.cpp'
    digest = hashlib.sha256(b'exports-v2' + source.read_bytes() + (source.parent.parent / 'rule_ai/search_native.cpp').read_bytes()).hexdigest()[:16]
    directory = Path(__file__).parent / '.native'
    directory.mkdir(exist_ok=True)
    target = directory / (digest + ('.dll' if os.name == 'nt' else '.so'))
    if not target.exists():
        compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            raise RuntimeError('开启 discard_postprocess 需要 C++17 编译器：g++ 或 clang++')
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            output = Path(temporary) / target.name
            command = [compiler, '-O3', '-std=c++17', '-shared']
            if os.name == 'nt':
                target_triple = subprocess.check_output([compiler, '-dumpmachine'], text=True).strip()
                if 'msvc' in target_triple:
                    command += ['-Wl,/EXPORT:' + name for name in ('rb_shape_one', 'rb_value', 'rb_missing', 'rb_discard')]
                else:
                    command += ['-Wl,--export-all-symbols']
            else:
                command += ['-fPIC', '-pthread']
            subprocess.run(command + [str(source), '-o', str(output)], check=True, capture_output=True)
            os.replace(output, target)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.rb_shape_one.argtypes = [ptr, ptr] + [ctypes.c_int]*4 + [ctypes.POINTER(ctypes.c_double), ptr]
    lib.rb_shape_one.restype = None
    lib.rb_value.argtypes = [ptr, ptr] + [ctypes.c_int]*4 + [ctypes.c_double]*3 + [ctypes.c_int]*2 + [ctypes.POINTER(ctypes.c_double)]
    lib.rb_value.restype = None
    lib.rb_missing.argtypes = [ptr, ctypes.c_int]
    lib.rb_missing.restype = ctypes.c_int
    lib.rb_discard.argtypes = [ptr, ptr, ctypes.c_int, ptr]
    lib.rb_discard.restype = ctypes.c_int
    return lib

