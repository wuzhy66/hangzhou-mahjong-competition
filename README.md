# 麻将 ONNX 参赛客户端

## 环境依赖

- **Windows**：Git for Windows（使用 Git Bash）和 Windows PowerShell，运行原生 Windows Python。
- **Linux**：Bash、Git，以及 `curl` 或 `wget`。
- **Python 3.12**：`setup.sh` 自动安装 uv、Python 和依赖，默认 ONNX 模式无需预装 Python、PyTorch、CUDA 或编译器。可选后处理需要 C++17 编译器（Linux 的 g++，Windows 的 clang++ 及 MSVC Build Tools），在启动前编译。
- 首次安装依赖需要联网；本仓库只提交源码，ONNX 模型需另行放置并通过 onnx 配置指定路径。模型、私有配置和运行日志均被 Git 忽略。

Python 依赖由 `requirements.txt` 固定版本，自动安装：

| 库 | 版本 | 用途 |
|---|---|---|
| numpy | 2.5.3 | 观测与数组处理 |
| onnxruntime | 1.26.0 | CPU ONNX 推理 |
| filelock | 4.0.1 | 防止重复启动 |
| flatbuffers | 25.12.19 | ONNX Runtime 依赖 |
| packaging | 26.3 | ONNX Runtime 依赖 |
| protobuf | 7.36.2 | ONNX Runtime 依赖 |

HTTP/SSE、TLS 和线程调度使用 Python 标准库。推理核心保留 WSL 原有 package 结构，位于 `vendor/`，来源文件 SHA256 见 `vendor/SOURCES.json`；Windows native 编译、加载适配集中在 `native_runtime.py`。

## 配置

编辑仓库中的 `config.json`：

```json
{
  "token": "填写参赛令牌",
  "onnx": "bc-mortal-v4-r1-13.onnx",
  "mode": "auto",
  "server": "https://10.240.169.190:18080"
}
```

| 字段 | 说明 |
|---|---|
| token | 参赛令牌，运行前必填；提交 Git 时留空 |
| onnx | ONNX 文件名或路径，相对路径以仓库目录为基准 |
| mode | `auto` 自动识别、`ranking` 打榜、`tournament` 比赛 |
| server | 官方 API 基础地址，不加 `/portal` 或 `/api` 路径 |

`auto` 下，全局 token 默认打榜，绑定比赛的 token 自动参加对应比赛。全局 token 参赛时，另加 `"tournament_id": "官方比赛ID"`，或使用下方命令参数。

可选字段无需填写 `null`，直接省略即可。默认单次 ONNX 推理使用 1 个 CPU 线程、日志保存在 `runs/client`、启用 TLS 验证。模型须兼容本客户端的 6122 维观测和 74 维动作布局。模型文件不随源码提交，可自行提供 `bc-mortal-v4-r1-13.onnx`、`bc-mortal-v4-0-1.onnx` 或 `bc-mortal-v4-r1-6.onnx`；选择 `bc-mortal-v4-0-1.onnx` 并沿用其原部署线程配置时，可增加 `"threads": 8`。

可选 `"discard_postprocess": true` 开启 `shape_lookahead` 弃牌后处理（Top-k=4、Top-p=0.9）；默认关闭。腾蛇当前使用 r1-6、后处理开启、`"threads": 1`。

## 运行

Windows 在 **Git Bash** 中执行，Linux 在 Bash 中执行：

```bash
cd mahjong-competition
bash setup.sh
# 编辑 config.json，填写 token
bash run.sh
```

`setup.sh` 不覆盖已有配置。打榜持续匹配；比赛自动报名、确认出席并处理对局，整场结束后退出。按 `Ctrl+C` 停止，同一账号不要在其他机器重复运行。

```bash
# 全局 token 参加指定比赛
bash run.sh --tournament-id 官方比赛ID

# 离线检查模型并预热
bash run.sh --check

# 检查平台身份，不报名、不匹配、不提交动作
bash run.sh --check-platform
```

当前默认地址为内网平台，需要对应网络/VPN 权限。使用私有 CA 时提供官方证书；仅在明确接受不验证 TLS 时使用 `--insecure`。这些参数可与比赛或检查参数组合：

```bash
bash run.sh --ca-file /path/to/官方CA.pem
# 或显式关闭证书验证
bash run.sh --insecure
```

按实际桌数分配 game worker，打榜最多 10 个、比赛最多 16 个，每桌独立接收 SSE 通知。网络或平台延迟仍可能导致超时，无法保证代打率为零。
