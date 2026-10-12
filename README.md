# 麻将 ONNX 参赛客户端
## 使用程序参赛的说明

从官方平台取得参赛 token 与比赛 ID，确认 `server` 填写官方 API 基础地址（不带 `/portal` 或 `/api`），将 token 写入本地 `config.json` 并放好 ONNX 模型。运行 `bash setup.sh`、`bash run.sh --check-platform` 核对模型和账号后，用 `bash run.sh --tournament-id 官方比赛ID` 参赛；如果 token 已绑定比赛，保持 `"mode": "auto"` 并直接运行 `bash run.sh`。客户端会自动报名、确认出席和提交动作。

## 环境依赖

- **Windows**：Git for Windows（使用 Git Bash）和 Windows PowerShell，运行原生 Windows Python。
- **Linux**：Bash、Git，以及 `curl` 或 `wget`。
- **Python 3.12**：`setup.sh` 自动安装 uv、Python 和依赖，默认安装 CPU ONNX Runtime，无需预装 Python、PyTorch 或 CUDA；`bash setup.sh --cuda` 可另装独立 GPU 环境（需要 NVIDIA GPU 和驱动）。可选后处理需要 C++17 编译器（Linux 的 g++，Windows 的 clang++ 及 MSVC Build Tools），在启动前编译。
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
  "onnx": "bc-mortal-v4-r1-17.onnx",
  "inference_mode": "probability_ensemble",
  "discard_postprocess": false,
  "ensemble_onnx": ["bc-mortal-v4-r1-17.onnx", "bc-mortal-v4-r1-12.onnx", "bc-mortal-v4-r1-14.onnx"],
  "mode": "auto",
  "server": "https://10.240.169.190:18080"
}
```

| 字段 | 说明 |
|---|---|
| token | 参赛令牌，运行前必填；提交 Git 时留空 |
| onnx | ONNX 文件名或路径，相对路径以仓库目录为基准；单模型默认 r1-17 |
| inference_mode | 默认 `probability_ensemble`，启用 r1-17、r1-12、r1-14 三模型概率集成；显式 `single` 使用 `onnx` 指定的单模型 |
| ensemble_onnx | 概率集成的三个不同 ONNX 路径，默认 r1-17、r1-12、r1-14 |
| ensemble_sha256 | 可选，按 `ensemble_onnx` 顺序提供三个预期 SHA256，启动不匹配即报错 |
| ensemble_workers | 可选，全客户端推理线程池容量，范围 3–48；CUDA 默认 3；CPU 按可用 CPU 数确定，至少 3、最多 48 |
| device | 省略时按模式选择：单模型用 `cpu`，概率集成用 `cuda`。显式配置或 `--device` 优先；CUDA 需要独立 GPU Runtime |
| gpu_memory_mb | CUDA 每个模型的 ORT arena 上限，默认 1024 MiB；不包含驱动/context 等额外显存 |
| gpu_batch_size | CUDA 集成跨桌合批上限，默认 16，范围 1–16；模型须支持动态 batch |
| gpu_batch_wait_ms | 主动合批窗口，默认 20 ms，允许 0–20 ms；从最早请求入队时计算 |
| mode | `auto` 自动识别、`ranking` 打榜、`tournament` 比赛 |
| server | 官方 API 基础地址，不加 `/portal` 或 `/api` 路径 |

`auto` 下，全局 token 默认打榜，绑定比赛的 token 自动参加对应比赛。全局 token 参赛时，另加 `"tournament_id": "官方比赛ID"`，或使用下方命令参数。

可选字段无需填写 `null`，直接省略即可。默认单次 ONNX 推理使用 1 个 CPU 线程、日志保存在 `runs/client`、启用 TLS 验证。模型须兼容本客户端的 6122 维观测和 74 维动作布局。模型文件不随源码提交；默认集成需提供 r1-17、r1-12、r1-14 三份完整 ONNX 文件。也可自行指定其他兼容模型；选择 `bc-mortal-v4-0-1.onnx` 并沿用其原部署线程配置时，可增加 `"threads": 8`。

可选 `"discard_postprocess": true` 开启 `shape_lookahead` 弃牌后处理（Top-k=4、Top-p=0.9）；默认关闭。

## 运行

概率集成将三个模型对同一 observation 的**合法动作 softmax 概率等权相加，按 Round v8 转为 float32 后取 argmax**，覆盖所有动作类型。不执行额外弃牌搜索，因此集成模式必须关闭 `discard_postprocess`。模型名单不会自动跟随评测排行榜变化，调整时需显式配置或热更。

三个 ONNX Session 在启动时加载和预热，每个决策提交三路并行推理。所有对局共享常驻的有界线程池；CPU 模式下容量默认按可用 CPU 数确定（3–48），也可通过 `ensemble_workers` 调整。`threads` 是每个模型的 ORT 线程数，默认 1，避免模型内部线程造成过度争抢。CPU 多桌时将线程池固定为 3 会限制吞吐，应保留多桌的并行容量。并行能缩短有空闲 CPU 核时的等待，实际加速取决于硬件及对局并发，不保证等于单模型耗时；计算总量和模型内存仍约为三份。`active_model.json` 和启动日志记录三份模型 SHA、集成算法和线程配置。

模型权重按当前仓库规则留在本地；新环境需提供三个完整 ONNX 文件，缺失或 SHA 不匹配时拒绝启动。`setup.sh` 按配置检查单模型或全部三模型；已有 LFS 指针会按需下载。

```bash
# 默认三模型概率集成 + CUDA，不带后处理
bash setup.sh --cuda
# 离线检查，不发起对局
bash run.sh --check
# 使用单模型
bash run.sh --inference-mode single --onnx bc-mortal-v4-r1-17.onnx
# 显式替换三个集成模型
bash run.sh --inference-mode probability_ensemble --ensemble-onnx a.onnx b.onnx c.onnx
```

离线验证与基准（使用运行客户端的 Python 环境）：

```bash
python -m unittest discover -s tests -v
python benchmark_ensemble.py --observations /path/to/verification_observations.npz
```

基准会先核对真实局面的串行/并行概率和完全一致，再测量 1 桌和 10 桌下的延迟与吞吐，结果写入 `runs/ensemble-benchmark.json`。

GPU 模式使用同一份 ONNX 和 FP32 推理，关闭 TF32 与 CUDA Graph；默认全客户端 3 个推理 worker，另有一个跨桌合批调度线程。多桌 observation 合成同一个 batch，三个模型各推理整批，然后按原请求顺序回送每桌的概率和。每行独立使用自己的合法动作 mask。

默认最大 batch=16，主动合批窗口默认 20 ms，可通过 `--gpu-batch-wait-ms 1` 改为 1 ms：满批立即提交；不满时按最早请求的入队时刻设截止时间；若 GPU 忙时已消耗掉配置的窗口，轮到该请求时不再额外等待。**配置窗口是主动凑批的时间预算，不包含 GPU 繁忙造成的排队、推理本身和 OS 调度延迟。** 队列最多容纳 256 行，满时明确报错；停止时排空已接受请求并回收线程。启动时预热 1–16 的 batch 大小。

CUDA 初始化或执行异常直接报错，不自动切换 CPU。合批执行失败会通知所有等待方并拒绝后续请求，避免挂起或复用异常 Session。它仍需少量 CPU 做 observation、softmax 和平台通信；GPU/CPU 浮点结果不保证逐位相同，需要检查最终动作与并发稳定性。

可另建虚拟环境安装 `requirements-cuda.txt`，不要在同一环境中同时安装 `onnxruntime` 和 `onnxruntime-gpu`。以下示例使用 Linux/WSL，Windows 对应虚拟环境的 `Scripts/python.exe`：

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install -r requirements-cuda.txt
.venv-cuda/bin/python run.py --device cuda --inference-mode probability_ensemble --check
# 本机验证通过后，再用同一环境运行客户端
.venv-cuda/bin/python run.py --device cuda --inference-mode probability_ensemble
# 显式指定本次验证的合批参数
.venv-cuda/bin/python run.py --device cuda --inference-mode probability_ensemble --gpu-batch-size 16 --gpu-batch-wait-ms 20
# 离线检查 batch=1–16、跨桌分发和持续并发；不连接平台
.venv-cuda/bin/python verify_gpu_batching.py --observations /path/to/verification_observations.npz
```

`setup.sh` 默认安装 CPU 环境；`setup.sh --cuda` 另装 GPU 环境。`run.sh` 根据配置和 CLI 选择环境：默认三模型概率集成 + CUDA + 20 ms、batch=16；`--inference-mode single` 显式选择单模型 + CPU。也可用 `--device cpu` 显式让集成跑 CPU。不要在默认配置中写死 device，否则它会覆盖模式的默认设备。

Windows 在 **Git Bash** 中执行，Linux 在 Bash 中执行：

```bash
cd mahjong-competition
bash setup.sh
# 编辑 config.json，填写 token
bash run.sh
```

`setup.sh` 不覆盖已有配置。打榜持续匹配；比赛自动报名、确认出席并处理对局，整场结束后退出。按 `Ctrl+C` 停止，同一账号不要在其他机器重复运行。

打榜匹配遇到连接中断、429 或可恢复的 5xx 时，先每 5 秒查询本人 `active_games`：唯一可识别房间会自动接管；连续六次明确空响应（至少 30 秒）后才重新匹配。查询失败、字段缺失或房间归属有歧义会继续等待并清零空响应计数；401/403 或身份变化仍停止。30 秒是应对索引延迟的恢复宽限，不是服务端幂等保证。

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

## 运行中切回单模型 CPU

打榜（ranking）和比赛（tournament）均支持热切换。客户端必须以包含此功能的版本启动；已运行的旧版本需下次启动后才能使用。

发现概率集成异常时，在另一个终端执行（不需要 GPU Runtime）：

```bash
python switch_to_cpu.py --output runs/client
```

`--output` 必须与运行客户端的输出目录一致。工具读取当前 run_id，提交本地请求并等待确认；只有收到 applied 才报告成功。默认每 0.5 秒检查请求，再加载、SHA 校验并预热启动配置中 `onnx` 指定的原单模型（默认 r1-17）。加载期间正常对局继续；新决策原子切到 CPU，已经开始的决策继续持有旧模型，完成后才释放旧模型资源。回退模型 SHA 在客户端启动时固定。

切换失败保留原模型，错误写入 model-control-status.json；成功同步 active_model.json 和 model_hot_switch 日志。不会改 token、比赛/房间或匹配进程，也不改变 config.json，因此重启仍按配置选择模式。显式请求热切换，不会因效果波动或未经确认的异常自动切换。CPU 模型加载时间与已卡住的 CUDA 调用不受 0.5 秒检查周期保证。


## historical297 · argmax

WSL 新增独立启动入口，沿用 `config.json` 中的平台、身份和输出设置：

```bash
bash setup-historical297.sh                   # 新机器首次安装独立环境
bash run-historical297.sh --check              # 仅离线加载校验，不连接平台
bash run-historical297.sh --mode ranking       # 使用现有身份打榜
# 无 CUDA 时可以显式加 --device cpu
```

启动前先正常退出使用同一身份的旧客户端；原有唯一身份锁仍然生效。
此入口使用客户端目录内 `.venv-historical297` 的 PyTorch，默认 CUDA。也可用 `HISTORICAL297_PYTHON`
指定兼容环境。通用 Python 入口支持 `--inference-mode historical297` 和
`--historical297-model PATH`；模型路径默认 `historical297.pt`。
固定模型 SHA256 为 `90799eafbb2a11a3eb586fcec6f945411a1bc4315b8cd329c1340a5732649453`。

此策略使用冻结公开模型的7113维观测、41动作合法 mask 和原始 argmax，关闭 TF32，
不使用 shape-lookahead。自杠和弃牌共享动作编号时按模型原规则优先自杠。
客户端复用已对账的公开快照和历史，保存实际7113维模型输入，模型源码独立放在
`historical297_source`，不替换原有6122维/74动作策略。
`discard_postprocess` 必须为 false。运行时原有“回退 CPU”控制仍表示切到配置中的
旧 ONNX 单模型；297的CPU推理可通过下述显式策略热更选择 `device: "cpu"`。

离线检查：固定权重 SHA、CPU/CUDA 模型预热，以及本地历史快照回放脚本
`PYTHONPATH=. .venv-historical297/bin/python tests/check_historical297_snapshots.py`。

模型权重已包含在客户端根目录；单独复制本目录后，执行上述安装命令即可，不依赖上级 rlai-car 项目。Python 虚拟环境不保证跨机器可搬迁，新机器请重新安装依赖。

## 三模型与 historical297 双向热更

打榜和比赛共用同一控制监听。新版客户端支持三模型 probability ensemble 与
historical297 · argmax 双向切换，不需要重新匹配或重新接入比赛。未提交请求时不会
加载候选策略，不改变原有模型、推理、线程或出牌路径；没有自动切换策略的逻辑。

从客户端目录执行，`--output` 与目标配置的 `output` 都必须指向本实例：

```bash
python switch_ensemble.py --output runs/client --config switch-target.json
```

切入297的 `switch-target.json`（无需 token）：

```json
{
  "output": "runs/client",
  "inference_mode": "historical297",
  "historical297_model": "historical297.pt",
  "device": "cuda",
  "discard_postprocess": false
}
```

切回三模型使用同一命令，目标配置的 `inference_mode` 为 `probability_ensemble`，
提供 `ensemble_onnx` 三个本地文件名及对应的 `ensemble_sha256`，并设置
`device: "cuda"`、`discard_postprocess: false`。r1-17/r1-12/r1-14 的顺序应与原配置一致。
工具只传递策略和权重选择；线程数、ensemble_workers、batch参数沿用运行实例，
不修改平台身份、房间、比赛或启动配置。只有 `applied` 表示完成，超时不能当作成功。

297热更只接受本地 `historical297.pt`，loader 校验上述固定 SHA 并预热。
加载、校验或预热失败时保留旧策略；加载期间旧策略继续服务，新策略发布后
新决策使用新观测编码，在途决策仍持有原模型至完成。

**运行前提**：当前进程必须已经用新版代码启动，并且其 Python 环境同时具备
PyTorch 和 ONNX Runtime CUDA 依赖；仅更新磁盘文件不会升级已经运行的进程。
可在后续计划启动时使用 `.venv-historical297/bin/python run.py --config ...` 选择
任一初始策略（不要用强制初始297的 wrapper 来启动三模型）。只有 ONNX 环境而缺少
PyTorch 的实例会拒绝297切换并保留旧策略，不会自动安装依赖或重启。
本地控制文件仍使用 run_id 隔离旧实例请求，已有 CPU 回退及三模型热更命令保持兼容。

### 双策略 CUDA 常驻快速切换（显式启用）

```bash
bash setup-resident.sh    # 新建独立 .venv-resident，已有环境拒绝覆盖
bash run-resident.sh --config config.json --check
bash run-resident.sh --config config.json
```

初始策略仍由配置选择，打榜和比赛均适用。也可在兼容环境中使用
`python run.py --resident-strategies --device cuda --config config.json`。
默认 `resident_strategies: false`，未启用时保留原 ModelStore、500ms监听及加载路径。
常驻模式仅支持固定 r1-17/r1-12/r1-14 三模型和固定 historical297，启动时全部校验、
加载并预热完成后才连接平台；`active_model.json` 的 `resident_ready: true` 表示准备完成。

仍使用上述 `switch_ensemble.py` 提交目标配置。常驻模式监听间隔为20ms，
收到合法请求后仅写入状态、切换已有模型引用，不加载权重、不运行预热、不卸载旧策略。
两策略一直保留到客户端退出，在途决策可以跨越多次切换而继续使用原模型。
SHA/设备/文件名与常驻目标不符的请求直接拒绝，不会在局中偷偷加载新模型。
初次紧急 CPU 回退仍需加载单模型，不属于此快速切换保证范围。

常驻模式增加显存占用，要求同一环境同时有 PyTorch CUDA 和 ONNX Runtime GPU；
启动准备失败不会接入比赛。请求确认仍以 `applied` 为准，20ms是轮询间隔，
不是最坏端到端延迟或1秒出牌窗口的硬保证。更新代码后需在安全边界重启实例一次；
之后双策略间切换不需要重启、不改变平台身份或房间。
