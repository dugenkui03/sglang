# 本地 Qwen3.5 调试：服务与进程结构

本地调试示例启动后，有两个 HTTP 服务和四个实际业务进程单元。

| 单元 | HTTP 地址或进程 | 作用 |
| --- | --- | --- |
| SGLang 主进程 | `127.0.0.1:30000` | 对外提供 `/v1/chat/completions`、`/generate`、`/health`；其中包含 HTTP 服务和 `TokenizerManager`。 |
| Scheduler 子进程 | `sglang::scheduler` | 接收已分词请求、组织 batch，并处理 Prefill 和 Decode 调度。 |
| Detokenizer 子进程 | `sglang::detokenizer` | 将模型输出的 token IDs 转回文本。 |
| 调试网页服务 | `127.0.0.1:30001` | 只提供 `qwen35_macos_web.html` 静态页面。 |

Python 还会创建一个 `resource_tracker` 辅助进程，用于管理 multiprocessing 资源；它不参与推理。

当前未启用 PD 分离，因此没有独立的 Prefill 进程和 Decode 进程；两者都由同一个 Scheduler 子进程处理。

## 环境要求

- Apple Silicon Mac（M1/M2/M3/M4）和 macOS 14 或更新版本。
- Python **3.12**；脚本会创建 `.venv/qwen35-mac`，但不会替你安装系统 Python。
- `uv`，用于创建虚拟环境和安装 MLX 依赖。
- 至少预留约 6 GB 磁盘空间，用于 Python 环境、模型权重和 Hugging Face 缓存。

使用 Homebrew 可安装所需工具：

```bash
brew install python@3.12 uv
```

如果 `python3.12` 不在 `PATH` 中，后续初始化时显式指定解释器：

```bash
PYTHON_BIN=/opt/homebrew/opt/python@3.12/bin/python3.12 \
  ./scripts/learn/setup_qwen35_macos.sh
```

```text
浏览器
  │ 访问调试页面
  ▼
静态网页服务 :30001
  │ 页面中的 fetch 请求
  ▼
SGLang 主进程 :30000（HTTP + TokenizerManager）
  │ ZeroMQ
  ▼
Scheduler 子进程（Prefill + Decode）
  │
  ▼
Detokenizer 子进程
  │
  └── 回到主进程 → HTTP 响应给浏览器
```

启动命令：

```bash
# 首次使用：创建 Python 3.12 / MLX 环境
./scripts/learn/setup_qwen35_macos.sh

# 终端 1：模型推理服务
./scripts/learn/run_qwen35_macos.sh

# 终端 2：调试网页
./scripts/learn/serve_qwen35_macos_web.sh
```

浏览器访问 <http://127.0.0.1:30001/qwen35_macos_web.html>，页面会直接向 `127.0.0.1:30000/v1/chat/completions` 发起请求。

## 模型权重下载

模型 `Qwen/Qwen3.5-0.8B` 的权重不提交到 Git。第一次运行 `./scripts/learn/run_qwen35_macos.sh` 时，MLX 会自动从 Hugging Face 下载官方模型到本机缓存；后续启动会复用缓存。

默认缓存位置通常是 `~/.cache/huggingface/hub`。如需在启动服务前预下载，可执行：

```bash
.venv/qwen35-mac/bin/hf download Qwen/Qwen3.5-0.8B
```

如需将缓存放到其他磁盘，在下载和启动前设置 `HF_HOME`：

```bash
export HF_HOME=/path/to/huggingface-cache
./scripts/learn/run_qwen35_macos.sh
```
