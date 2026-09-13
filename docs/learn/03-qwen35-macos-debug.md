# 三、在 Mac 上调试 Qwen3.5-0.8B 的一次文本推理

这个示例使用官方模型 `Qwen/Qwen3.5-0.8B` 在 Apple Silicon Mac 上启动 SGLang，发送一条非流式 `/v1/chat/completions` 请求，并按源码链路跟踪一次推理。

它只关注文本链路：不启用 PD 分离、DFlash、量化或图片/视频输入。

## 1. 准备环境

前提：Apple Silicon、macOS 14 或更新版本、Python 3.12 和 `uv`。当前机器上的环境会创建在被 Git 忽略的 `.venv/qwen35-mac` 中；请至少预留约 6 GB 磁盘空间给 Python 环境、模型权重和 Hugging Face 缓存。

如果尚未安装 Python 3.12 和 `uv`，可使用 Homebrew：

```bash
brew install python@3.12 uv
```

```bash
./scripts/learn/setup_qwen35_macos.sh
```

如果 `python3.12` 不在 `PATH` 中：

```bash
PYTHON_BIN=/opt/homebrew/opt/python@3.12/bin/python3.12 \
  ./scripts/learn/setup_qwen35_macos.sh
```

脚本会从 `python/pyproject_other.toml` 的 `srt_mps` 依赖安装 MLX、MLX-LM 和 Apple 版 PyTorch，但不会修改当前工作区中的 `python/pyproject.toml` 或 `python/pyproject_other.toml`。启动时通过 `PYTHONPATH` 导入当前分支源码，因此在源码中下断点立即生效。

## 2. 启动服务并发送一条请求

在第一个终端启动服务：

```bash
./scripts/learn/run_qwen35_macos.sh
```

首次启动会将官方模型下载到 Hugging Face 本地缓存；之后会复用缓存。服务只监听 `127.0.0.1:30000`。

在第二个终端发送固定请求：

```bash
.venv/qwen35-mac/bin/python scripts/learn/request_qwen35_macos.py
```

脚本会调用 `POST /v1/chat/completions`，使用 `stream=false`、`temperature=0` 和最多 8 个输出 token，并打印完整 OpenAI 格式 JSON。服务终端按 `Ctrl+C` 停止。

`Qwen/Qwen3.5-0.8B` 的配置属于多模态模型，但当前 SGLang 的 `--language-model-only` 不支持这个架构，因此示例不传该参数。请求不携带媒体数据，`_tokenize_one_request` 不会执行多模态数据处理分支。

也可以在第三个终端启动一个本地页面：

```bash
./scripts/learn/serve_qwen35_macos_web.sh
```

浏览器打开 <http://127.0.0.1:30001/qwen35_macos_web.html>，填写提示词后点击“发送请求”。页面直接调用已启动服务的 `/v1/chat/completions`，会同时显示生成文本和完整 JSON 响应；点击“检查服务”会访问 `/health`。

## 3. 为什么这是顺序调试链路

启动参数将 `--max-running-requests` 设为 `1`，使这个示例一次只处理一个请求；`--max-total-tokens 2048` 限制调试时的 KV Cache 池大小。

`--disable-radix-cache` 关闭跨请求复用的 Radix Cache。Qwen3.5 的混合状态在当前 MLX 后端尚不支持该缓存路径；关闭它还能避免缓存逻辑干扰这次单请求调试。

`--disable-overlap-schedule` 关闭 MLX 默认的双 in-flight 图调度，因此 Scheduler 会从 `event_loop_overlap_mlx` 走到 `Scheduler.event_loop_normal`。这不会消除进程边界，只是让 CPU 处理与 MLX 图执行不再交错，适合逐步查看一次请求。

```text
客户端
  -> HTTP /v1/chat/completions
  -> OpenAIServingChat
  -> TokenizerManager：参数整理、文本分词、ZeroMQ 发送
  -> Scheduler 子进程：收请求、组 batch、调用 worker
  -> MLX ModelRunner：Prefill / Decode
  -> Detokenizer 子进程：token IDs 转文本
  -> TokenizerManager / HTTP 响应客户端
```

## 4. 建议断点

| 进程 | 文件与方法 | 看到什么 |
| --- | --- | --- |
| HTTP 主进程 | `python/sglang/srt/entrypoints/http_server.py` 的 `openai_v1_chat_completions` | 收到 OpenAI 协议请求。 |
| HTTP 主进程 | `serving_chat.py` 的 `_handle_non_streaming_request` | Chat 请求转换为 SGLang 内部生成请求。 |
| HTTP 主进程 | `tokenizer_manager.py` 的 `generate_request`、`_tokenize_one_request`、`_dispatch_to_scheduler` | 输入分词并发送已 token 化请求。 |
| Scheduler 子进程 | `scheduler.py` 的 `event_loop_normal`、`handle_generate_request` | 接收请求、创建 Request 并加入调度队列。 |
| Scheduler 子进程 | `hardware_backend/mlx/tp_worker.py` 的 `forward_batch_generation` | Scheduler 将 batch 交给 MLX worker。 |
| Scheduler 子进程 | `hardware_backend/mlx/model_runner.py` 的 `_load_model`、`prefill`、`decode` | MLX-LM 加载模型，并执行 Prefill / Decode。 |

HTTP 服务与 `TokenizerManager` 在主进程；Scheduler 和 Detokenizer 是由 `engine.py` 创建的子进程。若断点设置在 Scheduler 或 MLX 代码，请让 IDE 跟随子进程，或从启动日志中找到 Scheduler PID 后附加调试器。
