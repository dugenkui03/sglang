#!/usr/bin/env bash
# Run on a GPU Worker with shared model weights and a branch-compatible environment.
# 遇到命令失败、未定义变量或管道中任一命令失败时立即退出脚本。
set -euo pipefail

# 当前分支使用独立的 Worker 本地 Python 环境，可通过环境变量覆盖路径。
SGLANG_PYTHON_ENV="${SGLANG_PYTHON_ENV:-/tmp/sglang-learn-debug}"

# 让 Python 优先从当前调试工作树加载 SGLang 源码。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/python"
# 服务及其子进程也使用带文件名、行号和方法名的日志格式。
export SGLANG_LOGGING_CONFIG_PATH="$REPO_ROOT/log_config_debug.json"
# 禁用 Python 标准输出缓冲，使服务日志立即写到终端或日志文件。
export PYTHONUNBUFFERED=1
# 指定 SGLang 及其第三方组件在 Worker 本地存放缓存的根目录。
export SGLANG_CACHE_DIR=/tmp/sglang-qwen38-jit-cache
# 指定 SGLang 即时编译(JIT)产物在 Worker 本地的缓存目录。
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-qwen38-jit-cache/jit
# 指定 Python 环境中安装的 CUDA 13 工具链根目录。
export CUDA_HOME=/home/tiger/workspace/envs/sglang-qwen38-debug/lib/python3.12/site-packages/nvidia/cu13
# 让 Shell 优先找到调试环境中的 Python 和 CUDA 13 命令，再查找系统命令。
export PATH="$SGLANG_PYTHON_ENV/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# 打印当前 Python 实际加载的 SGLang 源码路径，避免误用虚拟环境中的旧版本。
"$SGLANG_PYTHON_ENV/bin/python" -c \
    'import importlib.util; print("SGLang source:", importlib.util.find_spec("sglang").origin, flush=True)'

# Small, single-request setup for stepping through the Python inference path.
# Original BF16 weights; no LoRA, quantization, or speculative decoding.
# 经由 CLI main() 启动，打印入口日志后再分派到 LLM 服务。
exec "$SGLANG_PYTHON_ENV/bin/python" -c 'from sglang.cli.main import main; main()' serve \
    --model-path /home/tiger/models/Qwen3.8-27B \
    --weight-loader-prefetch-checkpoints \
    --served-model-name qwen38-27b \
    --host 127.0.0.1 \
    --port 30000 \
    --tp-size 1 \
    --dtype bfloat16 \
    --context-length 112048 \
    --max-running-requests 1 \
    --max-total-tokens 4096 \
    --chunked-prefill-size 512 \
    --mem-fraction-static 0.85 \
    --mamba-ssm-dtype float32 \
    --max-mamba-cache-size 5 \
    --disable-prefill-cuda-graph \
    --disable-decode-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-cache-report \
    --log-requests \
    --log-requests-level 2 \
    --log-requests-target stdout \
    --show-time-cost
