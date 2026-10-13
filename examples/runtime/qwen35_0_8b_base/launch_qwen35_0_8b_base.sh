#!/usr/bin/env bash
# 独立的 Qwen3.5-0.8B-Base 调试脚本；在 GPU Worker 上手动执行。
set -euo pipefail

# 复用已有 Python 环境，不安装或修改其中的依赖。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SGLANG_PYTHON_ENV="${SGLANG_PYTHON_ENV:-$(dirname -- "$REPO_ROOT")/envs/sglang-qwen38-debug}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-0.8B-Base}"
export PYTHONPATH="$REPO_ROOT/python"
export PYTHONUNBUFFERED=1
export SGLANG_LOGGING_CONFIG_PATH="$REPO_ROOT/log_config_debug.json"

# 使用单独的即时编译(JIT)缓存目录。
export SGLANG_CACHE_DIR=/tmp/sglang-qwen35-0_8b-base-jit-cache
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
export CUDA_HOME="$SGLANG_PYTHON_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$SGLANG_PYTHON_ENV/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

"$SGLANG_PYTHON_ENV/bin/python" -c \
    'import importlib.util; print("SGLang source:", importlib.util.find_spec("sglang").origin, flush=True)'

# 基础模型(Base Model)以文本续写验证推理，不启用推理内容或工具调用解析器。
# 单卡 BF16、单请求；30001 端口与 27B 脚本的 30000 端口分开。
# 默认静态 GPU 显存(GPU memory)预算为总量的 10%，面向 B200 调试。
exec "$SGLANG_PYTHON_ENV/bin/python" -c 'from sglang.cli.main import main; main()' serve \
    --model-path "$MODEL_PATH" \
    --served-model-name qwen35-0.8b-base \
    --host 127.0.0.1 \
    --port 30001 \
    --tp-size 1 \
    --dtype bfloat16 \
    --context-length 4096 \
    --max-running-requests 1 \
    --max-total-tokens 4096 \
    --chunked-prefill-size 512 \
    --mem-fraction-static 0.1 \
    --mamba-ssm-dtype float32 \
    --max-mamba-cache-size 5 \
    --disable-prefill-cuda-graph \
    --disable-decode-cuda-graph \
    --enable-cache-report \
    --log-requests \
    --log-requests-level 2 \
    --log-requests-target stdout \
    --show-time-cost

# 服务就绪后，可在同一 Worker 的另一个终端测试文本续写：
# curl http://127.0.0.1:30001/v1/completions -H 'Content-Type: application/json' \
#   -d '{"model":"qwen35-0.8b-base","prompt":"春天到了，","max_tokens":32,"temperature":0}'
