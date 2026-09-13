#!/usr/bin/env bash
# 使用官方 Qwen/Qwen3.5-0.8B 启动本地单请求、顺序调度的 MLX 服务。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/qwen35-mac/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "未找到调试环境，请先运行：./scripts/learn/setup_qwen35_macos.sh" >&2
    exit 1
fi

# 使用当前 checkout 源码；依赖包来自 setup 脚本创建的隔离虚拟环境。
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_USE_MLX=1

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-0.8B \
    --host 127.0.0.1 \
    --port 30000 \
    --max-running-requests 1 \
    --max-total-tokens 2048 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled
