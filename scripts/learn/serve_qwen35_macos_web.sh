#!/usr/bin/env bash
# 启动一个静态网页，用于向本机 Qwen3.5 SGLang 服务发送调试请求。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/qwen35-mac/bin/python"
PAGE_DIR="${REPO_ROOT}/scripts/learn"
PORT="${QWEN35_WEB_PORT:-30001}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "未找到调试环境，请先运行：./scripts/learn/setup_qwen35_macos.sh" >&2
    exit 1
fi

echo "调试页面地址：http://127.0.0.1:${PORT}/qwen35_macos_web.html"
echo "请保持此终端运行；按 Ctrl+C 停止网页服务。"

exec "${PYTHON_BIN}" -m http.server "${PORT}" \
    --bind 127.0.0.1 \
    --directory "${PAGE_DIR}"
