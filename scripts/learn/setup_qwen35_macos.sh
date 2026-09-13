#!/usr/bin/env bash
# 为 Qwen3.5-0.8B 的 Apple Silicon 文本调试示例创建独立的 MLX 环境。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv/qwen35-mac"
STAGE_DIR="${VENV_DIR}/build-package"
PACKAGE_DIR="${STAGE_DIR}/python"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "缺少命令：${command_name}" >&2
        exit 1
    fi
}

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "这个示例只支持 Apple Silicon Mac（Darwin/arm64）。" >&2
    exit 1
fi

require_command uv
require_command ditto
require_command "${PYTHON_BIN}"

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
    echo "需要 Python 3.12，当前 ${PYTHON_BIN} 是 Python ${PYTHON_VERSION}。" >&2
    echo "可通过 PYTHON_BIN=/path/to/python3.12 指定解释器。" >&2
    exit 1
fi

echo "[1/4] 创建独立虚拟环境：${VENV_DIR}"
uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"

# 当前 checkout 的 pyproject.toml 面向 CUDA。复制一份仅用于打包安装的源码，
# 再替换为 Apple 平台依赖清单；不会改动工作区内受 Git 跟踪的配置文件。
echo "[2/4] 准备 Apple MLX 依赖安装副本"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
ditto "${REPO_ROOT}/python" "${PACKAGE_DIR}"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${PACKAGE_DIR}/pyproject.toml"

# 纯文本 MLX 示例不需要 Rust 扩展；运行时由 PYTHONPATH 指向当前 checkout 源码。
echo "[3/4] 安装 srt_mps 依赖（首次运行会下载 Python 包）"
SGLANG_BUILD_RUST_EXTS=none uv pip install \
    --python "${VENV_DIR}/bin/python" \
    "${PACKAGE_DIR}[srt_mps]"
rm -rf "${STAGE_DIR}"

# 这里同时确认当前分支源码会被导入，以及 Torch MPS 和 MLX Metal 均可用。
echo "[4/4] 验证当前源码、Torch MPS 与 MLX Metal"
SGLANG_USE_MLX=1 \
SGLANG_DEMO_REPO_ROOT="${REPO_ROOT}" \
PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
"${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path

import mlx.core as mx
import sglang
import torch
from sglang.srt.hardware_backend.mlx.runtime import use_mlx

repo_python = (Path(os.environ["SGLANG_DEMO_REPO_ROOT"]) / "python").resolve()
sglang_path = Path(sglang.__file__).resolve()
if repo_python not in sglang_path.parents:
    raise SystemExit(
        f"当前导入的不是 checkout 源码：{sglang_path}（期望位于 {repo_python}）"
    )
if not use_mlx():
    raise SystemExit("SGLANG_USE_MLX=1 未启用 MLX 后端")

print(f"SGLang source: {sglang_path}")
print(f"Torch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
print(f"MLX: {mx.__version__}, Metal: {mx.metal.is_available()}")
PY

echo
echo "环境已准备好。启动服务："
echo "  ./scripts/learn/run_qwen35_macos.sh"
