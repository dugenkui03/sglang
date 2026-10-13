#!/usr/bin/env python3
"""NOTE 安装经过本机测试的 OmniVoice；只下载处理器版运行库，不修改系统包。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setup_audio8 import check_system, run

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/k2-fsa/OmniVoice.git"
SOURCE_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
MODEL_ID = "k2-fsa/OmniVoice"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
DEFAULT_VOICE = {"instruct": "male, middle-aged, low pitch", "num_step": 16, "speed": 0.95}
MODEL_FILES = [
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors", "audio_tokenizer/preprocessor_config.json",
]


def write_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".setup-", suffix=".json", dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_runtime(runtime: Path, uv: str) -> None:
    # 【Step 1】独立检出固定源码版本，拒绝覆盖本地修改。
    if not runtime.exists():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth=1", SOURCE_URL, str(runtime))
    if not (runtime / ".git").exists():
        raise RuntimeError(f"目标不是独立的源码仓库，请更换 --runtime-root：{runtime}")
    dirty = subprocess.check_output(
        ["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"源码有本地修改，安装器不会覆盖：{runtime}")
    run("git", "fetch", "--depth=1", "origin", SOURCE_REVISION, cwd=runtime)
    run("git", "checkout", "--detach", SOURCE_REVISION, cwd=runtime)
    python = runtime / ".venv/bin/python"
    if not python.exists():
        run(uv, "venv", "--python", "3.11", str(runtime / ".venv"))
    # 【Step 2】显式使用处理器包，避免上游默认设置下载显卡运行库。
    run(uv, "pip", "install", "--python", str(python),
        "torch==2.8.0+cpu", "torchaudio==2.8.0+cpu",
        "--index-url", "https://download.pytorch.org/whl/cpu")
    run(uv, "pip", "install", "--python", str(python), "--no-deps", "-e", str(runtime))
    run(uv, "pip", "install", "--python", str(python),
        "transformers==5.3.0", "accelerate", "pydub", "gradio", "tensorboardX",
        "webdataset", "numpy", "soundfile", "librosa", "Pillow>=10,<13")
    # 【Step 3】主模型、文字分词器和音频编解码器一并下载，以后推理可离线。
    hf = shutil.which("hf")
    command = [hf] if hf else [uv, "tool", "run", "--from", "huggingface_hub", "hf"]
    run(*command, "download", MODEL_ID, "--revision", MODEL_REVISION,
        "--local-dir", str(runtime / "model"))


def validate_runtime(runtime: Path) -> Path:
    for relative in ["omnivoice/models/omnivoice.py", *["model/" + name for name in MODEL_FILES]]:
        path = runtime / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"运行环境不完整，缺少：{path}")
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"源码版本与已验证版本不同：{revision}；请使用默认安装流程建立固定版本环境")
    python = runtime / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"独立解释器不存在：{python}")
    run(str(python), "-c",
        "import torch, torchaudio, transformers, soundfile, PIL; "
        "from omnivoice import OmniVoice; "
        "assert torch.version.cuda is None, '需要处理器版 PyTorch'; "
        "print('依赖检查通过：', torch.__version__, transformers.__version__)")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-runtime", type=Path, help="复用已安装的固定版本源码根目录")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / ".runtime/omnivoice")
    parser.add_argument("--font", type=Path, help="中文字体文件")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须是正整数")
    try:
        font = check_system(args.font)
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("缺少 uv，请按 https://docs.astral.sh/uv/getting-started/installation/ 安装")
        if not shutil.which("git"):
            raise RuntimeError("缺少 git，请先安装：sudo apt-get install git")
        runtime = (args.reuse_runtime or args.runtime_root).expanduser().resolve()
        if args.reuse_runtime is None:
            install_runtime(runtime, uv)
        python = validate_runtime(runtime)
        settings = runtime / "quick_summary_voice.json"
        if not settings.exists():
            write_json(settings, DEFAULT_VOICE)
        write_json(runtime / "runtime-info.json", {
            "backend": "omnivoice", "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION, "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "torch": "2.8.0+cpu",
            "transformers": "5.3.0", "device": "cpu", "dtype": "float32",
        })
        write_json(ROOT / "config.local.json", {
            "backend": "omnivoice", "runtime_dir": str(runtime),
            "python": str(python), "font": str(font), "threads": args.threads,
        })
        print(f"已写入本机配置：{ROOT / 'config.local.json'}")
        print(f"声音配置：{settings}；更改后自动使配音缓存失效。")
        print("请先生成短句试听。自动波形检查不能代替对发音、漏句、重复和情感的人工验收。")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"安装失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
