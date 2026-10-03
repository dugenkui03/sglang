#!/usr/bin/env python3
"""NOTE 建立独立 Audio8 运行环境，或复用现有环境；不修改系统包。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/Edge0-AI/Audio8_TTS.git"
SOURCE_REVISION = "07e40f5d0b03fc473635ef378654bfb581027ac3"
MODEL_ID = "Edge0/audio8-TTS-0.1B-ONNX-INT8"
MODEL_REVISION = "317c12d4e0da83847b594fcf8bd74bf2c76615ec"
RUNTIME_SUBDIR = "onnx_runtime_0_1b_int8"
MODEL_FILES = [
    "runtime_manifest.json", "config.json", "README.md", "reference_codes.npy",
    "tokenizer/tokenizer.json", "slow_ar_int8.onnx", "slow_ar_int8.onnx.data",
    "fast_ar_int8.onnx", "fast_ar_int8.onnx.data",
    "codec_decoder_fp16.onnx", "codec_decoder_fp16.onnx.data",
]


def run(*args: str, cwd: Path | None = None) -> None:
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def check_system(font: Path | None) -> Path:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError(
            f"缺少 {', '.join(missing)}。请先安装：sudo apt-get install ffmpeg"
        )
    if font is not None:
        if not font.is_file():
            raise FileNotFoundError(f"指定字体不存在：{font}")
        return font.resolve()
    for candidate in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
    ):
        if candidate.is_file():
            return candidate
    if shutil.which("fc-match"):
        candidate = Path(subprocess.check_output(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
        ).strip())
        if candidate.is_file() and "CJK" in candidate.name:
            return candidate.resolve()
    raise RuntimeError(
        "缺少中文字体。请安装 sudo apt-get install fonts-noto-cjk，"
        "或者用 --font 指向已安装的中文字体。"
    )


def install_runtime(destination: Path, uv: str) -> Path:
    checkout = destination.resolve()
    # 【Step 1】使用独立目录和固定版本，避免污染学习框架的依赖。
    created = not checkout.exists()
    if created:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_URL, str(checkout))
    if not (checkout / ".git").exists():
        raise RuntimeError(f"目标目录已存在且不是预期的仓库，请换 --runtime-root：{checkout}")
    if not created:
        dirty = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"Audio8 源码存在修改，安装器不会覆盖：{checkout}")
    run("git", "sparse-checkout", "init", "--cone", cwd=checkout)
    run("git", "sparse-checkout", "set", RUNTIME_SUBDIR, cwd=checkout)
    run("git", "fetch", "--depth=1", "origin", SOURCE_REVISION, cwd=checkout)
    run("git", "checkout", "--detach", SOURCE_REVISION, cwd=checkout)
    runtime = checkout / RUNTIME_SUBDIR
    python = runtime / ".venv/bin/python"
    if not python.exists():
        run(uv, "venv", "--python", "3.11", str(runtime / ".venv"))
    run(uv, "pip", "install", "--python", str(python), "-r", str(runtime / "requirements.txt"), "Pillow>=10,<13")
    # 【Step 2】只下载推理文件；默认声音使用随模型提供的参考声码。
    hf = shutil.which("hf")
    command = [hf] if hf else [uv, "tool", "run", "--from", "huggingface_hub", "hf"]
    run(*command, "download", MODEL_ID, *MODEL_FILES,
        "--revision", MODEL_REVISION, "--local-dir", str(runtime / "model"))
    return runtime


def validate_runtime(runtime: Path, uv: str) -> Path:
    python = runtime / ".venv/bin/python"
    required = ["arktts_runtime/runtime.py", "scripts/register_default_voice.py"]
    required += ["model/" + name for name in MODEL_FILES]
    for relative in required:
        path = runtime / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"运行环境不完整，缺少文件：{path}")
    if not python.is_file():
        raise FileNotFoundError(f"找不到独立环境解释器：{python}")
    # 【Step 3】复用现有环境时仍保证视频制图所需依赖可用。
    run(uv, "pip", "install", "--python", str(python), "Pillow>=10,<13")
    run(str(python), "-c", "import numpy, onnxruntime, tokenizers, soundfile, PIL; print('运行依赖检查通过')")
    if not (runtime / "voices/default/meta.json").is_file():
        run(str(python), str(runtime / "scripts/register_default_voice.py"))
    elif not (runtime / "voices/default/codes.npy").is_file():
        raise RuntimeError("默认声音记录不完整，请检查 voices/default 目录后重试")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-runtime", type=Path, help="复用含 model、voices、.venv 的官方运行目录")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / ".runtime/audio8")
    parser.add_argument("--font", type=Path, help="中文字体文件")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须是正整数")
    try:
        font = check_system(args.font)
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("缺少 uv，请先按 https://docs.astral.sh/uv/getting-started/installation/ 安装")
        if not shutil.which("git"):
            raise RuntimeError("缺少 git，请先安装：sudo apt-get install git")
        runtime = (args.reuse_runtime.expanduser().resolve() if args.reuse_runtime
                   else install_runtime(args.runtime_root, uv))
        python = validate_runtime(runtime, uv)
        config = {
            "runtime_dir": str(runtime), "python": str(python),
            "font": str(font), "threads": args.threads,
        }
        config_path = ROOT / "config.local.json"
        handle, temporary = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=ROOT)
        os.close(handle)
        try:
            Path(temporary).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, config_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"已写入本机配置：{config_path}")
        print("安装完成不代表音质验收。请先用短句配音，人工核对漏句、重复、发音，再生成长视频。")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"安装失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
