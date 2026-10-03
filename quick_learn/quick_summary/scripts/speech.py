"""NOTE 默认使用 OmniVoice 本机配音；Audio8 仅保留作未通过验收的实验后端。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def engine_signature(runtime_dir: Path) -> str:
    """NOTE 不导入模型依赖，返回源码、模型、声音与环境的缓存标识。"""
    root = Path(runtime_dir).expanduser().resolve()
    if not any((root / name).is_file() for name in ("model/config.json", "model/runtime_manifest.json")):
        raise FileNotFoundError(f"语音运行目录缺少模型清单：{root}")
    digest = hashlib.sha256()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "not-a-git-checkout"
    digest.update(revision.encode())
    paths = [Path(__file__).resolve()]
    paths += list((root / "arktts_runtime").glob("*.py"))
    paths += list((root / "omnivoice").rglob("*.py"))
    paths += list(root.glob("runtime-info.json"))
    paths += list(root.glob("quick_summary_voice.json"))
    paths += list((root / "voices/default").glob("*"))
    paths += list((root / "model").rglob("*"))
    paths += list((root / ".venv/lib").glob("python*/site-packages/*.dist-info/METADATA"))
    for path in sorted(set(paths)):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        # 【Step 1】大权重使用大小和修改时间，其余文件计算内容散列。
        name = str(path.relative_to(root)) if path.is_relative_to(root) else "speech.py"
        digest.update(name.encode())
        info = path.stat()
        if info.st_size > 16 * 1024 * 1024:
            digest.update(f"{info.st_size}:{info.st_mtime_ns}".encode())
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


class LocalSpeech:
    """NOTE OmniVoice 模型只加载一次，各段复用；从本机配置读取声音参数。"""

    def __init__(self, runtime_dir: Path, threads: int = 4):
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        if threads < 1:
            raise ValueError("threads 必须是正整数")
        if not (self.runtime_dir / "omnivoice/models/omnivoice.py").is_file():
            raise FileNotFoundError(f"找不到 OmniVoice 运行目录：{self.runtime_dir}；请运行 setup_omnivoice.py")
        if not (self.runtime_dir / "model/audio_tokenizer/model.safetensors").is_file():
            raise FileNotFoundError("缺少本地音频编解码模型，请重新运行安装脚本")
        self.threads = threads
        settings_path = self.runtime_dir / "quick_summary_voice.json"
        self.settings = {
            "instruct": "male, middle-aged, low pitch", "num_step": 16, "speed": 0.95,
        }
        if settings_path.is_file():
            self.settings.update(json.loads(settings_path.read_text(encoding="utf-8")))
        self.steps = int(self.settings["num_step"])
        if not 1 <= self.steps <= 128:
            raise ValueError("声音配置的 num_step 必须在一到一百二十八之间")
        if not math.isfinite(float(self.settings["speed"])) or float(self.settings["speed"]) <= 0:
            raise ValueError("声音配置的 speed 必须是有限正数")
        self.torch = importlib.import_module("torch")
        self.np = importlib.import_module("numpy")
        self.sf = importlib.import_module("soundfile")
        self.torch.set_num_threads(threads)
        try:
            self.torch.set_num_interop_threads(2)
        except RuntimeError:
            pass  # 同一进程已有线程池时不能重新设置，不影响设备选择。
        sys.path.insert(0, str(self.runtime_dir))
        module = importlib.import_module("omnivoice")
        if not Path(module.__file__).resolve().is_relative_to(self.runtime_dir):
            raise RuntimeError("当前进程已经加载另一个 OmniVoice 版本，请重新启动")
        started = time.perf_counter()
        print(f"加载 OmniVoice，device=cpu，float32，threads={threads}", flush=True)
        self.model = module.OmniVoice.from_pretrained(
            str(self.runtime_dir / "model"), device_map="cpu", dtype=self.torch.float32,
            load_asr=False, local_files_only=True,
        )
        if any(parameter.device.type != "cpu" for parameter in self.model.parameters()):
            raise RuntimeError("OmniVoice 存在非本机处理器参数，停止生成")
        if any(parameter.device.type != "cpu" for parameter in self.model.audio_tokenizer.parameters()):
            raise RuntimeError("音频编解码器存在非本机处理器参数，停止生成")
        self.load_seconds = time.perf_counter() - started

    def synthesize(
        self, text: str, output: Path, seed: int = 42, max_new_tokens: int = 1024
    ) -> dict:
        """NOTE max_new_tokens 在此限制输出声码帧数；扩散模型不会据此截断音频。"""
        text = " ".join(text.split())
        output = Path(output)
        if not text:
            raise ValueError("配音讲稿不能为空")
        if output.suffix.lower() != ".wav":
            raise ValueError("配音输出必须以 .wav 结尾")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须是正整数")
        self.torch.manual_seed(seed)
        started = time.perf_counter()
        calls = 0

        def report_progress(_module, _inputs, _output):
            nonlocal calls
            calls += 1
            if calls % 4 == 0:
                print(f"  OmniVoice 已完成 {calls} 次模型前向计算，用时 {time.perf_counter() - started:.1f} 秒", flush=True)

        handle = self.model.llm.register_forward_hook(report_progress)
        try:
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    text=text, language="zh", instruct=self.settings["instruct"],
                    speed=float(self.settings["speed"]), num_step=self.steps,
                )
        finally:
            handle.remove()
        if len(outputs) != 1:
            raise RuntimeError("OmniVoice 未返回唯一音频，未写入文件")
        audio = self.np.asarray(outputs[0], dtype=self.np.float32)
        if audio.ndim != 1 or not audio.size or not self.np.isfinite(audio).all():
            raise RuntimeError("OmniVoice 生成空波形、非单声道或非有限数值，未写入音频")
        if float(self.np.max(self.np.abs(audio))) < 1e-5:
            raise RuntimeError("OmniVoice 生成静音，未写入音频")
        sample_rate = int(self.model.sampling_rate)
        duration = audio.size / sample_rate
        frame_rate = float(self.model.audio_tokenizer.config.frame_rate)
        if duration * frame_rate >= max_new_tokens:
            raise RuntimeError(
                f"OmniVoice 音频超过 {max_new_tokens} 声码帧的保护上限，请拆短讲稿；未截断或写入音频"
            )
        finished = time.perf_counter()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".omnivoice-", suffix=".wav", dir=output.parent)
        os.close(descriptor)
        try:
            self.sf.write(temporary, audio, sample_rate, subtype="PCM_16")
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "backend": "omnivoice", "text": text, "duration_seconds": duration,
            "synthesis_seconds": finished - started, "load_seconds": self.load_seconds,
            "sample_rate": sample_rate, "device": "cpu", "dtype": "float32",
            "threads": self.threads, "seed": seed, "num_step": self.steps,
            "voice": self.settings["instruct"], "speed": float(self.settings["speed"]),
            "language": "zh", "length_limit_reached": False,
            "output_codec_frames_estimate": math.ceil(duration * frame_rate),
            "quality_check": "扩散步骤完成且波形有效；未做逐字语义验证，仍需人工试听。",
        }


class Audio8Speech:
    """NOTE 一个实例加载一次模型，依次复用于多段讲稿；不支持并发调用。"""

    def __init__(self, runtime_dir: Path, threads: int = 4):
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        if threads < 1:
            raise ValueError("threads 必须是正整数")
        if not (self.runtime_dir / "arktts_runtime/runtime.py").is_file():
            raise FileNotFoundError(f"找不到 Audio8 官方运行器：{self.runtime_dir}")
        self.threads = threads
        # 【Step 1】保持模型实现来自固定版本的上游源码。
        sys.path.insert(0, str(self.runtime_dir))
        module = importlib.import_module("arktts_runtime.runtime")
        if not Path(module.__file__).resolve().is_relative_to(self.runtime_dir):
            raise RuntimeError("当前进程已经加载另一个 Audio8 运行目录，请重新启动")
        self.np = importlib.import_module("numpy")
        self.sf = importlib.import_module("soundfile")
        started = time.perf_counter()
        print(f"加载 Audio8，使用 CPUExecutionProvider，threads={threads}", flush=True)
        self.runtime = module.ArkTtsRuntime(
            self.runtime_dir / "model", self.runtime_dir / "voices", threads=threads
        )
        self.load_seconds = time.perf_counter() - started
        self.providers = {
            name: getattr(self.runtime, name).get_providers()
            for name in ("slow", "fast", "decoder")
        }
        if any(value != ["CPUExecutionProvider"] for value in self.providers.values()):
            raise RuntimeError(f"发现非预期执行设备：{self.providers}")
        self.runtime.voices.load("default")

    def synthesize(
        self, text: str, output: Path, seed: int = 42, max_new_tokens: int = 1024
    ) -> dict:
        """NOTE 仅自然停止且波形有效时写入单声道文件；达到长度上限报错。"""
        text = " ".join(text.split())
        output = Path(output)
        if not text:
            raise ValueError("配音讲稿不能为空")
        if output.suffix.lower() != ".wav":
            raise ValueError("配音输出必须以 .wav 结尾")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须是正整数")
        started = time.perf_counter()
        reference, meta = self.runtime.voices.load("default")
        prompt = self.runtime.prompt_builder.build(text, meta["reference_text"], reference)
        available = int(self.runtime.manifest["max_seq_len"]) - int(prompt.shape[2])
        limit = min(max_new_tokens, available)
        if limit < 1:
            raise ValueError("讲稿超出模型上下文长度，请拆成较短的独立句子")
        # 【Step 2】保留上游采样逻辑，检查是否真正生成结束符。
        frames = []
        for frame in self.runtime.iter_codes(
            text=text, voice="default", max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, top_k=50, seed=seed,
        ):
            frames.append(frame)
            if len(frames) == 1 or len(frames) % 64 == 0:
                elapsed = time.perf_counter() - started
                print(f"  Audio8 已生成 {len(frames)} 帧，用时 {elapsed:.1f} 秒", flush=True)
        generated = time.perf_counter()
        if not frames:
            raise RuntimeError("Audio8 未生成音频，请检查声音与输入文本")
        if len(frames) >= limit:
            raise RuntimeError(
                f"Audio8 达到 {limit} 帧上限，可能重复或被截断；未写入音频。"
                "请拆短讲稿后重试，并人工试听，不能将截断结果当成成功。"
            )
        audio = self.runtime.decode_codes(self.np.stack(frames, axis=1))
        audio = self.np.asarray(audio, dtype=self.np.float32).reshape(-1)
        if not audio.size or not self.np.isfinite(audio).all():
            raise RuntimeError("Audio8 生成空波形或非有限数值，未写入音频")
        if float(self.np.max(self.np.abs(audio))) < 1e-5:
            raise RuntimeError("Audio8 生成静音，未写入音频")
        sample_rate = int(self.runtime.manifest["sample_rate"])
        finished = time.perf_counter()
        # 【Step 3】原子写入，失败时不留下可被后续步骤误用的半成品。
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".audio8-", suffix=".wav", dir=output.parent)
        os.close(handle)
        try:
            self.sf.write(temporary, audio, sample_rate, subtype="PCM_16")
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "duration_seconds": audio.size / sample_rate,
            "synthesis_seconds": finished - started,
            "generation_seconds": generated - started,
            "decode_seconds": finished - generated,
            "load_seconds": self.load_seconds,
            "sample_rate": sample_rate,
            "providers": self.providers,
            "generated_frames": len(frames),
            "threads": self.threads,
            "seed": seed,
            "voice": "default",
            "length_limit_reached": False,
            "quality_check": "有效非静音波形、生成自然结束；仍需人工核对读音、漏句与重复。",
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("omnivoice", "audio8-experimental"), default="omnivoice")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    engine_class = LocalSpeech if args.backend == "omnivoice" else Audio8Speech
    result = engine_class(args.runtime_dir, args.threads).synthesize(
        text, args.output, args.seed, args.max_new_tokens
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
