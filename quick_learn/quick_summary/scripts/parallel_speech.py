"""NOTE 按讲稿片段并行配音，立即保存成功缓存，最后按键返回结果。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import resource
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable

_ENGINE = None


def available_cpu_count() -> int:
    """NOTE 同时考虑进程亲和性和容器配额，避免按宿主机核心数过量并行。"""
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    roots = {
        Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/cpu"),
        Path("/sys/fs/cgroup/cpu,cpuacct"),
    }
    directories = set(roots)
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _hierarchy, controllers, relative = line.split(":", 2)
            if controllers and "cpu" not in controllers.split(","):
                continue
            for root in roots:
                directory = root / relative.lstrip("/")
                # 同时检查父容器限制；即使当前组未限额，也不能超过父组。
                while directory.is_relative_to(root):
                    directories.add(directory)
                    if directory == root:
                        break
                    directory = directory.parent
    except (OSError, ValueError):
        pass
    for directory in directories:
        try:
            quota, period = (directory / "cpu.max").read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                limits.append(max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            pass
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text())
            period = int((directory / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, min(limits))


def _initialize_worker(runtime_dir: str, threads: int) -> None:
    global _ENGINE
    # 【Step 1】由父进程统一处理中断，每个工作进程只加载一次模型。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)
    for variable in ("OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # NOTE 算子间调度串行，避免各工作进程再叠加一组并行计算线程。
    import torch

    torch.set_num_interop_threads(1)
    from speech import LocalSpeech

    _ENGINE = LocalSpeech(Path(runtime_dir), threads=threads)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(path: Path, metadata: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".speech-meta-", suffix=".json", dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _synthesize_job(job: dict) -> dict:
    if _ENGINE is None:
        raise RuntimeError("配音工作进程未初始化模型")
    output = Path(job["path"])
    started = time.perf_counter()
    print(f"[片段开始] {output.name}，工作进程 {os.getpid()}", flush=True)
    statistics = _ENGINE.synthesize(job["text"], output, seed=job["seed"])
    metadata = {
        **statistics,
        "cache_key": job.get("cache_key", job["key"]),
        "segment_key": job["key"],
        "audio_sha256": _sha256(output),
        "spoken_text": job["text"],
        "worker_pid": os.getpid(),
        "torch_threads": _ENGINE.torch.get_num_threads(),
        "torch_interop_threads": _ENGINE.torch.get_num_interop_threads(),
        "peak_resident_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "os_threads_observed": len(list(Path('/proc/self/task').iterdir())),
        "job_seconds": time.perf_counter() - started,
    }
    # 【Step 2】成功即写入元数据，中断后仍可复用其他已完成片段。
    _write_metadata(output.with_suffix(".json"), metadata)
    return metadata


def _terminate_workers(executor: ProcessPoolExecutor) -> None:
    """NOTE 只终止此执行器创建的进程，兼容尚无 terminate_workers 的 Python。"""
    # Python 3.11 没有公开的强制终止接口，只读取当前执行器的进程句柄。
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            pass
    for process in processes:
        try:
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        except (OSError, ValueError):
            pass
    executor.shutdown(wait=False, cancel_futures=True)


class SpeechJobsError(RuntimeError):
    """NOTE 汇总失败，同时向调用方保留已完成结果，避免全部重新生成。"""

    def __init__(self, failures: dict[str, str], results: dict[str, dict]):
        self.failures = failures
        self.results = results
        details = "；".join(failures.values())
        super().__init__(f"{len(failures)} 个配音片段失败，已保留 {len(results)} 个成功缓存：{details}")


def synthesize_jobs(
    jobs: list[dict], runtime_dir: Path, threads: int, workers: int, retries: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """NOTE 按片段键返回结果；回调仅在父进程执行，调用方负责依原顺序拼接。"""
    if threads < 1 or workers < 1 or retries < 0:
        raise ValueError("threads、workers 必须为正整数，retries 不能为负数")
    if not jobs:
        return {}
    workers = min(workers, len(jobs))
    available = available_cpu_count()
    if workers * threads > available:
        raise ValueError(
            f"并行设置共需 {workers * threads} 个线程，但容器有效配额为 {available}；"
            "请降低 workers 或 threads"
        )
    seen_keys, seen_paths = set(), set()
    normalized = []
    for job in jobs:
        key = str(job.get("key", ""))
        text = job.get("text")
        output = Path(job.get("path", ""))
        if not key or not isinstance(text, str) or not text.strip():
            raise ValueError("每个配音片段必须有非空 key 和 text")
        if not output.is_absolute() or output.suffix.lower() != ".wav":
            raise ValueError(f"片段路径必须是绝对 .wav 路径：{output}")
        output = output.resolve()
        if key in seen_keys or output in seen_paths:
            raise ValueError(f"片段 key 或输出路径重复：{output.name}")
        seen_keys.add(key)
        seen_paths.add(output)
        normalized.append({"key": key, "cache_key": job.get("cache_key", key),
                           "text": text, "path": str(output), "seed": int(job.get("seed", 42))})
    runtime_dir = Path(runtime_dir).expanduser().resolve()
    results: dict[str, dict] = {}
    pending = normalized
    failures: dict[str, str] = {}
    print(
        f"[并行配音] {len(jobs)} 个片段，{workers} 个工作进程，每进程 {threads} 线程，"
        f"有效配额 {available} 核",
        flush=True,
    )
    executor = None
    futures = {}
    try:
        # 【Step 3】仅失败片段进入下一轮，普通重试继续复用已加载的模型。
        for attempt in range(retries + 1):
            if not pending:
                break
            if attempt:
                print(f"[片段重试] 第 {attempt}/{retries} 次，仅重试 {len(pending)} 个失败片段", flush=True)
            if executor is None:
                executor = ProcessPoolExecutor(
                    max_workers=min(workers, len(pending)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(str(runtime_dir), threads),
                )
            next_pending, futures, broken = [], {}, False
            for job in pending:
                futures[executor.submit(_synthesize_job, job)] = job
            for future in as_completed(futures):
                job = futures[future]
                try:
                    metadata = future.result()
                except Exception as error:
                    broken = broken or isinstance(error, BrokenProcessPool)
                    message = f"{Path(job['path']).name}: {type(error).__name__}: {error}"
                    failures[job["key"]] = message
                    next_pending.append(job)
                    print(f"[片段失败] {message}", flush=True)
                    event = {"key": job["key"], "status": "retrying" if attempt < retries else "failed",
                             "attempt": attempt + 1, "error": message}
                else:
                    results[job["key"]] = metadata
                    failures.pop(job["key"], None)
                    print(f"[片段完成 {len(results)}/{len(jobs)}] {Path(job['path']).name}", flush=True)
                    event = {"key": job["key"], "status": "complete", "attempt": attempt + 1,
                             "metadata": metadata}
                if on_progress:
                    on_progress(event)
            pending = next_pending
            if broken:
                # NOTE 进程退出后才重建执行器，已落盘的成功片段仍可复用。
                _terminate_workers(executor)
                executor = None
    except BaseException:
        for future in futures:
            future.cancel()
        if executor is not None:
            _terminate_workers(executor)
            executor = None
        print("[并行配音已中断] 本次工作进程已终止，成功片段缓存保留", flush=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        raise SpeechJobsError(failures, results)
    return results
