# LEARN: SGLang 怎么读

这份文档回答一件事：**从哪里开始读代码，才能用最少时间摸到引擎的骨头。**

仓库很大（约 225 万行真实代码）。不要通读。先把下面的**核心路径**和**核心文件**走通，再按任务下钻。

- 会用 API：`sglang serve` + OpenAI `/v1/chat/completions`，大约 1–2 天。
- 能改调度 / KV / 模型前向：有 vLLM 一类经验大约 3–6 周。
- 真正要精读的热路径大约 **3–4 万行**，不是 SRT 的 72 万行。

行数是 2026-09 对本仓库的扫描（含空行与注释）。文件会涨，以路径为准。

---

## 先建立心智模型

SRT（SGLang Runtime）是**三进程 ZMQ 管道**，不是单进程 Flask 服务。

```
主进程                         子进程                      子进程
HTTP + Engine + TokenizerManager ──ZMQ──► Scheduler ──ZMQ──► Detokenizer
        ▲                              │                         │
        │                              ▼                         │
        └────────────── ZMQ ──────── Detokenizer 回包 ───────────┘
```

官方说明在 [`engine.py`](python/sglang/srt/entrypoints/engine.py) 的 `Engine` 文档字符串：

1. **TokenizerManager**（主进程）：分词，把请求发给 Scheduler。
2. **Scheduler**（子进程）：收请求、组 batch、跑 GPU、把 token 发给 Detokenizer。
3. **DetokenizerManager**（子进程）：token → 文本，再回给 TokenizerManager。

HTTP server、Engine、TokenizerManager 都在主进程。进程间用 ZMQ。

CPU 侧状态和 GPU 侧状态是两套结构，官方写在 [`schedule_batch.py`](python/sglang/srt/managers/schedule_batch.py) 文件头：

```
Req → ScheduleBatch（Scheduler，多半在 CPU）
        ↓  ForwardBatch.init_new
     ForwardBatch（ModelRunner，多半是 GPU tensor）
```

中间的桥是 **`TpModelWorker`**。Scheduler **不直接**调 `ModelRunner`。

配置也分两层：`ServerArgs` 是启动时的原始记录（约 1.1 万行旗标）；`publish()` 之后业务代码读 `runtime_context` 的 namespace bag（`get_schedule()` / `get_memory()` / `get_exec()` 等），不要在热路径里直接翻 `ServerArgs` 字段。

---

## 核心路径（必须跟一遍）

这是学引擎**唯一**要跟的请求路径。其它特性（投机解码、PD 分离、LoRA、扩散）都是这条路上的支线。

```
sglang serve
    → http_server / Engine
        → OpenAI serving_chat / serving_completions
            → TokenizerManager.generate_request
                → tokenize + ZMQ
                    → [可选] DataParallelController
                        → Scheduler.event_loop
                            → process_input_requests
                            → get_next_batch_to_run   # ScheduleBatch
                            → TpModelWorker.run_batch
                                → ForwardBatch.init_new
                                → ModelRunner.forward
                                    → model.forward + Attention backend
                                    → mem_cache 读写 KV
                                    → Sampler
                            → process_batch_result
                            → ZMQ → DetokenizerManager
                        → ZMQ → TokenizerManager.handle_loop
                    → HTTP SSE / JSON 回客户端
```

| 步 | 发生什么 | 跟到哪个文件 |
|---|---|---|
| 1 启动 | 解析参数，拉起 tokenizer / scheduler / detokenizer | [`serve.py`](python/sglang/cli/serve.py) → [`engine.py`](python/sglang/srt/entrypoints/engine.py)、[`http_server.py`](python/sglang/srt/entrypoints/http_server.py) |
| 2 配置 | `ServerArgs` publish 成 RuntimeContext bags | [`server_args.py`](python/sglang/srt/server_args.py)、[`runtime_context.py`](python/sglang/srt/runtime_context.py)、[`environ.py`](python/sglang/srt/environ.py) |
| 3 API | HTTP 协议变成 `GenerateReqInput` | [`serving_chat.py`](python/sglang/srt/entrypoints/openai/serving_chat.py) 或 [`serving_completions.py`](python/sglang/srt/entrypoints/openai/serving_completions.py) |
| 4 分词 | chat template、tokenize，ZMQ 发给 Scheduler | [`tokenizer_manager.py`](python/sglang/srt/managers/tokenizer_manager.py) |
| 5 调度 | waiting/running 队列、Radix 匹配、组成 `ScheduleBatch` | [`scheduler.py`](python/sglang/srt/managers/scheduler.py)、[`schedule_batch.py`](python/sglang/srt/managers/schedule_batch.py)、[`schedule_policy.py`](python/sglang/srt/managers/schedule_policy.py) |
| 6 桥接 | `ScheduleBatch` → `ForwardBatch` | **[`tp_worker.py`](python/sglang/srt/managers/tp_worker.py)** |
| 7 前向 | eager / CUDA Graph，`model.forward` | [`model_runner.py`](python/sglang/srt/model_executor/model_runner.py)、[`forward_batch_info.py`](python/sglang/srt/model_executor/forward_batch_info.py) |
| 8 计算+缓存 | Attention 读写 KV；allocator 发槽位；radix 决定复用 | [`layers/attention/`](python/sglang/srt/layers/attention/)、[`memory_pool.py`](python/sglang/srt/mem_cache/memory_pool.py)、[`radix_cache.py`](python/sglang/srt/mem_cache/radix_cache.py) |
| 9 采样回包 | 采下一个 token，detokenize，流式回 HTTP | [`sampler.py`](python/sglang/srt/layers/sampler.py)、[`detokenizer_manager.py`](python/sglang/srt/managers/detokenizer_manager.py) |

Scheduler 主循环的骨架（`event_loop_normal`）是：

```
recv_requests → process_input_requests → get_next_batch_to_run → run_batch → process_batch_result
```

还有 overlap / PP / PD 分离等变体，都是这个骨架的分支。先读 `normal`。

IPC 消息类型全部在 [`io_struct.py`](python/sglang/srt/managers/io_struct.py)。改协议、加字段，先看它。

---

## 核心文件（按这个顺序读）

三大中枢类：`Scheduler`、`TokenizerManager`、`ModelRunner`。约定见 [`.claude/skills/large-class-style/SKILL.md`](.claude/skills/large-class-style/SKILL.md)。`ModelRunner` 已冻结为编排层——领域逻辑在 collaborator 里，顺着委托往下读，不要往这个文件里堆逻辑。

### P0 — 热路径上每一个都要打开

| 文件 | 约行数 | 为什么必须读 |
|---|---|---|
| [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py) | 5400 | 连续批处理主循环。几乎所有 serving 特性都挂在这里 |
| [`python/sglang/srt/managers/schedule_batch.py`](python/sglang/srt/managers/schedule_batch.py) | 3500 | `Req` / `ScheduleBatch`。请求状态机和 batch 字段 |
| [`python/sglang/srt/managers/tokenizer_manager.py`](python/sglang/srt/managers/tokenizer_manager.py) | 3700 | 主进程：tokenize、限流、和 Scheduler / Detokenizer 的 IPC |
| [`python/sglang/srt/managers/io_struct.py`](python/sglang/srt/managers/io_struct.py) | 2500 | 进程间消息 schema |
| [`python/sglang/srt/managers/tp_worker.py`](python/sglang/srt/managers/tp_worker.py) | 730 | **Scheduler → ModelRunner 的桥**。漏读这个会以为调度直接调模型 |
| [`python/sglang/srt/model_executor/model_runner.py`](python/sglang/srt/model_executor/model_runner.py) | 2200 | GPU worker 编排根。冻结文件 |
| [`python/sglang/srt/model_executor/forward_batch_info.py`](python/sglang/srt/model_executor/forward_batch_info.py) | 1900 | 一步 forward 的 GPU 布局，连接调度和 kernel |

建议读法：用一个最短 generate 请求，对着上表从 HTTP 跟到 `ModelRunner.forward`，再跟回 HTTP。不要平行打开 7 个文件扫一遍。

### P1 — 热路径的两端（启动、缓存、回包）

| 文件 | 约行数 | 为什么要读 |
|---|---|---|
| [`python/sglang/srt/entrypoints/engine.py`](python/sglang/srt/entrypoints/engine.py) | 1900 | 三进程怎么被拉起 |
| [`python/sglang/srt/entrypoints/http_server.py`](python/sglang/srt/entrypoints/http_server.py) | 2800 | HTTP 路由进 TokenizerManager |
| [`python/sglang/srt/entrypoints/openai/serving_chat.py`](python/sglang/srt/entrypoints/openai/serving_chat.py) | 2700 | 最常用的 OpenAI chat 适配 |
| [`python/sglang/srt/mem_cache/memory_pool.py`](python/sglang/srt/mem_cache/memory_pool.py) | 5200 | 设备 KV / SSM pool。Radix 之下的物理层 |
| [`python/sglang/srt/mem_cache/radix_cache.py`](python/sglang/srt/mem_cache/radix_cache.py) | 860 | 前缀缓存。SGLang 相对 vLLM 的标志设计 |
| [`python/sglang/srt/mem_cache/unified_radix_cache.py`](python/sglang/srt/mem_cache/unified_radix_cache.py) | 3000 | 正在收敛的统一 radix（Full / SWA / Mamba） |
| [`python/sglang/srt/mem_cache/README.md`](python/sglang/srt/mem_cache/README.md) | — | KV 分层：allocation → allocator → pool → host → storage |
| [`python/sglang/srt/managers/detokenizer_manager.py`](python/sglang/srt/managers/detokenizer_manager.py) | 560 | token → 文本，体积小但在回包路径上 |
| [`python/sglang/srt/layers/sampler.py`](python/sglang/srt/layers/sampler.py) | 800 | 采样 |
| [`python/sglang/srt/layers/radix_attention.py`](python/sglang/srt/layers/radix_attention.py) | 640 | 模型层怎么接到 attention backend |

再加 **一个** dense 模型（例如较小的 Llama / Qwen 实现）和 **一个** Attention backend（常见是 [`flashattention_backend.py`](python/sglang/srt/layers/attention/flashattention_backend.py)）。不要读 [`models/`](python/sglang/srt/models/) 下 220 个文件。

### P2 — 改配置、并行、调度策略时再读

| 文件 | 约行数 | 什么时候读 |
|---|---|---|
| [`python/sglang/srt/managers/schedule_policy.py`](python/sglang/srt/managers/schedule_policy.py) | 1500 | 改 FCFS / 优先级 / 缓存感知策略 |
| [`python/sglang/srt/distributed/parallel_state.py`](python/sglang/srt/distributed/parallel_state.py) | 3100 | 做 TP / PP / DP / EP |
| [`python/sglang/srt/configs/model_config.py`](python/sglang/srt/configs/model_config.py) | 2300 | 从 HF config 投影运行时能力 |
| [`python/sglang/srt/runtime_context.py`](python/sglang/srt/runtime_context.py) | 1700 | 读配置、改 override 的正确入口 |
| [`python/sglang/srt/environ.py`](python/sglang/srt/environ.py) | 1800 | `SGLANG_*` 环境变量登记处 |
| [`python/sglang/srt/server_args.py`](python/sglang/srt/server_args.py) | **11300** | 旗标百科。**搜索，不要通读** |
| [`.claude/skills/sglang-runtime-context/SKILL.md`](.claude/skills/sglang-runtime-context/SKILL.md) | — | RuntimeContext 设计：publish、bags、禁止读 seed |

KV 栈分层（读 `mem_cache/README.md` 比直接跳进 6 万行代码快）：

```
scheduler / model_runner / attention backend
                    ↓
            allocation.py          每 batch 分配策略
                    ↓
            hybrid_cache/          layer_id → pool
                    ↓
            allocator/             need_size → 槽位
                    ↓
            pool/ (GPU L1)  --hicache-->  pool_host/ (Host L2)  -->  storage/ (L3)
            radix cache                前缀 → 节点（复用 / 淘汰）
```

---

## 按时间学

**2 周：能跟请求、能改小功能**

1. 读完上面 P0 + `engine.py`。
2. 自己加一个 `ServerArgs` 字段，在 Scheduler 里打日志，确认请求能走到那一行。
3. 对照 `test/registered/core/` 跑一个最小测试。

产出：能画出三进程图，能回答「这个请求卡在哪一层」。

**6 周：能改调度或 KV**

1. 精读 `schedule_batch.py`、`radix_cache.py` / `unified_radix_cache.py`、`memory_pool.py`。
2. 跟一个 CUDA Graph runner（`model_executor/runner/decode_cuda_graph_runner.py`）。
3. 对照 `test/registered/radix_cache/`、`test/registered/scheduler/`。

产出：能独立做 prefix cache / 批策略类 PR。

**按任务加模块，不要通关**

| 你要做的事 | 再打开 |
|---|---|
| 新模型 | `srt/models/<name>.py`、`srt/configs/` |
| 投机解码 | `srt/speculative/`（EAGLE / DFlash / ngram） |
| Prefill–Decode 分离 | `srt/disaggregation/` |
| 多 LoRA | `srt/lora/` |
| 工具调用 / 结构化输出 | `srt/function_call/`、`srt/constrained/` |
| 集群路由 | `sgl-model-gateway/`（不是 `experimental/sgl-router/`） |
| 图像 / 视频生成 | `python/sglang/multimodal_gen/`（独立 runtime，不走 Scheduler） |
| 改 kernel / 性能 | `python/sglang/kernels/`（`ops/` + `BaseFusedOp`） |
| 写 CI | `test/README.md`、`.claude/skills/write-sglang-test/SKILL.md` |

**不要一上来做的**

- 通读 `server_args.py`（1.1 万行旗标）。
- 通读 `layers/attention/`（约 6 万行，大量后端变体）。
- 通读 `models/`（220 个模型文件，16 万行）。
- 先学 NPU / MLX / XPU（`hardware_backend/`）。
- 先学 Diffusion。
- 先学 Gateway。

---

## 仓库有几个子系统

按产品分 **5 个子系统**。学 serving 引擎只先啃 SRT。

| 子系统 | 路径 | 大约代码量 | 干什么 | 现在学吗 |
|---|---|---|---|---|
| **SRT** | `python/sglang/srt/` | 72.5 万行 Python，42 个一级包 | 自回归 LLM / VLM：调度、KV、前向、OpenAI API | **先学这个** |
| **Kernels** | `python/sglang/kernels/` | 38.7 万（含 CUDA/C++） | 统一算子：AOT 轮子、JIT、Triton / FlashInfer | 改性能再进 |
| **Diffusion** | `python/sglang/multimodal_gen/` | 32.4 万 Python | 图像 / 视频 / 音频扩散，**独立 runtime** | 做出图再学 |
| **Model Gateway** | `sgl-model-gateway/` | 12.2 万（Rust 为主） | 多 worker 路由、PD 负载均衡、cache-aware LB | 做集群再学 |
| **Rust 改写** | `rust/` | 2.7 万 | `sglang-server` / `sglang-grpc` / `sglang-mm` | 可选，进行中 |

支撑目录：`test/`（48 万，CI）、`docs/`、`benchmark/`、`scripts/`、`.github/`。

其它边界：

- 正式入口是 **`sglang serve`**（`python/sglang/cli/`），`launch_server.py` 是兼容层。
- `python/sglang/lang/` 是旧前端 DSL，不再维护。
- `experimental/sgl-router/` 是精简实验路由；生产功能在 `sgl-model-gateway/`。
- TPU / JAX **不在本仓库**，在独立的 [sglang-jax](https://github.com/sgl-project/sglang-jax)。

---

## SRT 一级包：必须 / 进阶 / 按需

SRT 下约 42 个一级包。**必须**包体积大，是因为里面有 220 个模型和大量 attention / MoE 变体；你仍然只精读热路径那 3–4 万行。

### 必须（先认门，再抽样）

| 包 | 大约千行 | 里面真正要读的 |
|---|---|---|
| `managers/` | 39 | Scheduler、TokenizerManager、`tp_worker`、`schedule_batch`、`io_struct` |
| `model_executor/` | 20 | `model_runner.py`、`forward_batch_info.py` |
| `mem_cache/` | 63 | README、`memory_pool.py`、radix / unified radix |
| `entrypoints/` | 26 | `engine.py`、`http_server.py`、`openai/serving_chat.py` |
| `layers/` | 151 | `radix_attention.py`、`sampler.py`、**一个** attention backend |
| `models/` | 165 | **一个** dense 模型，不要通读 |
| `distributed/` | 10 | `parallel_state.py`（做并行时） |
| `sampling/` | 2 | 采样参数，体积小但热路径必经 |
| `configs/` | 16 | `model_config.py` |
| 根文件 | 15 | `runtime_context.py`、`environ.py`；`server_args.py` 只搜索 |

### 进阶（做对应特性再打开）

`speculative/`、`disaggregation/`、`multimodal/`（VLM 预处理，不是扩散）、`lora/`、`model_loader/`、`function_call/`、`constrained/`、`observability/`、`eplb/`、`batch_overlap/`、`compilation/`、`arg_groups/`。

### 按需（默认跳过）

`hardware_backend/`（NPU / MLX / XPU / MUSA）、`debug_utils/`、`kv_canary/`、`dllm/`、`ray/`、`platforms/`、`tokenizer/`（薄封装，真逻辑在 `managers/`）、以及其它小包。

`layers/` 里还有约 5.6 万行 JSON，主要是 MoE Triton 调参，不是要读的代码。

---

## 另外四个子系统（知道门在哪即可）

**Kernels** — 统一 import 面是 `sglang.kernels.ops.*`。先读 `python/sglang/kernels/README.md`、`fused_op.py`、`registry.py`。体积最大的是 `ops/attention` 和 `aot/csrc`、`jit/csrc`。

**Diffusion** — 入口 `DiffGenerator` / `sglang generate`。和 SRT 共享 kernels，**不共享 Scheduler**。核心在 `multimodal_gen/runtime/{models,pipelines_core,layers,entrypoints}`。

**Model Gateway** — Rust 控制面 + 数据面。`src/routers/`（HTTP / gRPC / OpenAI proxy）、`src/core/`（worker、熔断）、`src/policies/`（cache-aware 等 LB）。

**Rust 改写** — `rust/sglang-server`（HTTP / tokenizer 雏形）、`sglang-grpc`、`sglang-mm`（VLM 预处理）。

---

## 代码量（给体感，不是阅读作业）

| 范围 | 大约行数 |
|---|---|
| 全仓库代码（py / rs / cu / c++ / go） | 225 万 |
| `python/` | 149 万 |
| `test/` | 48 万 |
| SRT Python | 72.5 万 |
| Kernels（含 CUDA） | 38.7 万 |
| Diffusion | 32.4 万 |
| Gateway | 12.2 万 |
| Rust 改写 | 2.7 万 |

SRT 里体积最大的三块：`models` 16.5 万、`layers` 15.1 万 Python、`mem_cache` 6.3 万。它们大，不代表都要读。

---

## 相关文档

| 文档 | 用途 |
|---|---|
| [`python/sglang/README.md`](python/sglang/README.md) | Python 包目录说明 |
| [`python/sglang/srt/mem_cache/README.md`](python/sglang/srt/mem_cache/README.md) | KV 分层 |
| [`python/sglang/kernels/README.md`](python/sglang/kernels/README.md) | 算子命名空间 |
| [`test/README.md`](test/README.md) | 怎么跑 / 注册 CI |
| [`.claude/skills/large-class-style/SKILL.md`](.claude/skills/large-class-style/SKILL.md) | Scheduler / TokenizerManager / ModelRunner 写法 |
| [`.claude/skills/sglang-runtime-context/SKILL.md`](.claude/skills/sglang-runtime-context/SKILL.md) | 配置怎么读、怎么改 |
| [docs.sglang.io](https://docs.sglang.io/) | 安装、API、特性说明 |

读代码从本文的「核心路径」和「核心文件」开始。其它文档是词典，不是路线图。
