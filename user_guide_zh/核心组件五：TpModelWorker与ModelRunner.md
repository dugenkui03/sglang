# 核心组件五：TpModelWorker 与 ModelRunner

> **每张 GPU 一对，和 Scheduler 在同一个进程里：启动时加载权重、分配 KV 池、捕获 CUDA Graph；每轮把 ScheduleBatch 转成 GPU 张量，前向得到 logits，再采样出下一个 token。**

## 1. 在核心链路中的位置

```mermaid
flowchart LR
    H["HTTP 接口"]
    subgraph Components["由 Engine._launch_subprocesses() 启动的三个组件"]
        T["TokenizerManager<br/>分词、提交请求、接收结果"]
        S["Scheduler<br/>排队、组批、安排执行"]
        D["DetokenizerManager<br/>生成任务：Token 转文本<br/>嵌入任务：直接转发向量"]
        T -->|"⑥ 提交请求"| S
        S -->|"⑧ 输出结果"| D
        D -->|"⑨ 回传结果"| T
    end
    M["TpModelWorker / ModelRunner<br/>调用模型执行 GPU 计算"]
    WT["权重文件<br/>safetensors"]
    KV["KV 缓存池<br/>req_to_token_pool / token_to_kv_pool"]
    AG["注意力后端<br/>CUDA Graph"]
    S -.->|"① init_model_worker：创建"| M
    M -.->|"② load_model：加载权重"| WT
    M -.->|"③ alloc_memory_pool：分配 KV 池"| KV
    M -.->|"④ 初始化注意力后端、捕获 CUDA Graph"| AG
    H <-->|"⑤ 发起请求<br/>⑩ 返回结果"| T
    S <-->|"⑦ 执行批次 ScheduleBatch<br/>返回 GenerationBatchResult"| M
    style Components fill:none,stroke:#5684c4,stroke-width:2px,stroke-dasharray:6 4
    style M fill:#fff0c2,stroke:#b7791f,stroke-width:3px
```

虚线是启动阶段（①–④），实线是请求处理链路（⑤–⑩）。

- **启动**：Scheduler 的 [`init_model_worker`](../python/sglang/srt/managers/scheduler.py#L1031) 按下面的顺序调用。
  - ① [`init_tp_model_worker`](../python/sglang/srt/managers/scheduler.py#L939) 创建 [`TpModelWorker`](../python/sglang/srt/managers/tp_worker.py#L312)：先由 [`_init_model_config`](../python/sglang/srt/managers/tp_worker.py#L461) 读模型配置，再由 [`_init_model_runner`](../python/sglang/srt/managers/tp_worker.py#L482) 创建 [`ModelRunner`](../python/sglang/srt/model_executor/model_runner.py#L291)。
  - ② `ModelRunner.__init__` 先 [`init_torch_distributed`](../python/sglang/srt/model_executor/model_runner.py#L1103) 建好多卡通信，再进 [`initialize`](../python/sglang/srt/model_executor/model_runner.py#L647)：创建采样器，[`load_model`](../python/sglang/srt/model_executor/model_runner.py#L1121) 加载权重，确定这张卡负责哪几层（流水线并行时只负责一部分，见 [1.7](<1.7 长上下文流水线并行(Pipeline Parallelism for Long Context).md>)）。
  - ③ [`init_memory_pools`](../python/sglang/srt/managers/scheduler.py#L1004) → [`ModelRunner.alloc_memory_pool`](../python/sglang/srt/model_executor/model_runner.py#L873)：按 `mem_fraction_static` 和加载权重后剩下的显存，算出 KV 池能放多少 token。
  - ④ [`init_attention_backends`](../python/sglang/srt/model_executor/model_runner.py#L993)、[`init_cuda_graphs`](../python/sglang/srt/model_executor/model_runner.py#L1058)：初始化注意力后端，再按一组批大小把 decode 的前向录成 CUDA Graph。
  - 最后 [`get_worker_info`](../python/sglang/srt/managers/tp_worker.py#L547) 把 `max_total_num_tokens`、`max_running_requests` 等交回 Scheduler，组批时的预算就来自这里。
- **要点**：
  - Scheduler 调用它们是普通的函数调用，不走进程间通信。
  - TpModelWorker 是薄的一层：把批次交给 ModelRunner，再把结果装进 `GenerationBatchResult`；模型、KV 池、注意力后端和 CUDA Graph 都挂在 ModelRunner 上。
  - 张量并行时，同一个 TP 组的每张卡各有一对，拿到相同的批次一起计算，每张卡只持有 1/N 的权重。
  - 开了推测解码时，同一进程里还有草稿模型的 worker，它和目标模型共用 KV 池；这时 `run_batch` 调用的是草稿 worker（[L1067](../python/sglang/srt/managers/scheduler.py#L1067)），由它再调用目标模型做验证。
  - 一个 TpModelWorker 通常只有一个 ModelRunner；只有多层 EAGLE 草稿模型会在 `model_runner_list` 里为每一步建一个（[`_init_multi_layer_eagle_model_runners`](../python/sglang/srt/managers/tp_worker.py#L503)）。

### TpModelWorker 与 ModelRunner 内部

```mermaid
flowchart LR
    S["Scheduler.run_batch()"]
    subgraph TW["TpModelWorker.forward_batch_generation()"]
        FB["ForwardBatch.init_new()<br/>ScheduleBatch 转 GPU 张量"]
        RES["GenerationBatchResult<br/>logits_output、next_token_ids"]
    end
    subgraph MR["ModelRunner（一张 GPU 一个）"]
        F["forward() → _forward_raw()"]
        DG["decode_cuda_graph_runner<br/>decode 的 CUDA Graph"]
        PG["prefill_cuda_graph_runner<br/>分段 CUDA Graph"]
        ER["eager_runner<br/>decode / extend / idle"]
        MF["model.forward()<br/>逐层计算"]
        AB["attn_backend<br/>读写 KV 池"]
        SP["sample() → Sampler<br/>温度、top-k、top-p"]
    end
    S -->|"① ScheduleBatch"| FB
    FB -->|"② ForwardBatch"| F
    F -->|"③ decode 且图已捕获"| DG
    F -->|"③ prefill 且图可用"| PG
    F -->|"③ 其他情况"| ER
    DG -->|"④ 重放"| MF
    PG -->|"④ 重放"| MF
    ER -->|"④ 直接调用"| MF
    MF <-->|"⑤ 注意力层读写 KV"| AB
    MF -->|"⑥ logits"| SP
    SP -->|"⑦ next_token_ids"| RES
    RES -->|"⑧ 返回"| S
    style TW fill:none,stroke:#5684c4,stroke-width:2px,stroke-dasharray:6 4
    style MR fill:none,stroke:#589765,stroke-width:2px,stroke-dasharray:6 4
    style FB fill:#fff0c2,stroke:#b7791f,stroke-width:3px
    style F fill:#fff0c2,stroke:#b7791f,stroke-width:3px
    style SP fill:#fff0c2,stroke:#b7791f,stroke-width:3px
```

- 编号对应三步：①② 构造 ForwardBatch，③–⑤ 前向，⑥⑦ 采样；黄框是三步各自的入口。
- ⑥ 实际是 TpModelWorker 拿到 logits 后再调 `model_runner.sample`；overlap 模式下带语法约束时，这一步推迟到 Scheduler 下一轮。
- CUDA Graph 把整次前向的 GPU kernel 录下来，之后同样形状的批直接重放，省掉 CPU 逐个下发 kernel 的时间；条件不满足时退回 eager。

## 2. 浅层时序图(Sequence Diagram)

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant W as TpModelWorker
    participant FB as ForwardBatch
    participant R as ModelRunner
    participant G as CUDA Graph / EagerRunner
    participant A as attn_backend / KV 池
    participant SP as Sampler
    rect rgb(242, 242, 242)
        Note over S,R: 启动（init_model_worker）
        S->>W: TpModelWorker(...)
        W->>R: ModelRunner(...)：init_torch_distributed、initialize（load_model）
        S->>W: alloc_memory_pool、init_attention_backends、init_cuda_graphs
        W->>R: 分配 KV 池、初始化注意力后端、捕获 CUDA Graph
        W-->>S: get_worker_info：max_total_num_tokens、max_running_requests
    end
    rect rgb(232, 242, 255)
        Note over S,FB: Step 1 构造 ForwardBatch
        S->>W: forward_batch_generation(batch)
        W->>FB: ForwardBatch.init_new(batch, model_runner)
        FB-->>W: input_ids、positions、seq_lens、out_cache_loc 等 GPU 张量
    end
    rect rgb(233, 247, 237)
        Note over W,A: Step 2 前向
        W->>R: forward(forward_batch)
        R->>G: 能用 CUDA Graph 就重放，否则 eager 执行
        G->>A: 注意力层读这批请求的历史 KV，写入本轮的新 KV
        G-->>R: logits
        R-->>W: ModelRunnerOutput（logits_output、can_run_graph）
    end
    rect rgb(255, 243, 225)
        Note over W,SP: Step 3 采样
        W->>R: sample(logits_output, forward_batch)
        R->>SP: 加语法掩码和 logit 偏置后，按温度、top-k、top-p 采样
        SP-->>R: next_token_ids
        R-->>W: next_token_ids
        W-->>S: GenerationBatchResult
    end
```

- **启动**：见第 1 节。
  - 顺序不能换：KV 池的大小取决于权重加载完还剩多少显存，CUDA Graph 要在模型和注意力后端都就绪后才能录。
  - 投机解码时，草稿 worker 也走一遍同样的流程，但直接复用目标模型的 KV 池（[`init_memory_pools`](../python/sglang/srt/managers/scheduler.py#L1004)）。
- **Step 1 构造 ForwardBatch**：[`ForwardBatch.init_new`](../python/sglang/srt/model_executor/forward_batch_info.py#L723)，对应代码里的 [【Step 1】](../python/sglang/srt/managers/tp_worker.py#L614)。
  - `ScheduleBatch` 是调度视角的批次（请求列表、前缀长度等，大多在 CPU 上）；`ForwardBatch` 是模型视角：`input_ids`、`positions`、`seq_lens`、`req_pool_indices`、`out_cache_loc`（本轮新 KV 写到哪些槽位）、`sampling_info`。
  - `forward_mode` 决定后面怎么跑：[`ForwardMode`](../python/sglang/srt/model_executor/forward_batch_info.py#L104) 里 `EXTEND` 就是 prefill，`DECODE` 每个请求算 1 个 token，`IDLE` 是 DP 模式下没分到请求的卡空跑一轮。
- **Step 2 前向**：[`ModelRunner.forward`](../python/sglang/srt/model_executor/model_runner.py#L1572) → [`_forward_raw`](../python/sglang/srt/model_executor/model_runner.py#L1717)，对应 [【Step 2】](../python/sglang/srt/managers/tp_worker.py#L635)。
  - decode 批满足条件（批大小在捕获范围内等，见 [`can_run_graph`](../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L676)）时，[重放 decode 的 CUDA Graph](../python/sglang/srt/model_executor/model_runner.py#L1749)。
  - 否则先看 prefill 的[分段 CUDA Graph](../python/sglang/srt/model_executor/model_runner.py#L1800) 能不能用（见 [5.1](<5.1 分段 CUDA 图(Piecewise CUDA Graph).md>)），都不行就交给 [`EagerRunner.execute`](../python/sglang/srt/model_executor/runner/eager_runner.py#L212)：按 decode / idle / extend 分派，注意力后端准备好元数据后调用 `model.forward(input_ids, positions, forward_batch)`。
  - 返回 [`ModelRunnerOutput`](../python/sglang/srt/model_executor/model_runner.py#L266)：`logits_output` 和这次是否用了 CUDA Graph。两种执行路径的对比见 [docs/learn/04](../docs/learn/04-model-forward-execution.md)。
- **Step 3 采样**：[`ModelRunner.sample`](../python/sglang/srt/model_executor/model_runner.py#L1846) 调用 [`Sampler`](../python/sglang/srt/layers/sampler.py#L71)，对应 [【Step 3】](../python/sglang/srt/managers/tp_worker.py#L655)。
  - 先由 [`_preprocess_logits`](../python/sglang/srt/model_executor/model_runner.py#L1818) 加上结构化输出的词表掩码和 logit 偏置。
  - 全部请求都是贪心（temperature 为 0）时[直接取分数最高的 token](../python/sglang/srt/layers/sampler.py#L130)；否则 logits 除以温度、softmax，再按 top-k / top-p / min-p 采样。请求里的 temperature、top_k 等参数就在这里生效。
  - TpModelWorker 把 `next_token_ids` 和 `logits_output` 一起装进 [`GenerationBatchResult`](../python/sglang/srt/managers/utils.py#L45) 交回 Scheduler，接上[核心组件四](核心组件四：Scheduler.md)的「处理结果」。
- **图中省略的分支**：
  - 推迟采样：overlap 模式下带语法约束（或开了 `SGLANG_ENABLE_DELAY_SAMPLE`）时，[先返回](../python/sglang/srt/managers/tp_worker.py#L658)带 `delay_sample_func` 的结果，Scheduler 在 [`launch_batch_sample_if_needed`](../python/sglang/srt/managers/scheduler.py#L4174) 里更新完语法状态再采样。
  - 只做 prefill 的请求（例如只要 logprob）不采样，[填 0 占位](../python/sglang/srt/managers/tp_worker.py#L684)。
  - 流水线并行的非最后一段只算本段的层，把隐状态交给下一段，[不采样](../python/sglang/srt/managers/tp_worker.py#L702)。

## 3. 继承关系与职责

`class TpModelWorker(BaseTpWorker)`；`ModelRunner` 没有父类，执行方式拆成了几个 runner 对象。

```mermaid
classDiagram
    class BaseTpWorker {
        <<abstract>>
        +forward_batch_generation(batch)*
        +get_memory_pool()
        +update_weights_from_disk(recv_req)
    }
    class TpModelWorker {
        +model_config
        +model_runner
        +model_runner_list
        +forward_batch_generation(batch)
        +alloc_memory_pool()
        +init_cuda_graphs()
        +get_worker_info()
    }
    class ModelRunner {
        +model
        +req_to_token_pool
        +token_to_kv_pool
        +attn_backend
        +initialize()
        +forward(forward_batch)
        +sample(logits_output, forward_batch)
    }
    class BaseRunner {
        <<abstract>>
        +execute(forward_batch)*
    }
    class EagerRunner {
        +execute(forward_batch)
    }
    class DecodeCudaGraphRunner {
        +can_run_graph(forward_batch)
        +execute(forward_batch)
    }
    class PrefillCudaGraphRunner {
        +execute(forward_batch)
    }
    class Sampler {
        +forward(logits_output, sampling_info)
    }
    BaseTpWorker <|-- TpModelWorker
    TpModelWorker *-- ModelRunner : model_runner
    ModelRunner *-- EagerRunner : eager_runner
    ModelRunner *-- DecodeCudaGraphRunner : decode_cuda_graph_runner
    ModelRunner *-- PrefillCudaGraphRunner : prefill_cuda_graph_runner
    ModelRunner *-- Sampler : sampler
    BaseRunner <|-- EagerRunner
    BaseRunner <|-- DecodeCudaGraphRunner
    BaseRunner <|-- PrefillCudaGraphRunner
    style TpModelWorker fill:#fff0c2,stroke:#b7791f,stroke-width:3px
    style ModelRunner fill:#fff0c2,stroke:#b7791f,stroke-width:3px
```

| 类 | 职责 | 核心方法 | 说明 |
|---|---|---|---|
| [`BaseTpWorker`](../python/sglang/srt/managers/tp_worker.py#L82) | worker 的公共接口 | `forward_batch_generation`（抽象） | MLX 等硬件后端有自己的 worker 实现 |
| - | - | `get_memory_pool` / `update_weights_from_disk` | 把 KV 池交给 Scheduler；在线换权重，转交给 ModelRunner |
| [`TpModelWorker`](../python/sglang/srt/managers/tp_worker.py#L312) | Scheduler 调用模型的入口，一张卡一个 | `forward_batch_generation` | 三步：构造 ForwardBatch、前向、采样，结果装进 `GenerationBatchResult` |
| - | - | `alloc_memory_pool` / `init_attention_backends` / `init_cuda_graphs` | 启动时由 Scheduler 依次调用，转给 `model_runner_list` 里的每个 runner |
| - | - | `get_worker_info` | 把 KV 池容量、最大并发请求数等交回 Scheduler |
| [`ModelRunner`](../python/sglang/srt/model_executor/model_runner.py#L291) | 持有一张卡上的模型和显存 | `initialize` | 建通信、创建采样器、加载权重、确定本卡负责的层 |
| - | - | `alloc_memory_pool` / `init_attention_backends` / `init_cuda_graphs` | 分配 KV 池、初始化注意力后端、捕获 CUDA Graph |
| - | - | `forward` / `sample` | 选 CUDA Graph 或 eager 执行前向；调用 Sampler 采样 |
| [`EagerRunner`](../python/sglang/srt/model_executor/runner/eager_runner.py#L84) | 普通执行，不用 CUDA Graph | `execute` | 按 decode / idle / extend 分派，调用 `model.forward` |
| [`DecodeCudaGraphRunner`](../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L209) | 录制并重放 decode 的 CUDA Graph | `can_run_graph` / `execute` | 启动时按一组批大小录好，满足条件的 decode 批直接重放 |
| [`PrefillCudaGraphRunner`](../python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py#L255) | prefill 的分段 CUDA Graph | `execute` | 只在分段 CUDA Graph 可用时启用 |
| [`Sampler`](../python/sglang/srt/layers/sampler.py#L71) | 从 logits 选出下一个 token | `forward` | 由 [`create_sampler`](../python/sglang/srt/layers/sampler.py#L545) 创建，一个 ModelRunner 一个 |

- **为什么这样拆**：
  - TpModelWorker 只做编排，所以推测解码、dLLM 这类算法可以包住或替换它，不用改 ModelRunner。
  - ModelRunner 管「这张卡上的东西」：模型、KV 池、注意力后端、CUDA Graph。它属于代码量很大的中枢类，初始化细节拆到 `model_executor/model_runner_components/`，执行方式拆成 `runner/` 里的几个 runner，每轮按批次挑一个。

## 术语与生词

| 术语或单词 | 中文释义 | 简明英文释义 |
|---|---|---|
| TpModelWorker | 张量并行模型 worker：Scheduler 调用模型的入口，一张卡一个 | The per-GPU entry point the Scheduler calls to run the model. |
| ModelRunner | 模型运行器：持有本卡的模型、KV 池和注意力后端，执行前向和采样 | Owns the model, KV pool and attention backend on one GPU. |
| ForwardBatch | 前向批次：一轮前向要用的 GPU 张量和元数据 | GPU tensors and metadata for one forward pass. |
| Logits /ˈlɒdʒɪts/ | 模型给词表里每个 token 打的原始分数，softmax 后变成概率 | Raw per-token scores that softmax turns into probabilities. |
| Sampling /ˈsæmplɪŋ/ | 采样：按温度、top-k、top-p 从概率分布里选出下一个 token | Picking the next token from the probability distribution. |
| Greedy /ˈɡriːdi/ | 贪心：直接取分数最高的 token | Always taking the highest-scoring token. |
| CUDA Graph | CUDA 图：把一串 GPU kernel 录下来，之后一次性重放 | A recorded sequence of GPU kernels that can be replayed at once. |
| Eager /ˈiːɡər/ | 即时执行：不用 CUDA Graph，由 Python 逐个下发 kernel | Running kernels one by one from Python, without a graph. |
| Attention Backend | 注意力后端：注意力计算的具体实现（FlashInfer、FlashAttention、Triton 等），负责读写 KV 池 | The kernel library that computes attention and reads/writes the KV pool. |
