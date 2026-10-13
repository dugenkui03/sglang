---
title: "四、模型推理执行：CUDA Graph 与 Eager"
description: "通过一张分支图，理解 Prefill、Decode 与 CUDA Graph、Eager 的关系，以及模型推理在哪里执行。"
---

# 四、模型推理执行：CUDA Graph 与 Eager

**CUDA Graph 和 Eager 都会执行模型推理，`ModelRunner._forward_raw()` 为本轮 batch 选择执行路径。**

Prefill / Decode 决定“本轮算什么”：处理输入，或继续生成；CUDA Graph / Eager 决定“怎么执行”：运行捕获的计算图，或直接调用模型前向方法。

普通文本生成的核心分支如下：

```text
ModelRunner.forward()
    ↓
ModelRunner._forward_raw()
    │
    ├─ 满足 Decode 计算图条件
    │    └─ decode_cuda_graph_runner.execute()
    │
    ├─ 满足 Prefill 计算图条件
    │    └─ prefill_cuda_graph_runner.execute()
    │         ├─ _execute_body_capture()
    │         └─ _execute_tc_piecewise()
    │
    └─ 使用 Eager 执行
         └─ eager_runner.execute()
              ├─ Prefill 阶段 → _execute_extend()
              └─ Decode 阶段  → _execute_decode()
```

一次请求先完成 Prefill，之后通常经过多轮 Decode；调度循环每轮选择 batch，执行器再根据当前条件选择路径。
`_execute_extend()` 和 `_execute_decode()` 分别处理对应阶段，由 `EagerRunner.execute()` 根据本轮模式调用。

## 哪个调用执行推理

| 路径 | 执行模型计算的位置 |
|---|---|
| Prefill CUDA Graph | `prefill_cuda_graph_runner.execute()` 内通过 `backend.replay()` 执行捕获的模型计算。 |
| Eager Prefill / Decode | `_execute_extend()` / `_execute_decode()` 内调用 `model_runner.model.forward(...)`。 |

CUDA Graph 使用本轮输入重新运行计算；例如 `_execute_body_capture()` 用计算图执行 Transformer 主体，输出头等部分仍由普通模型代码执行。
`load_batch()` 准备本轮输入，`_finalize_execute_output()` 整理输出，整个 `execute()` 调用包含实际推理。

## 代码入口

- [model_runner.py](../../python/sglang/srt/model_executor/model_runner.py)：`forward()` 调用 `_forward_raw()`，选择执行路径。
- [prefill_cuda_graph_runner.py](../../python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py)：`execute()` 进入 Prefill 计算图执行流程。
- [eager_runner.py](../../python/sglang/srt/model_executor/runner/eager_runner.py)：`execute()` 分发到 `_execute_extend()` 或 `_execute_decode()`。
