---
title: "三、PD 分离：一次请求怎样接力完成"
description: "理解 SMG 如何分发请求，以及 Prefill 与 Decode 实例如何通过 KV Cache 传输完成一次推理。"
---

# 三、PD 分离：一次请求怎样接力完成

普通模式下，一个 SGLang 实例负责 Prefill 和 Decode；PD 分离时，两个实例各负责一个阶段，共同完成一次请求。

## 请求怎样经过两个实例

SMG 网关选择一对 P/D 实例，将同一次用户请求分发给两边；每个实例都有自己的 HTTP、TokenizerManager 和 Scheduler。

```text
                         用户请求
                            ↓
                        SMG 网关
                  ┌─────────┴─────────┐
                  ↓                   ↓
             Prefill 实例         Decode 实例
           HTTP + Tokenizer      HTTP + Tokenizer
                  ↓                   ↓
             Scheduler           Scheduler
            模式 PREFILL         模式 DECODE
                  ↓                   ↓
              处理输入           准备接收空间
              计算 KV ── KV 传输 ──→ 等待 KV 到齐
                                      ↓
                                 持续生成 token
                                      ↓
                                SMG → 返回用户
```

两边可以同时收到请求，Decode 等待所需的 KV Cache 到齐后继续生成；`bootstrap_room` 用于配对两边同一次请求的 KV 传输。

## 对应到正在看的代码

`Scheduler._add_request_to_queue()` 根据当前实例的运行模式入队：

| 分支 | 请求进入哪里 | 接下来做什么 |
|---|---|---|
| `NULL` | `waiting_queue` | 在本实例中调度 Prefill 和 Decode。 |
| `PREFILL` | `disagg_prefill_bootstrap_queue` | 准备传输连接，随后执行 Prefill 并发送 KV。 |
| `DECODE` | `disagg_decode_prealloc_queue` | 准备接收空间，随后接收 KV 并执行 Decode。 |

同一份 Scheduler 代码分别运行在 P、D 实例中，各自进入对应分支，通过 KV 传输衔接两个阶段。

分发入口是 `sgl-model-gateway/src/routers/http/pd_router.rs` 中的 `execute_dual_dispatch_internal()`，它向 P、D 两边发送请求。
部署方式可参考 [SGLang 官方 PD 分离文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。
