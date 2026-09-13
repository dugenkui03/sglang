---
title: "二、DFlash：草稿生成与并行校验"
description: "以 Qwen3.8-27B 和配套 DFlash2 草稿模型为例，理解模型搭配、并行校验、失败修正与 SGLang 配置。"
---

# 二、DFlash：草稿生成与并行校验

DFlash 的核心流程：**小模型快速生成多个候选 token → 目标模型并行校验 → 接受连续通过的部分 → 修正失败位置 → 继续生成。**

![DFlash 科普图：配套模型、并行草稿、目标校验与修正、训练和加速原理](/Users/bytedance/github/sglang/docs/learn/dflash-4x3.png)

## 1. 两个模型怎样配套

- 目标模型：**Qwen3.8-27B**，负责处理提示词、提供内部特征，并决定最终输出。
- 草稿模型：**incoai/Qwen3.8-27B-DFlash2**，利用目标模型的内部特征，快速提出一组候选 token。
- 草稿需要针对目标模型训练和适配；训练时固定目标模型，训练草稿的参数。
- 有现成的配套草稿权重时，直接加载即可；特征维度、token 编号和模型接口需要兼容。

这里使用 DFlash2 草稿检查点，在当前 SGLang 中通过 `DFLASH` 算法入口加载。

## 2. 一次生成怎样执行

1. 目标模型执行 Prefill，处理输入，建立 KV Cache，生成首个 token，并提供内部特征。
2. 草稿模型根据已确认的上下文和目标特征，一次并行预测后续一组候选 token。
3. 目标模型对给定草稿做一次前向计算，同时得到多个位置的预测分布。
4. 按顺序接受连续通过的候选；首个失败位置产生替代 token，后面的旧草稿丢弃。
5. 更新已确认的上下文与缓存，继续下一轮，直到满足停止条件。

假设 A 已确认，B、C、D 是草稿；字母仅表示示意 token：

```text
校验 B：依据 A          → 通过
校验 C：依据 A、B       → 失败
校验 D：依据 A、B、C    → 通过，但依赖未被接受的 C
最终保留：A → B → C′   → 从这里继续下一轮
```

C′ 根据本次校验已算出的分布，按生成规则确定，通常无需为这个位置额外跑一次前向计算。
随机采样时，接受、拒绝和修正规则保持目标模型的输出分布，因此可以保留生成的多样性。

## 3. 为什么可能更快

- 草稿模型层数少，单次计算较轻；并行预测一组 token，又减少了逐个生成的等待。
- 草稿已经给出候选上下文，目标模型可以同时计算多个位置，分摊权重读取开销并提高 GPU 利用率。
- 草稿计算和被丢弃的校验会增加开销；连续接受的 token 足够多，才能抵消这些开销。

判断收益可以看：`平均每 token 耗时 ≈（草稿耗时 + 校验耗时）/ 本轮平均实际推进的 token 数`。
数学、代码、聊天都可以使用；具体收益取决于草稿质量、采样参数和负载，写作场景也需要实际测量。

## 4. 在 SGLang 中怎样使用

以 Linux / NVIDIA CUDA 单实例、TP=1、PP=1、关闭 DP Attention 为配置示意；环境需支持该模型和 DFlash2，并有足够显存。指定以下参数，其余配置参考 Cookbook：

| 参数 | 示例值 |
|---|---|
| `--model-path` | `Qwen/Qwen3.8-27B` |
| `--speculative-algorithm` | `DFLASH` |
| `--speculative-draft-model-path` | `incoai/Qwen3.8-27B-DFlash2` |
| `--speculative-num-draft-tokens` | `8` |

这里的 8 是窗口长度，实际每轮推进多少 token 取决于接受结果；客户端仍使用普通生成接口。
[Qwen3.8-27B 配置依据](/Users/bytedance/github/sglang/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx:246)。

## 5. 对应哪些核心代码

- [Scheduler 请求检查](/Users/bytedance/github/sglang/python/sglang/srt/managers/scheduler.py:2662)：检查 DFlash 模式是否支持当前请求参数；失败则安排错误响应。
- [DFlashWorkerV2.forward_batch_generation](/Users/bytedance/github/sglang/python/sglang/srt/speculative/dflash_worker_v2.py:1660)：组织目标模型 Prefill、草稿生成、目标校验和结果提交。
- [DFlashDraftModel](/Users/bytedance/github/sglang/python/sglang/srt/models/dflash.py:545)：草稿模型实现；[启动参数检查](/Users/bytedance/github/sglang/python/sglang/srt/arg_groups/speculative_hook.py:185)：检查设备、并行配置和草稿路径。

参考：[DFlash 原论文](https://arxiv.org/html/2602.06036v1)、[项目与配套权重](https://github.com/z-lab/dflash)、[随机采样原理](https://arxiv.org/abs/2211.17192)。
