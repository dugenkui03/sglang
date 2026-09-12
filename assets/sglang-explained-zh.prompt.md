# SGLang 中文科普图：生成提示词与依据

- 成图：[sglang-explained-zh.png](sglang-explained-zh.png)
- 生成方式：内置 `image_gen`，单张新图（infographic-diagram）。
- 研究范围：当前工作区的 `LEARN.md` 与源码；Git HEAD 为 `2380121e9`。
- 内容范围：普通自回归 SRT 请求主链路，省略并行副本、PD 分离及其它可选分支。

## 核对依据

- [LEARN.md](../LEARN.md)：请求主线、子系统边界与阅读路线。
- [README.md](../README.md)：项目定位与运行时加速特性。
- [engine.py](../python/sglang/srt/entrypoints/engine.py)：`Engine` 文档字符串明确三个组件、主进程职责与 ZMQ IPC。
- [scheduler.py](../python/sglang/srt/managers/scheduler.py)：`event_loop_normal`、`event_loop_overlap`、`run_batch`。
- [tp_worker.py](../python/sglang/srt/managers/tp_worker.py)：`TpModelWorker.forward_batch_generation` 构造 `ForwardBatch` 并调用 `ModelRunner.forward`；图中用类名表达桥接关系，未沿用学习文档中简写的 `TpModelWorker.run_batch` 方法名。
- [schedule_batch.py](../python/sglang/srt/managers/schedule_batch.py)：`ScheduleBatch → ForwardBatch` 的数据结构职责。
- [model_runner.py](../python/sglang/srt/model_executor/model_runner.py)：模型前向编排。
- [detokenizer_manager.py](../python/sglang/srt/managers/detokenizer_manager.py)：接收调度器输出、反分词后发送给 TokenizerManager。
- [radix_cache.py](../python/sglang/srt/mem_cache/radix_cache.py)：匹配最长可复用 token 前缀；实际复用还受缓存命名空间、模型/适配器状态与缓存可用性等约束。
- [mem_cache/README.md](../python/sglang/srt/mem_cache/README.md)：缓存节点索引与 KV 物理存储分工。
- [Python 目录说明](../python/sglang/README.md)、[算子库](../python/sglang/kernels/README.md)、[Diffusion](../python/sglang/multimodal_gen/README.md)、[Model Gateway](../sgl-model-gateway/README.md)：仓库子系统边界。

## 最终生成提示词

```text
Use case: infographic-diagram
Asset type: a finished Chinese educational infographic for the SGLang repository, based on its LEARN.md and verified local source code.
Primary request: Create ONE exceptionally clear, polished, illustrated Chinese science-explainer poster about SGLang, explaining a request's journey, why inference is fast, and how the repository is organized. This must be a real, fully rendered raster illustration including all typography, not a webpage, mockup, screenshot, or code diagram.

Canvas and style:
Landscape 3:2, high resolution, preferably 3072 × 2048. Editorial technology explainer with generous whitespace, crisp ink-like outlines, restrained shallow-isometric GPU and token illustrations, and beautifully typeset Simplified Chinese sans-serif text. Cream-white paper background, near-black text, SGLang's existing burnt-orange accent (#D55816); muted blue/teal distinguish requests and cache data. Typography remains flat and perfectly readable. Build a coherent illustrated process, not a wall of text or a slide template. No mascots or invented corporate logos. Strong hierarchy: large headline, dominant process illustration, three compact explanation panels, compact repository map, reading-route footer. Thin panel boundaries and subtle tinted process zones. Use arrows only for actual dataflow; no arrow tangles.

All Chinese and English labels below are verbatim text to render. Do not add extra claims or text. Do not render the explanatory layout instructions. Prioritize large legible text over tiny decorative detail.

HEADER, about 11% of canvas:
Large wordmark text "SGLang", paired with the main Chinese headline "大模型回答背后的高效流水线".
One-line subtitle: "把模型部署成服务，让提问更快得到回答、更多请求同时运行。"

MAIN ILLUSTRATION, about 43% of canvas:
Section heading "01  一次提问，如何变成回答？"
Small subtitle "SRT 典型三进程示意".
Draw exactly three clearly bounded process zones across the width. Make the central zone substantially wider because it contains the scheduler AND the GPU execution chain. The client is a small chat terminal outside these zones, visually adjacent to the first zone; it is not another process. Draw client request entering the first zone, and returning response exiting the first zone back to the client. Put "用户提问" on the input and "SSE 流式 / JSON" on the client response.

First process zone, labels:
"① 接待与分词"
"主进程"
"HTTP / Engine"
"TokenizerManager"
"文本 → token ID"
Illustrate a chat bubble becoming a short sequence of small token tiles. Do not imply a Chinese character must equal a token.
A clear rightward arrow labeled "ZMQ" leads from TokenizerManager to the central scheduler.

Central process zone, labels:
"② 调度与计算"
"Scheduler 子进程"
An ordered chain entirely INSIDE this zone:
"Scheduler" with a small queue/batch illustration and caption "排队 · 组批";
then "TpModelWorker" as a clearly visible bridge;
then "ModelRunner" on an illustrated GPU computation module, with caption "模型前向 → 采样".
A subordinate annotation on the worker bridge reads "ScheduleBatch → ForwardBatch".
Place a small memory stack labeled "KV Cache" next to and connected bidirectionally with GPU execution, remaining inside the central process zone.
Under the computation chain, two compact phase labels:
"Prefill：处理提示词"
"Decode：逐步生成 token"
A small loop at Decode indicates repeated generation, labeled "循环至结束".
Make it explicit visually that Scheduler delegates THROUGH TpModelWorker into ModelRunner. The GPU worker is part of this central process zone, not a fourth IPC process.
After computation/iteration, a rightward arrow labeled "ZMQ · token ID" goes from the central zone to the third zone.

Third process zone, labels:
"③ 拼回文字"
"Detokenizer 子进程"
"DetokenizerManager"
"token ID → 文本"
Illustrate token tiles becoming a readable speech bubble.
From the bottom of this zone draw one distinct return arrow running underneath the process zones, pointing LEFT and ending precisely at TokenizerManager in the FIRST zone. Label this return arrow "ZMQ 回主进程". The response then leaves the first zone for the client. This return path must be unmistakable; no direct Detokenizer-to-client arrow.

SMALL EXPLANATION BAND, about 23% of canvas:
Heading "02  为什么快？"
Three equal, airy panels, each with ONE useful explanatory mini-illustration.
Panel 1 title "连续批处理".
Show two successive batch rows: first "A", "B", "C"; then "B", "C", "D". Consistent colors identify B and C across both rows. A finishes and disappears, D joins; small labels "完成" and "加入" point at A and D.
Caption "完成即离队，新请求补位".

Panel 2 title "RadixAttention 前缀复用".
Show a prefix-tree motif: ONE shared block labeled "公共前缀" branching into distinct suffixes labeled "问题 A" and "问题 B"; connect the shared prefix to one stack labeled "KV". This illustrates reuse of earlier computation, not copied answers.
Caption "相同 token 前缀复用 KV".
Small second line "KV = 注意力计算的中间状态".

Panel 3 title "计算与调度重叠".
Show two aligned time lanes labeled "CPU" and "GPU"; at the same time CPU does "准备 B" while GPU does "计算 A", then CPU does "准备 C" while GPU does "计算 B". Two clear columns, left to right.
Caption "CPU 准备下一批，GPU 计算当前批".
Small second line "高效算子 · CUDA Graph".

REPOSITORY BAND, about 15% of canvas:
Heading "03  仓库地图：先学 SRT，再按需展开".
Five compact equal tiles on ONE horizontal row, each with a simple related icon, big name and one description. They are repository subsystems, NOT consecutive request stages; do not connect them into a sequential pipeline.
Tile 1: "SRT" / "LLM / VLM 推理" / "srt/"
Tile 2: "Kernels" / "共享底层算子" / "kernels/"
Tile 3: "Diffusion" / "图像 / 视频生成" / "multimodal_gen/"
Tile 4: "Model Gateway" / "集群路由与负载均衡" / "sgl-model-gateway/"
Tile 5: "Rust" / "服务组件改写中" / "rust/"
A concise note below: "SRT 与 Diffusion 共享算子，各用自己的 runtime。"
Highlight SRT with the existing orange accent, while the other tiles are quieter.

FOOTER, about 8%:
A thin reading-route band:
"读代码：Engine → TokenizerManager → Scheduler → TpModelWorker → ModelRunner → 回包"
"先跟通一个请求，再读缓存与调度。"
Small source line "依据 LEARN.md 与当前仓库源码 · 示意省略并行副本与可选分支"

Technical invariants:
The poster is specifically the ordinary autoregressive SRT serving path, a simplified topology; it is not a universal statement about process count. HTTP, Engine, TokenizerManager share the main process. Scheduler and its worker/model runner stay in one child process in this simplified drawing. Detokenizer runs in another child process. ZMQ connects these processes. The detokenizer returns text to TokenizerManager before HTTP sends it to the client. Scheduler does not directly call ModelRunner. KV stores model attention intermediate state, not model weights and not final answers. Prefix reuse concerns matching token prefixes in a compatible cache context. Prefill processes the prompt and Decode generates subsequent tokens iteratively. Continuous batching admits new work as requests finish. Diffusion has its own runtime and does not run through the SRT Scheduler. Gateway routes between model workers at cluster level and is optional. Do not add numerical speedup claims, performance charts, code-volume statistics, training steps, external cloud-provider logos, or a deprecated DSL-first story.

Final quality:
Carefully typeset exact Chinese characters, correct SGLang capitalization, readable long names such as DetokenizerManager and TpModelWorker, consistent numbered headers, enough margins, no cropped text, no pseudo-writing, no watermarks. Keep diagram arrows well separated from text. The result should be suitable to share as an attractive finished educational poster.
```

