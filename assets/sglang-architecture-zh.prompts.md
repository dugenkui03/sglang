# SGLang 详细架构科普图：完整提示词

生成方式：内置 `image_gen`。三张独立成图，采用相同的中文技术图解风格；局部修订也使用内置图像编辑工具。最终图片及说明见 [图解索引](sglang-architecture-zh.md)。

## 01 项目架构总览：生成

```text
Use case: infographic-diagram.
Create a finished, detailed Simplified Chinese educational architecture infographic for the SGLang source repository. This is a technical explainer that must teach module responsibilities and relationships through clear diagrams and concise explanatory text. Use a landscape 3:2 canvas, high-resolution native raster output. Use crisp flat editorial diagrams, cream-white background, dark navy text, SGLang burnt orange (#D55816) for titles and SRT, muted blue for request arrows, teal for memory/shared compute, restrained lavender for Diffusion. Use small functional icons only. Chinese text must be accurate, sharply readable, and set in a modern sans-serif. English identifiers must retain exact spelling. No ornamental slogan, no mascots, no invented logo, no speedup numbers. Prefer generous space for informative nodes and arrows over decoration. All labels in quotes below are literal text; layout instructions are not image text.
This is page "01 / 03", the MAIN PROJECT ARCHITECTURE OVERVIEW. The user rejected a shallow request-flow poster and wants a thorough explanation of the project and its architecture. Create a new composition that communicates the whole project, with connected layers and explicit subsystem boundaries.

HEADER, about 13%:
Title "SGLang 项目与架构全景".
Subtitle "高性能模型推理与服务框架：把已训练模型接入应用，管理请求、计算与缓存。"
A compact explanatory sentence "模型提供参数与结构；SGLang 负责接入、调度、执行和资源管理。"
A tiny legend "实线：请求 / 执行方向    虚线：可选接入 / 支撑关系".

MAIN CONNECTED ARCHITECTURE, about 65%:
Use a clear layered diagram with three vertical domains. On the left, a narrower optional cluster gateway domain. In the middle, a large SRT domain. On the right, a distinct Diffusion domain. SRT and Diffusion both connect DOWN to one shared kernel foundation. Every package node must have its readable responsibility caption. This is a LOGICAL architecture map, not an OS-process diagram.

At top, a full-width application strip:
"应用与调用方"
"对话 · RAG · Agent · 批量推理 · RL rollout · 图像 / 视频生成"

LEFT optional gateway domain:
Label "集群接入（可选）"
Box "Model Gateway"
Path "sgl-model-gateway/"
Description "Rust 集群路由与管理"
Inside this domain show two small sub-boxes:
"控制面：注册 · 健康检查 · 服务发现"
"数据面：负载均衡 · 缓存感知 · 重试"
A dashed request arrow from application strip to Gateway, then dashed arrow from Gateway to the NETWORK entry of SRT, labeled "选择服务实例".
Also show a separate direct application arrow to SRT network entry, labeled "直连".
The gateway is optional and must NOT lie on the local Python Engine path.
A small cluster motif under gateway shows "实例 A" and "实例 B" as separately selectable service endpoints. It is a side illustration of the gateway's role, not a serial stage in SRT computation.

MIDDLE, larger outlined SRT domain:
Header "SRT：自回归 LLM / VLM 推理"
Path "python/sglang/srt/"
Inside its top edge, two parallel entry pills:
"在线：sglang serve · HTTP / gRPC"
"离线：Python Engine"
The application strip connects to both entries independently; Gateway connects only to the online entry.

Below entries, a vertically connected logical stack:
1. "entrypoints/" with "接口适配：OpenAI 等协议 → 内部请求".
2. "managers/" with "TokenizerManager：分词与请求生命周期" and "Scheduler：排队、组批、推进生成".
3. "model_executor/" with "TpModelWorker 桥接 → ModelRunner 编排".
IMPORTANT placement: TpModelWorker physically belongs to managers/, so draw it as a thin bridge crossing from managers/ to model_executor/, with small filename "managers/tp_worker.py". Do not place its file in model_executor/.
4. "models/ + layers/" with "模型网络 · Attention · MoE · 采样".

Along the side of this SRT stack, show two support boxes, attached by dashed lines to appropriate core layers:
"mem_cache/" / "KV / SSM 状态、前缀复用、内存槽位"
"distributed/" / "TP · PP · DP · EP 并行"
Place a narrow SRT-only config ribbon "配置：ServerArgs → publish → RuntimeContext". Do not apply this runtime-context ribbon to Diffusion.
A small note within SRT "图像输入预处理在 srt/multimodal/".
The model and layers box points DOWN to shared Kernels.
A small source-input node "模型配置与权重" points by a dependency line to the SRT model/executor side and is captioned "configs/ · model_loader/". Avoid confusing model weights with KV cache.

RIGHT, distinct outlined Diffusion domain:
Header "Diffusion：图像 / 视频 / 音频生成"
Path "python/sglang/multimodal_gen/"
Entry pill "sglang generate · DiffGenerator · HTTP API".
A request arrow from the application strip goes directly to this entry.
Inside, a simple vertical chain:
"独立 Scheduler + GPUWorker"
"pipelines_core/" / "编码 → 去噪 → 解码"
"runtime/models + layers"
Use a looping arrow only around the denoising stage to imply iterative generation.
A clear note "拥有自己的 runtime 与调度器".
Connect its models/layers DOWN to the shared Kernels foundation.
Do NOT connect Diffusion request flow into SRT Scheduler or SRT model_executor.

FOUNDATION spanning SRT and Diffusion:
One broad teal bar "共享算子库 Kernels"
"python/sglang/kernels/ · sglang.kernels.ops.*"
"AOT / JIT · CUDA / Triton · FlashInfer 等"
Both runtime model/layer stacks point to this bar. Beneath it one shared hardware bar:
"计算硬件：GPU · CPU · NPU 等"
"具体能力取决于模型、算子和设备后端"

BOTTOM SUPPORT BAND, about 18%:
Left broad block "Rust 服务组件（演进中）" with path "rust/".
Three compact entries "sglang-server：HTTP / tokenizer", "sglang-grpc：gRPC 服务桥接", "sglang-mm：多模态预处理".
Keep this block separate from Model Gateway. It is another repository subsystem, not a mandatory fourth stage of SRT execution.
Right block "工程支撑" with "test/：正确性与回归" and "benchmark/：吞吐与延迟" and "docs/ · scripts/ · CI：文档与工程流程".
At bottom, a concise key architectural statement:
"SRT 与 Diffusion 共用底层算子，各自管理请求和执行流程。"
Small source/navigation footer:
"依据 LEARN.md 与当前源码  ·  02 展开 SRT 内部  ·  03 展开缓存与分布式"

Technical constraints:
SGLang is an inference/serving framework; depict inference over pretrained model weights, not a new foundation model or training optimizer. VLM understanding in SRT and image/video generation in Diffusion must be distinguishable. Model Gateway is a separate optional network routing layer; Python Engine can run locally. Each SRT instance has scheduling, model execution and memory management. Shared kernels underlie both runtimes. Some operations can have native backend paths: shared foundation is conceptual, not a claim all calls use one exact kernel. The Rust crates are evolving service implementations; do not depict the whole project as Rust. Paths in the figure are repository paths and must be spelled correctly. Do not add deprecated lang/ as the main entry. Do not turn the five subsystems into five sequential boxes; show actual architectural dependencies. No decorative performance claims. Render all meaningful package captions, readable enough to learn from.
```

## 01 项目架构总览：局部校对

编辑目标为总览初稿。修正 Diffusion 流程的循环范围及组件说明，保持其余布局。

```text
Use case: infographic-diagram.
Create a finished, detailed Simplified Chinese educational architecture infographic for the SGLang source repository. This is a technical explainer that must teach module responsibilities and relationships through clear diagrams and concise explanatory text. Use a landscape 3:2 canvas, high-resolution native raster output. Use crisp flat editorial diagrams, cream-white background, dark navy text, SGLang burnt orange (#D55816) for titles and SRT, muted blue for request arrows, teal for memory/shared compute, restrained lavender for Diffusion. Use small functional icons only. Chinese text must be accurate, sharply readable, and set in a modern sans-serif. English identifiers must retain exact spelling. No ornamental slogan, no mascots, no invented logo, no speedup numbers. Prefer generous space for informative nodes and arrows over decoration. All labels in quotes below are literal text; layout instructions are not image text.
Input image 1 is the edit target: the completed page 01 SGLang project architecture overview.
Make ONE local correction inside the right-hand Diffusion domain only. Preserve every other part of the poster pixel-for-pixel as far as possible, including all SRT, Gateway, Rust, shared-kernel, hardware and engineering-support content. Preserve dimensions, typography, title, colors, and domain boundaries.

In the Diffusion "pipelines_core/" box, correct the three-stage mini-diagram:
Draw "编码（条件处理）" → "去噪（迭代生成）" → "解码（生成结果）" as a simple LEFT-TO-RIGHT chain.
The only cyclic arrow is a small self-loop that begins and ends at the MIDDLE denoising box. It must not wrap around or touch encoding or decoding. Encoding and decoding do not repeat in each ordinary denoising iteration.
Keep the existing heading "pipelines_core/" and the caption "编码 → 去噪 → 解码".

In the lower "runtime/models + layers" box of the SAME Diffusion domain, replace the detail line "扩散模型网络 · UNet · VAE · 采样器等" with this exact text:
"编码器 · 去噪模型 · 解码器等组件"
This avoids unsupported specificity and accurately describes the component roles.
Keep everything else unchanged. No new text, slogans, extra stages or decorations.
```

## 02 SRT 内部：生成

```text
Use case: infographic-diagram.
Create a finished, detailed Simplified Chinese educational architecture infographic for the SGLang source repository. This is a technical explainer that must teach module responsibilities and relationships through clear diagrams and concise explanatory text. Use a landscape 3:2 canvas, high-resolution native raster output. Use crisp flat editorial diagrams, cream-white background, dark navy text, SGLang burnt orange (#D55816) for titles and SRT, muted blue for request arrows, teal for memory/shared compute, restrained lavender for Diffusion. Use small functional icons only. Chinese text must be accurate, sharply readable, and set in a modern sans-serif. English identifiers must retain exact spelling. No ornamental slogan, no mascots, no invented logo, no speedup numbers. Prefer generous space for informative nodes and arrows over decoration. All labels in quotes below are literal text; layout instructions are not image text.
This is page "02 / 03", a deep architecture explanation of the SRT inference engine. It must contain substantially more meaningful internal detail than a simple tokenizer → GPU → text illustration. Keep visual continuity with the series: orange numbered headings, cream background, blue request arrows, teal memory. The MAIN SUBJECT is the relationship between request state, CPU batch planning, GPU execution, and response delivery.

HEADER, about 12%:
Title "SRT 内部：一次请求怎样被执行"
Subtitle "从协议适配到逐步生成：请求状态、批次计划与 GPU 张量各有分工。"
A thin, separate startup/configuration ribbon:
"启动与配置：sglang serve → ServerArgs → 启动各进程"
"publish → RuntimeContext · get_schedule() / get_memory() / get_exec()"
Startup/config arrows must be visually distinct from the per-request dataflow.

DOMINANT PROCESS DIAGRAM, about 57%:
Label "基础进程拓扑：主进程 + Scheduler 子进程 + Detokenizer 子进程".
Three bounded process zones, with the center substantially larger. A small client icon sits outside the left zone. The right-to-left text return path is explicit and unobstructed.

LEFT zone "进程 A · 主进程":
Online input arrow from "客户端" into "HTTP / OpenAI 协议适配".
Small filename "entrypoints/openai/serving_chat.py".
Caption "校验 · chat template · 构造内部请求".
Down arrow labeled "GenerateReqInput" into:
"TokenizerManager"
"generate_request"
"分词 · 输入预处理 · 请求状态管理".
Small filename "managers/tokenizer_manager.py".
A small separate side entry "离线 Python Engine" connects DIRECTLY to TokenizerManager rather than through HTTP.
A blue ZMQ arrow from TokenizerManager to the CENTRAL scheduler is labeled "ZMQ · token IDs / 请求元数据".
A nearby small note "IPC 消息定义：managers/io_struct.py".
Keep return delivery visibly connected to the same TokenizerManager and then back through HTTP to the client, labeled "SSE 流式 / JSON".

CENTER zone "进程 B · Scheduler 与模型执行":
At the top, a CPU-control group:
"Scheduler" / "managers/scheduler.py".
Show "waiting 队列" feeding a batch-selection node alongside "running 批次".
The selection node is labeled "get_next_batch_to_run".
Resulting plan node "ScheduleBatch" with caption "Req 请求状态 · CPU 调度数据为主".
Do not claim it contains no GPU tensors.
A clearly visible bridge below:
"TpModelWorker.forward_batch_generation"
"managers/tp_worker.py".
It converts the plan with a small annotation "ForwardBatch.init_new" to:
"ForwardBatch" / "GPU tensor 为主：输入、位置、序列长度、KV 槽位".
The bridge MUST separate Scheduler from ModelRunner.

Below, a large GPU-execution container:
"ModelRunner.forward" / "model_executor/model_runner.py".
Caption "选择 eager / CUDA Graph，编排一次模型执行".
Inside this container draw a model network block:
"model.forward" / "models/ + layers/".
Inside the model block use two compact sub-blocks:
"Attention backend" and "FFN / MoE 等网络层".
Connect the Attention backend BIDIRECTIONALLY to a teal side memory block "KV / SSM 状态" with path "mem_cache/".
A small adapter label next to Attention says "RadixAttention".
The model block yields a downward arrow labeled "logits：候选 token 分数" into a separate sibling sampling block "Sampler" / "layers/sampler.py" / "按采样参数选下一个 token".
IMPORTANT: End the ModelRunner.forward container after the model's logits output. Draw Sampler BELOW and OUTSIDE that forward container, still within process B. Sampler is a worker execution stage following ModelRunner.forward, not a layer inside the neural model. Never put sampling inside model.forward or ModelRunner.forward. The worker invokes a separate ModelRunner.sample operation.

After the worker execution, show a CPU result-handling node "process_batch_result" with caption "追加 token · 更新请求状态 · 判断结束".
A loop from this result node goes BACK to the Scheduler/running-batch area, labeled "未结束：进入后续生成步".
A separate outgoing arrow from this result-handling node goes to the RIGHT process, labeled "ZMQ · 输出 token IDs".
This illustrates incremental emission while generation can continue, not only one final message after all tokens are generated.

RIGHT zone "进程 C · 文本还原":
"DetokenizerManager"
"managers/detokenizer_manager.py"
"输出 token IDs → 文本片段".
An unmistakable return arrow from here goes LEFT underneath the process diagram and terminates at TokenizerManager in process A, labeled "ZMQ 回主进程".
Response then flows from process A to the client. Do not draw a direct Detokenizer-to-client arrow.
Small note below main diagram:
"开启 DP 时，可在 TokenizerManager 与 Scheduler 之间加入 DataParallelController；并行部署会增加进程数。"

THREE EXPLANATORY PANELS, about 18%:
Panel "Prefill 与 Decode".
A prompt-block illustration feeding cached context, then a repeated next-token loop.
Text "Prefill：处理提示词，建立 KV，可产生首个 token"
Text "Decode：利用已有 KV，逐步续写并更新状态".
Show Prefill once followed by repeated Decode steps for the ordinary autoregressive path.

Panel "连续批处理与分块 Prefill".
Show batches "A B C" → "B C D": A finishes, D enters, B and C persist.
Text "完成请求退出，新请求加入后续批次".
Below, show a long prompt broken into 3 chunks.
Text "长提示词分块，控制单次计算预算".

Panel "让 CPU 与 GPU 重叠工作".
Two time lanes: "CPU：准备 B | 准备 C"; "GPU：计算 A | 计算 B".
Text "调度准备与模型计算可重叠".
Text "CUDA Graph：减少重复的 CPU 发射开销".

BOTTOM OPTIONAL-FEATURE STRIP, about 10%:
Title "按需挂载的能力".
Four small parallel entries with directory and concrete role:
"speculative/" / "草稿候选 → 目标模型校验"
"constrained/" / "约束下一 token 的合法集合"
"lora/" / "基座模型 + 可选适配器"
"function_call/" / "解析工具调用字段"
Do not connect these four features as mandatory serial stages.
Small footer "依据 LEARN.md 与当前源码 · 普通自回归生成主线；特性与硬件会改变具体执行路径".

Technical invariants:
This is the basic single-instance ordinary SRT topology, with CPU scheduler and worker orchestration co-located in the scheduler subprocess. Additional TP/PP/DP ranks or optional services can increase process count. HTTP and Engine share the management machinery; offline Engine bypasses HTTP. Scheduler calls the worker, the worker constructs ForwardBatch and invokes model_runner.forward; the worker then calls model_runner.sample which uses Sampler. It is okay to simplify the call stack visually as worker execution but keep the model.forward neural network boundary precise. ScheduleBatch is mostly high-level CPU state; ForwardBatch predominantly tensor inputs, not a separate OS process. Attention accesses runtime KV memory. Model weights and KV are not the same. The output loop and return path must remain coherent. Do not turn a feature option into a required step. Carefully render long identifiers; never crop or shrink the main process diagram to make room for decoration.
```

## 02 SRT 内部：请求数据流连线校对

```text
Use case: infographic-diagram.
Input image 1 is the edit target, the existing page 02/03 SRT architecture infographic.
Make a precise DATAFLOW-ARROW CORRECTION in the main three-process diagram only. Keep all header, configuration ribbon, process boundaries, module boxes, typography, colors, bottom three mechanism panels, optional-feature strip, and all text unchanged except repositioning the small "ForwardBatch.init_new" label onto its correct conversion edge. Preserve image dimensions and overall composition. You may reroute arrows in the whitespace and minimally adjust spacing inside the main diagram to avoid overlaps. Do not redesign the poster.

The original image has several arrow endpoints on the wrong nodes. Render the following EXACT connectivity so readers cannot infer any shortcuts:

1. REQUEST IPC: The blue arrow leaving TokenizerManager on the left MUST enter the Scheduler CPU control box, near the waiting queue, at the TOP of process B. Route it upward along the gap between process A and B before entering Scheduler. Preserve its "ZMQ · token IDs / 请求元数据" label. REMOVE the existing shortcut arrow from TokenizerManager straight into ModelRunner.forward. Absolutely no TokenizerManager → ModelRunner edge.

2. BATCH BRIDGE: The solid data path is:
ScheduleBatch → TpModelWorker.forward_batch_generation → ForwardBatch → ModelRunner.forward.
Connect ScheduleBatch into the worker bridge. The worker-to-ForwardBatch edge is where "ForwardBatch.init_new" belongs. Connect the ForwardBatch box's bottom edge clearly down into the top edge of ModelRunner.forward. Remove the direct worker-to-ModelRunner downward shortcut if it makes the ForwardBatch seem bypassed. Keep the worker, ForwardBatch and ModelRunner node positions close to their original positions.

3. ATTENTION MEMORY: The teal bidirectional KV link MUST connect the "Attention backend" sub-box DIRECTLY with "KV / SSM 状态". It must NOT start on the FFN/MoE sub-box or merely the general model rectangle. Route a clean elbow line above the FFN/MoE box if necessary. Remove the current misleading link to FFN/MoE. The two neural-network component boxes can remain as labeled parts without arrows between them; do not draw reverse-computation arrows from FFN back to Attention.

4. OUTPUT IPC: The blue arrow leaving process_batch_result MUST enter the actual DetokenizerManager box in process C, preferably its left input edge. Route it up the LEFT gutter of process C to that box. Preserve the "ZMQ · 输出 token IDs" label. It must not join the outgoing text-return line partway down the process C column.
The DetokenizerManager's separate OUTPUT line must leave its bottom edge, run down the right-side whitespace, then left beneath the main diagram to enter TokenizerManager. Preserve "ZMQ 回主进程（文本片段）". Keep the two directions visibly separate and never merge them.

Keep the existing result-to-Scheduler generation loop and the TokenizerManager-to-client SSE / JSON return. Keep model logits → Sampler → process_batch_result unchanged. All four corrections concern one thing: faithful arrow endpoints in the main request dataflow. No extra content. Final diagram must be clean, highly legible, and every arrow must visibly end at its intended module.
```

## 02 SRT 内部：桥接与 Attention 缓存连线终校

```text
Use case: infographic-diagram.
Edit target: the supplied page 02 SRT architecture infographic. Make ONLY these two local corrections in the central process B. Preserve all other content, layout, boundaries, colors, text, client/IPC paths, footer panels, and image dimensions.

BATCH BRIDGE:
The existing ScheduleBatch box currently has downward arrows incorrectly entering ForwardBatch directly. ERASE EVERY ScheduleBatch → ForwardBatch edge.
Instead draw ONE clear elbow arrow from the BOTTOM of ScheduleBatch, down into the whitespace above the TpModelWorker box, then LEFT through that whitespace, then DOWN into the TOP of the TpModelWorker.forward_batch_generation box.
Keep the existing TpModelWorker → ForwardBatch rightward arrow.
Keep the existing ForwardBatch → ModelRunner.forward downward arrow.
Put the label "ForwardBatch.init_new" above the worker-to-ForwardBatch edge.
The only chain is ScheduleBatch → TpModelWorker → ForwardBatch → ModelRunner. TpModelWorker must visibly have an incoming arrow. There must be ZERO direct arrows from ScheduleBatch into ForwardBatch. Do not leave any orphaned arrowheads.

MODEL COMPONENT INSET:
Inside the dotted "模型网络 / model.forward" inset, rearrange the TWO component boxes:
Place "FFN / MoE 等网络层" as the LEFT component box.
Place "Attention backend" with small label "RadixAttention" as the RIGHT component box nearest the external KV memory box.
REMOVE ALL arrows between these two component boxes. They are shown as parts of the neural network, with no internal execution order being illustrated.
Now draw one SHORT DIRECT HORIZONTAL bidirectional teal arrow between the RIGHT Attention backend box and the "KV / SSM 状态" box.
This must visibly start on the Attention backend edge and visibly end on the KV box edge. Absolutely no KV link to FFN/MoE.
Keep the enclosing model.forward rectangle, ModelRunner.forward container, logits output, Sampler, process_batch_result, and generation loop unchanged.

This is a minimal correction of exactly two local areas. Do not change any unrelated labels or arrows. Check the visible arrow endpoints before returning.
```

## 03 KV 缓存与分布式：生成

```text
Use case: infographic-diagram.
Create a finished, detailed Simplified Chinese educational architecture infographic for the SGLang source repository. This is a technical explainer that must teach module responsibilities and relationships through clear diagrams and concise explanatory text. Use a landscape 3:2 canvas, high-resolution native raster output. Use crisp flat editorial diagrams, cream-white background, dark navy text, SGLang burnt orange (#D55816) for titles and SRT, muted blue for request arrows, teal for memory/shared compute, restrained lavender for Diffusion. Use small functional icons only. Chinese text must be accurate, sharply readable, and set in a modern sans-serif. English identifiers must retain exact spelling. No ornamental slogan, no mascots, no invented logo, no speedup numbers. Prefer generous space for informative nodes and arrows over decoration. All labels in quotes below are literal text; layout instructions are not image text.
This is page "03 / 03", a detailed educational architecture diagram of KV caching and distributed serving. It completes the series by explaining where state lives and the distinct roles of routing, parallel execution, and stage disaggregation. Use TWO major horizontal sections, memory above and distributed serving below. The diagrams and labels must remain readable.

HEADER, about 9%:
Title "KV 缓存与分布式部署"
Subtitle "复用已经算过的上下文，把请求与模型计算分配到合适的设备。"
Small legend "实线：数据 / 执行流    虚线：索引 / 可选协调".

UPPER MEMORY SECTION, about 47%:
Heading "A  缓存架构：复用逻辑与物理存储各有分工".
Introductory line "KV Cache 保存注意力层的 Key / Value 中间状态；模型权重由模型加载器载入。"

LEFT HALF: logical prefix reuse, title "Radix：决定哪些前缀可复用".
Draw one root block "公共 token 前缀", directly associated with one shared memory block "共享 KV".
From the shared prefix/state, branch into separate suffixes "问题 A" and "问题 B", each with its OWN further block "新增 KV A" and "新增 KV B". Common state must be physically shown once, new states separately. Do not merge the distinct suffixes into one shared answer.
Below the mini-diagram, text:
"匹配最长可复用前缀 → 复用槽位 → 计算新增部分"
"radix_cache.py / unified_radix_cache.py"
"token 前缀及模型、缓存上下文兼容时才可复用".
A small note "混合模型还会维护 SSM、滑窗等状态".

RIGHT HALF: physical allocation, title "内存栈：把请求映射到实际张量".
Draw a compact vertical allocation chain, each node has a path plus a readable responsibility:
"allocation.py" / "每个 batch 的分配策略"
down to "hybrid_cache/" / "多内存池的层映射"
down to "allocator/" / "分配与释放可用槽位"
down to "pool/ · L1 设备内存" / "持有实际 KV / SSM 张量".
From the Radix block in the left half, draw a GRAY DASHED reference line directly to the pool node labeled "索引已有 KV 槽位". This is a logical index relation, not data flowing through Radix as a physical storage tier. Keep the allocation stack a separate axis.

Across the bottom of this upper section, a horizontal optional hierarchical-cache strip:
Title "HiCache 分层缓存（可选）".
Three storage blocks with bidirectional transfer arrows:
"L1 设备内存 · pool/" ↔ "L2 主机内存 · pool_host/" ↔ "L3 外部存储 · storage/".
Caption "按配置在不同层间备份、预取与恢复缓存".
This L1 represents the same device pool as the allocation chain; join with a subtle connector or label "同一设备池", not a fourth extra cache tier.
Small relationship statement "Radix 管复用与淘汰；allocator 管槽位；pool 管数据。"

LOWER DISTRIBUTED SECTION, about 39%:
Heading "B  扩展架构：实例路由、模型并行与 P/D 分离".
Three equally sized explanatory panels, visually separated so readers do not mistake them for a serial pipeline.

PANEL 1 "集群路由：选择服务实例".
A top "应用请求" node points to "Model Gateway（可选）", then branches to two independently selectable service boxes:
"实例 A" / "Tokenizer → Scheduler → Worker"
"实例 B" / "Tokenizer → Scheduler → Worker".
Label the fanout "负载 / 缓存感知".
Important: this is selecting an endpoint for a request, not always broadcasting the same request to all instances.
Caption "网关选实例；Scheduler 在实例内选下一批".
Small path "sgl-model-gateway/".
A separate compact note below:
"实例内 DP：TokenizerManager"
"→ DataParallelController → 调度副本".
Clearly mark this as an OPTIONAL internal dispatch mechanism, distinct from the external Gateway. Do not insert Gateway into every local request path.

PANEL 2 "模型与工作量怎样分到多卡".
Use a clean 2 × 2 micro-diagram grid:
"DP 数据并行" / "模型副本分流请求", with independent requests into two replicas.
"TP 张量并行" / "同一层由多卡协作", with a matrix split across two GPU icons.
"PP 流水线并行" / "网络层分配到不同阶段", with consecutive layer groups across GPUs.
"EP 专家并行" / "MoE 专家分布到多卡", with distinct expert blocks on separate GPUs.
Do not imply DP partitions weight tensors, TP routes unrelated requests, or EP applies to every dense model.
Path "distributed/parallel_state.py".
Caption "并行方式可组合，需匹配模型与配置".

PANEL 3 "P/D 分离：拆开两类计算（可选）".
A small coordinator label "请求入口 / 协调" has dashed control arrows to both worker boxes.
Main data path is:
"Prefill worker" / "处理提示词，建立 KV"
then a thick rightward or downward arrow labeled "KV / 相关状态传输"
then "Decode worker" / "接续上下文，逐步生成".
Draw a self-loop at Decode only, then a response arrow "流式输出".
A small caption "握手与预分配 · 状态传输 · 接续生成".
Path "srt/disaggregation/".
Second caption "按阶段配置算力，并协调状态传输".
Do not depict model weights being transported on the KV transfer arrow; each worker has its model execution capability.

FOOTER, about 5%:
"位置索引：srt/mem_cache/ · srt/distributed/ · srt/disaggregation/ · sgl-model-gateway/"
"依据 LEARN.md 与当前源码 · 主机 / 外部缓存及 P/D 分离按配置启用；部署拓扑随并行配置变化".

Scientific/technical accuracy:
KV consists of intermediate attention states associated with processed tokens, not stored final answers or model weights. Prefix matching also respects cache namespace/context compatibility. Radix is an index and reuse/eviction policy, not a layer inserted between allocator and pool. Host and external cache tiers are optional. For hybrid attention/state models there can be multiple state types and pools. Multi-instance model routing, SRT's internal DP dispatcher, TP/PP/DP/EP parallelism, and prefill/decode disaggregation are distinct architectural concepts and must not collapse into one generic load-balancer icon. TP/PP/EP require rank communication, which this conceptual picture may omit. Gateway has policy choice rather than mandatory broadcast. Do not draw cache storage through Model Gateway. No speedup guarantees, data plots, percentages, or unsupported hardware limits. Preserve readable exact Chinese labels and visually traceable arrow endpoints.
```
