# SGLang Overlap 学习路线与现状

先读 Qwen3_5ForConditionalGeneration 的模型初始化与权重加载，再跟非流式文本生成请求：从 OpenAIServingChat._handle_non_streaming_request 到完整响应；调度只沿 event_loop_overlap，包含共享的 Prefill/Decode、KV、Qwen 3.5 dense 推理、采样与回包。补入 QKV 投影及父类的核心构造与未量化前向方法。

**Overlap 路线阅读覆盖：84/127（66.1%），待覆盖 43 个。**

直接注释 D：68；链路补全 C：11；分支补全 B：5。

| 阅读阶段 | 已覆盖 / 总数 | 待覆盖 | 进度 |
| --- | ---: | ---: | ---: |
| 启动准备：Qwen 3.5 初始化 → 权重加载 | 13/17 | 4 | 76.5% |
| 接收请求 → 分词 → 发给 Scheduler | 9/9 | 0 | 100.0% |
| Overlap 循环 → 收请求 → 入队 | 8/8 | 0 | 100.0% |
| 选择 Prefill 请求 → 匹配前缀 → 建立批次 | 11/15 | 4 | 73.3% |
| 为 Prefill 分配 KV 空间 | 2/4 | 2 | 50.0% |
| 决定本轮重叠 → 解析输入 → 提交前向 | 4/6 | 2 | 66.7% |
| 模型前向 → Qwen 3.5 → logits 与采样 | 18/27 | 9 | 66.7% |
| 保存本批结果 → 交接下一轮输入 | 2/5 | 3 | 40.0% |
| 处理上一批结果 → 结束判断 → 缓存与释放 | 4/13 | 9 | 30.8% |
| 请求未结束 → 准备下一轮 Decode | 2/10 | 8 | 20.0% |
| 输出 token → 解码文本 → 完整 HTTP 响应 | 11/11 | 0 | 100.0% |
| 条件分支：Decode CUDA Graph | 0/2 | 2 | 0.0% |

## 统计口径与阅读约定

路线 v3（2026-10-07）：在 v2 的 121 项基础上补入 4 个 QKV/线性层构造与参数创建方法、2 个线性投影前向方法，共 127 项。继承的 forward 按实际定义只计一次；统计基线与 D/C/B 规则保持不变。分母变化不代表学习退步，也不与全核心总览混算。

- 每个方法等权计 1；范围内每个方法在完整路线中只出现一次。按方法粒度判定，不对方法内部每个条件分支单独计分。类名与方法名分列，模块级函数的类名为 `—`。
- ✅ 已读（D）：源码相对固定基线新增中文学习说明；☑️ 推定（C/B）：按约定补全；⬜ 未读（U）：当前统计未覆盖。未读不等于断言实际没看过。
- 先 D，再一轮链路 C，最后一轮分支 B：同一概念分支至少 2 个 D，且 (D+C)/分支方法数 ≥ 2/3；C/B 不作新端点。只在本路线白名单中独立计算。
- 启动准备单列模型构造与权重加载，完成后再进入请求路线；请求调度只沿 event_loop_overlap，共享方法内只看相应可达分支。AI 文档、图片、聊天次数不计分，也不要求考试。
- 编号是阅读顺序，不是串行调用栈；Prefill/Decode、两类模型层、Eager/CUDA Graph 和结束/未结束请求都是分支。通信跨越进程，后台接收和 GPU 执行可能并发。
- 通常先提交本批，再处理上一批；关闭本轮重叠时提前处理上一批。只有存在 delay_sample_func 时，才在上一批处理后采样本批。

模型固定为 Qwen3_5ForConditionalGeneration（dense）：构造函数在 qwen3_5.py，forward 继承 qwen3_vl.py 的 Qwen3VLForConditionalGeneration；self.model 是 qwen3_5.py 的 Qwen3_5ForCausalLM。qwen3_5_text.py 中的同名类是另一个文本模型入口，不计入这条路线。初始化取 DefaultModelLoader 的常规加载路径；普通 Attention 后端取 FlashAttention，线性注意力跟到 GatedDeltaNet 与 RadixLinearAttention 接口；前缀缓存取基础 RadixCache，KV 槽位取 paged allocator。同类实现不重复扩展分母。QKV 线性投影补读 QKVParallelLinear、ColumnParallelLinear、LinearBase 的构造，以及 UnquantizedLinearMethod 的参数创建和普通 F.linear 前向分支；共用方法只计一次。

排除范围：

- 通用日志、序列化、socket 包装、字符串拼接、简单 getter/setter
- 运维与权重更新 API、LoRA、PD/EPD、推测解码、MoE/EP、复杂并行拓扑
- 视觉编码与多模态融合、独立 Mamba/SWA 专题、GDN 状态池与后端算子、量化和硬件专用变体、底层算子实现
- AI 生成文档、图片及其文件数量不作为直接已读证据
- event_loop_normal 及其独有执行分支；所有展示的共用方法只沿 Overlap 可达分支阅读
- 服务启动与配置发布的其余路径、启动期缓存池初始化和 CUDA Graph capture；本路线的启动准备只包含所列模型构造与权重加载方法
- 非流式处理入口之前的 HTTP 路由与请求转换、原生 generate 和 SSE 响应分支；需要时另外查看全核心总览
- 简单队列/属性操作和 pop_and_process 局部包装函数不单独计数，时序在路线说明中保留
- 启动期权重加载重叠、远程/IPC/其他 loader 变体；本次只补齐常规模型构造和权重加载
- LinearBase.forward 抽象占位、PyTorch nn.Module 通用机制及 F.linear 底层算子实现

## 接下来重点看

按固定优先级从当前未覆盖方法中选择，再按路线顺序展示；学习后重算会自动跳过已覆盖项。序号对应下方完整路线。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 11 | ⬜ 未读 | LinearBase | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L167) | 保存线性层配置并选择 quant_method 计算实现 |
| 13 | ⬜ 未读 | Qwen3_5LinearDecoderLayer | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L795) | 创建 GatedDeltaNet、dense MLP、归一化与层间通信组件 |
| 14 | ⬜ 未读 | Qwen3_5GatedDeltaNet | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L267) | 创建 GDN 输入投影、卷积参数、RadixLinearAttention、门控归一化与输出投影 |
| 15 | ⬜ 未读 | Qwen2MoeMLP | [\_\_init\_\_](python/sglang/srt/models/qwen2_moe.py#L177) | 创建 dense gate/up、SiLU 门控与 down 投影 |
| 71 | ⬜ 未读 | ColumnParallelLinear | [forward](python/sglang/srt/layers/linear.py#L492) | QKVParallelLinear 继承的前向入口，调用 quant_method.apply 执行投影 |
| 72 | ⬜ 未读 | UnquantizedLinearMethod | [apply](python/sglang/srt/layers/quantization/unquant.py#L255) | 未量化线性计算的普通分支，以 F.linear 执行输入与权重的投影 |
| 77 | ⬜ 未读 | Qwen3_5LinearDecoderLayer | [forward](python/sglang/srt/models/qwen3_5.py#L875) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 |
| 78 | ⬜ 未读 | Qwen3_5GatedDeltaNet | [forward](python/sglang/srt/models/qwen3_5.py#L708) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 |
| 79 | ⬜ 未读 | RadixLinearAttention | [forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 |
| 80 | ⬜ 未读 | Qwen2MoeMLP | [forward](python/sglang/srt/models/qwen2_moe.py#L249) | Qwen 3.5 dense 复用的门控前馈网络 |

## 启动准备：Qwen 3.5 初始化 → 权重加载

**已覆盖 13/17（76.5%），待覆盖 4。**

发生在请求执行前，一次初始化服务后续批次。DefaultModelLoader.load_model 先调用 _initialize_model 构造模型树，构造返回后再加载权重。入口子类调用父类并注入文本主干；主干按 layers_block_type 创建两类解码层。编号是阅读顺序，两类层和共用 MLP 不是串行调用。详细对象关系见 [LEARN.md 第 3.1 节](LEARN.md#31-固定模型qwen3_5forconditionalgeneration)。QKV 投影沿继承链初始化，父类构造返回后才创建权重参数；两条调用链分开判定。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 1 | ✅ 已读 | ModelRunner | [load\_model](python/sglang/srt/model_executor/model_runner.py#L1163) | 启动期加载模型并保存 loader 与外层模型实例 |
| 2 | ✅ 已读 | — | [load\_model\_with\_memory\_saver](python/sglang/srt/model_executor/model_runner_components/load_model_utils.py#L265) | 在权重内存区域中选择 loader 并加载模型 |
| 3 | ✅ 已读 | DefaultModelLoader | [load\_model](python/sglang/srt/model_loader/loader.py#L962) | 先创建模型结构，再加载权重并进入 eval 模式 |
| 4 | ✅ 已读 | — | [\_initialize\_model](python/sglang/srt/model_loader/loader.py#L271) | 按模型架构解析具体类并调用构造函数 |
| 5 | ✅ 已读 | Qwen3_5ForConditionalGeneration | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1914) | 注入 Qwen3_5ForCausalLM，调用父类构造并设置位置编码信息 |
| 6 | ✅ 已读 | Qwen3VLForConditionalGeneration | [\_\_init\_\_](python/sglang/srt/models/qwen3_vl.py#L1260) | 创建文本主干、LM head、LogitsProcessor 与可选视觉模块 |
| 7 | ✅ 已读 | Qwen3_5ForCausalLM | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1442) | 创建 embedding、按配置选择两类解码层并创建最终归一化 |
| 8 | ☑️ 推定 · C | Qwen3_5AttentionDecoderLayer | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L938) | 创建 QKV 投影、RoPE、RadixAttention、dense MLP 与归一化 |
| 9 | ☑️ 推定 · C | QKVParallelLinear | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L970) | 确定 QKV 合并投影的维度、头划分与权重布局 |
| 10 | ☑️ 推定 · C | ColumnParallelLinear | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L334) | 配置列并行线性层，创建权重并绑定加载回调 |
| 11 | ⬜ 未读 | LinearBase | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L167) | 保存线性层配置并选择 quant_method 计算实现 |
| 12 | ✅ 已读 | UnquantizedLinearMethod | [create\_weights](python/sglang/srt/layers/quantization/unquant.py#L228) | 创建并注册未量化线性层的 weight 参数及加载属性 |
| 13 | ⬜ 未读 | Qwen3_5LinearDecoderLayer | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L795) | 创建 GatedDeltaNet、dense MLP、归一化与层间通信组件 |
| 14 | ⬜ 未读 | Qwen3_5GatedDeltaNet | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L267) | 创建 GDN 输入投影、卷积参数、RadixLinearAttention、门控归一化与输出投影 |
| 15 | ⬜ 未读 | Qwen2MoeMLP | [\_\_init\_\_](python/sglang/srt/models/qwen2_moe.py#L177) | 创建 dense gate/up、SiLU 门控与 down 投影 |
| 16 | ✅ 已读 | DefaultModelLoader | [load\_weights\_and\_postprocess](python/sglang/srt/model_loader/loader.py#L1014) | 调用模型的 load_weights 并执行加载后处理 |
| 17 | ☑️ 推定 · B | Qwen3_5ForConditionalGeneration | [load\_weights](python/sglang/srt/models/qwen3_5.py#L1972) | 映射 checkpoint 名称，将 QKV、MLP 和 GDN 权重装入参数 |

## 接收请求 → 分词 → 发给 Scheduler

**已覆盖 9/9（100.0%），待覆盖 0。**

以单个非流式生成请求为主；generate_request 调用 _wait_one_response 后挂起，结果到达时才继续 yield。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 18 | ✅ 已读 | OpenAIServingChat | [\_handle\_non\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1826) | 收集完整结果并构造 chat 响应 |
| 19 | ✅ 已读 | TokenizerManager | [generate\_request](python/sglang/srt/managers/tokenizer_manager.py#L850) | 统一请求生命周期入口 |
| 20 | ✅ 已读 | TokenizerManager | [\_init\_req\_state](python/sglang/srt/managers/tokenizer_manager.py#L3582) | 以 rid 登记请求状态 |
| 21 | ✅ 已读 | TokenizerManager | [\_tokenize\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1069) | 构造单个 tokenized 请求 |
| 22 | ✅ 已读 | TokenizerManager | [\_tokenize\_texts](python/sglang/srt/managers/tokenizer_manager.py#L995) | 文本与 token 输入的分词路径 |
| 23 | ✅ 已读 | TokenizerManager | [\_validate\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1283) | 上下文长度与生成预算检查 |
| 24 | ✅ 已读 | SamplingParams | [normalize](python/sglang/srt/sampling/sampling_params.py#L218) | 将生成参数规范化为采样语义 |
| 25 | ✅ 已读 | TokenizerManager | [\_create\_tokenized\_object](python/sglang/srt/managers/tokenizer_manager.py#L1462) | 生成跨进程请求对象 |
| 26 | ✅ 已读 | TokenizerManager | [\_send\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1704) | 提交请求并关联返回状态 |

## Overlap 循环 → 收请求 → 入队

**已覆盖 8/8（100.0%），待覆盖 0。**

Scheduler 循环已在后台运行，TokenizerManager 通过 ZMQ 发送请求；多 rank 广播按配置执行。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 27 | ✅ 已读 | Scheduler | [event\_loop\_overlap](python/sglang/srt/managers/scheduler.py#L1889) | CPU 调度和 GPU 执行交错推进 |
| 28 | ✅ 已读 | SchedulerRequestReceiver | [recv\_requests](python/sglang/srt/managers/scheduler_components/request_receiver.py#L88) | 接收本轮请求并做 rank 同步 |
| 29 | ✅ 已读 | SchedulerRequestReceiver | [\_pull\_raw\_reqs](python/sglang/srt/managers/scheduler_components/request_receiver.py#L125) | 从主进程管道接收消息 |
| 30 | ☑️ 推定 · B | SchedulerRequestReceiver | [\_broadcast\_reqs\_across\_ranks](python/sglang/srt/managers/scheduler_components/request_receiver.py#L183) | 将请求分发到参与执行的 rank |
| 31 | ✅ 已读 | Scheduler | [process\_input\_requests](python/sglang/srt/managers/scheduler.py#L2057) | 按消息类型分派处理 |
| 32 | ✅ 已读 | Scheduler | [handle\_generate\_request](python/sglang/srt/managers/scheduler.py#L2608) | 将 tokenized 输入变为调度请求 |
| 33 | ✅ 已读 | Req | [\_\_init\_\_](python/sglang/srt/managers/schedule_batch.py#L839) | 建立请求、生成和缓存状态 |
| 34 | ✅ 已读 | Scheduler | [\_add\_request\_to\_queue](python/sglang/srt/managers/scheduler.py#L3000) | 进入等待队列 |

## 选择 Prefill 请求 → 匹配前缀 → 建立批次

**已覆盖 11/15（73.3%），待覆盖 4。**

展示父方法后再下钻；分块 Prefill、前缀匹配属于条件路径。RadixCache 是学习代表实现，实际缓存由模型和配置选择。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 35 | ✅ 已读 | Scheduler | [get\_next\_batch\_to\_run](python/sglang/srt/managers/scheduler.py#L3300) | 选择下一轮 prefill/decode 批次 |
| 36 | ✅ 已读 | Scheduler | [get\_new\_batch\_prefill](python/sglang/srt/managers/scheduler.py#L3503) | 建立 prefill 批次入口 |
| 37 | ✅ 已读 | Scheduler | [\_get\_new\_batch\_prefill\_raw](python/sglang/srt/managers/scheduler.py#L3535) | 选择可接纳的等待请求 |
| 38 | ✅ 已读 | SchedulePolicy | [calc\_priority](python/sglang/srt/managers/schedule_policy.py#L242) | 确定等待队列处理顺序 |
| 39 | ✅ 已读 | PrefillAdder | [\_\_init\_\_](python/sglang/srt/managers/schedule_policy.py#L527) | 建立 token 与显存接纳预算 |
| 40 | ⬜ 未读 | Req | [init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1343) | 按已缓存前缀准备下轮输入 |
| 41 | ⬜ 未读 | RadixCache | [match\_prefix](python/sglang/srt/mem_cache/radix_cache.py#L391) | 查找可复用前缀 |
| 42 | ⬜ 未读 | RadixCache | [\_match\_prefix\_helper](python/sglang/srt/mem_cache/radix_cache.py#L702) | 沿树匹配 token 序列 |
| 43 | ⬜ 未读 | RadixCache | [\_split\_node](python/sglang/srt/mem_cache/radix_cache.py#L728) | 部分命中时拆分树节点 |
| 44 | ✅ 已读 | PrefillAdder | [add\_one\_req](python/sglang/srt/managers/schedule_policy.py#L1246) | 检查预算并接纳请求 |
| 45 | ✅ 已读 | RadixCache | [inc\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L637) | 保护活跃请求使用的缓存 |
| 46 | ✅ 已读 | PrefillAdder | [add\_chunked\_req](python/sglang/srt/managers/schedule_policy.py#L1038) | 继续处理分块 prefill 请求 |
| 47 | ✅ 已读 | PrefillAdder | [\_update\_prefill\_budget](python/sglang/srt/managers/schedule_policy.py#L888) | 扣减本轮 prefill 预算 |
| 48 | ✅ 已读 | ScheduleBatch | [init\_new](python/sglang/srt/managers/schedule_batch.py#L2278) | 从请求集合建立调度批次 |
| 49 | ✅ 已读 | ScheduleBatch | [prepare\_for\_extend](python/sglang/srt/managers/schedule_batch.py#L2465) | 准备 prefill token、长度和缓存位置 |

## 为 Prefill 分配 KV 空间

**已覆盖 2/4（50.0%），待覆盖 2。**

关注请求映射表行、逻辑 token 位置和物理 KV 槽位的关系；paged allocator 为代表实现。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 50 | ✅ 已读 | — | [alloc\_for\_extend](python/sglang/srt/mem_cache/allocation.py#L282) | 为新增 prefill token 分配 KV |
| 51 | ✅ 已读 | ReqToTokenPool | [alloc](python/sglang/srt/mem_cache/memory_pool.py#L317) | 为活跃请求分配映射表行 |
| 52 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [alloc\_extend](python/sglang/srt/mem_cache/allocator/paged.py#L172) | 按页接纳 prefill 的 KV 增量 |
| 53 | ⬜ 未读 | — | [assign\_req\_to\_token\_pool](python/sglang/srt/mem_cache/allocation.py#L617) | 将逻辑 token 位置映射到 KV 槽位 |

## 决定本轮重叠 → 解析输入 → 提交前向

**已覆盖 4/6（66.7%），待覆盖 2。**

run_batch 只读 enable_overlap 分支；resolve_forward_inputs 在 worker 调用和 ForwardBatch.init_new 之前。若需关闭本轮重叠，先处理队列中的上一批结果。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 54 | ⬜ 未读 | Scheduler | [is\_disable\_overlap\_for\_batch](python/sglang/srt/managers/scheduler.py#L2008) | 判断是否需要先处理上一批结果 |
| 55 | ✅ 已读 | Scheduler | [run\_batch](python/sglang/srt/managers/scheduler.py#L4058) | 提交一轮模型计算并接收结果 |
| 56 | ✅ 已读 | — | [resolve\_forward\_inputs](python/sglang/srt/managers/overlap_utils.py#L87) | 将 overlap 的未来 token 解析为实际输入 |
| 57 | ⬜ 未读 | Scheduler | [\_forward\_isolation](python/sglang/srt/managers/scheduler.py#L4014) | 隔离批次可变状态并保留跨流张量引用 |
| 58 | ✅ 已读 | TpModelWorker | [forward\_batch\_generation](python/sglang/srt/managers/tp_worker.py#L603) | 组织 ForwardBatch、前向与采样 |
| 59 | ✅ 已读 | ForwardBatch | [init\_new](python/sglang/srt/model_executor/forward_batch_info.py#L744) | 调度视图转换为 GPU 批次视图 |

## 模型前向 → Qwen 3.5 → logits 与采样

**已覆盖 18/27（66.7%），待覆盖 9。**

外层实例是 Qwen3_5ForConditionalGeneration，执行继承的 Qwen3VLForConditionalGeneration.forward。language_model_only=True 时直接调用 self.model；否则纯文本也先经 general_mm_embed_routine，再进入 qwen3_5.py 的 Qwen3_5ForCausalLM.forward。主干返回 hidden states，外层调用 LogitsProcessor 得到 logits，ModelRunner.sample 再采样。Prefill/Decode、普通 Attention/GDN、Eager/CUDA Graph 分别是不同分支；FlashAttention 是普通 Attention 后端代表。满足延迟采样条件时，sample 移到上一批结果处理之后。先读 QKV 线性投影，再看注意力缓存读写；这是阅读顺序，投影方法不调用 RadixAttention。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 60 | ✅ 已读 | ModelRunner | [forward](python/sglang/srt/model_executor/model_runner.py#L1645) | 模型前向总入口 |
| 61 | ✅ 已读 | ModelRunner | [\_forward\_raw](python/sglang/srt/model_executor/model_runner.py#L1820) | 选择图执行或 eager 路径 |
| 62 | ☑️ 推定 · C | EagerRunner | [execute](python/sglang/srt/model_executor/runner/eager_runner.py#L216) | 按 ForwardMode 分派执行 |
| 63 | ✅ 已读 | EagerRunner | [\_execute\_extend](python/sglang/srt/model_executor/runner/eager_runner.py#L290) | Prefill 的模型执行入口 |
| 64 | ✅ 已读 | EagerRunner | [\_execute\_decode](python/sglang/srt/model_executor/runner/eager_runner.py#L249) | Decode 的模型执行入口 |
| 65 | ☑️ 推定 · B | FlashAttentionBackend | [init\_forward\_metadata](python/sglang/srt/layers/attention/flashattention_backend.py#L683) | 准备批次和 KV 索引元数据 |
| 66 | ✅ 已读 | Qwen3VLForConditionalGeneration | [forward](python/sglang/srt/models/qwen3_vl.py#L1447) | Qwen 3.5 继承的入口：执行文本主干并产出 logits |
| 67 | ☑️ 推定 · C | — | [general\_mm\_embed\_routine](python/sglang/srt/managers/mm_utils.py#L608) | 非 language_model_only 配置下，纯文本也经此完成 embedding 并调用主干 |
| 68 | ✅ 已读 | Qwen3_5ForCausalLM | [forward](python/sglang/srt/models/qwen3_5.py#L1534) | 嵌入、混合类型解码层逐层执行与最终归一化 |
| 69 | ✅ 已读 | Qwen3_5AttentionDecoderLayer | [forward](python/sglang/srt/models/qwen3_5.py#L1285) | 普通 Attention 层的注意力、MLP 与残差连接 |
| 70 | ✅ 已读 | Qwen3_5AttentionDecoderLayer | [self\_attention](python/sglang/srt/models/qwen3_5.py#L1237) | QKV 与位置编码准备、RadixAttention 和输出门控 |
| 71 | ⬜ 未读 | ColumnParallelLinear | [forward](python/sglang/srt/layers/linear.py#L492) | QKVParallelLinear 继承的前向入口，调用 quant_method.apply 执行投影 |
| 72 | ⬜ 未读 | UnquantizedLinearMethod | [apply](python/sglang/srt/layers/quantization/unquant.py#L255) | 未量化线性计算的普通分支，以 F.linear 执行输入与权重的投影 |
| 73 | ✅ 已读 | RadixAttention | [forward](python/sglang/srt/layers/radix_attention.py#L161) | 模型层接入 SGLang 注意力后端 |
| 74 | ☑️ 推定 · C | FlashAttentionBackend | [forward\_extend](python/sglang/srt/layers/attention/flashattention_backend.py#L1228) | Prefill 注意力与 KV 写入 |
| 75 | ☑️ 推定 · C | FlashAttentionBackend | [forward\_decode](python/sglang/srt/layers/attention/flashattention_backend.py#L1804) | 读取历史 KV 并计算 decode 注意力 |
| 76 | ✅ 已读 | MHATokenToKVPool | [set\_kv\_buffer](python/sglang/srt/mem_cache/memory_pool.py#L2431) | 将新 token 的 K/V 写入槽位 |
| 77 | ⬜ 未读 | Qwen3_5LinearDecoderLayer | [forward](python/sglang/srt/models/qwen3_5.py#L875) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 |
| 78 | ⬜ 未读 | Qwen3_5GatedDeltaNet | [forward](python/sglang/srt/models/qwen3_5.py#L708) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 |
| 79 | ⬜ 未读 | RadixLinearAttention | [forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 |
| 80 | ⬜ 未读 | Qwen2MoeMLP | [forward](python/sglang/srt/models/qwen2_moe.py#L249) | Qwen 3.5 dense 复用的门控前馈网络 |
| 81 | ✅ 已读 | LogitsProcessor | [forward](python/sglang/srt/layers/logits_processor.py#L343) | 选择需要输出的 token 并形成 logits |
| 82 | ⬜ 未读 | LogitsProcessor | [\_get\_logits](python/sglang/srt/layers/logits_processor.py#L668) | 隐藏状态经过 LM head 得到词表分数 |
| 83 | ⬜ 未读 | LogitsProcessor | [\_compute\_lm\_head](python/sglang/srt/layers/logits_processor.py#L722) | 执行 LM head 词表投影，普通权重路径计算 hidden_states × weight.T |
| 84 | ☑️ 推定 · C | ModelRunner | [sample](python/sglang/srt/model_executor/model_runner.py#L1955) | 调用 sampler 并返回 token ID |
| 85 | ⬜ 未读 | ModelRunner | [\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L1927) | 惩罚项、掩码与 logit 偏置 |
| 86 | ✅ 已读 | Sampler | [forward](python/sglang/srt/layers/sampler.py#L98) | 贪心与概率采样的核心分派 |

## 保存本批结果 → 交接下一轮输入

**已覆盖 2/5（40.0%），待覆盖 3。**

publish 发布序列长度等元数据，stash 保存生成 token 等下一轮输入。copy_to_cpu 安排异步复制；随后将 batch 快照与结果放入 result_queue。若延迟采样，relay/stash/copy 在延迟采样后执行。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 87 | ⬜ 未读 | FutureMap | [publish](python/sglang/srt/managers/overlap_utils.py#L532) | 发布下一轮序列长度等元数据 |
| 88 | ☑️ 推定 · C | Scheduler | [\_relay\_forward\_payload](python/sglang/srt/managers/scheduler.py#L4333) | 将采样结果交给下一轮输入通道 |
| 89 | ✅ 已读 | FutureMap | [stash](python/sglang/srt/managers/overlap_utils.py#L557) | 保存生成 token 等下一轮输入数据 |
| 90 | ⬜ 未读 | GenerationBatchResult | [copy\_to\_cpu](python/sglang/srt/managers/utils.py#L123) | 异步复制结果到 CPU 并记录完成事件 |
| 91 | ⬜ 未读 | Scheduler | [\_apply\_war\_barrier](python/sglang/srt/managers/scheduler.py#L1834) | 保护本批共享读取，约束后续调度写入 |

## 处理上一批结果 → 结束判断 → 缓存与释放

**已覆盖 4/13（30.8%），待覆盖 9。**

event_loop_overlap 内的 pop_and_process 从队头取出上一批并调用 process_batch_result；此处通常与本批 GPU 工作重叠。完成/未完成请求走不同分支；最后 launch_batch_sample_if_needed 仅在存在 delay_sample_func 时实际采样。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 92 | ✅ 已读 | Scheduler | [process\_batch\_result](python/sglang/srt/managers/scheduler.py#L4424) | 按批次类型处理前向结果 |
| 93 | ✅ 已读 | SchedulerBatchResultProcessor | [process\_batch\_result\_prefill](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L240) | 处理 prefill 输出和请求状态 |
| 94 | ✅ 已读 | SchedulerBatchResultProcessor | [process\_batch\_result\_decode](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L870) | 接收新 token 并推进 decode 状态 |
| 95 | ⬜ 未读 | Req | [update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1674) | 判断长度、终止 token 和停止字符串 |
| 96 | ⬜ 未读 | RadixCache | [cache\_unfinished\_req](python/sglang/srt/mem_cache/radix_cache.py#L530) | 缓存继续运行请求的前缀 |
| 97 | ⬜ 未读 | — | [release\_kv\_cache](python/sglang/srt/mem_cache/common.py#L202) | 结束请求时转交缓存并释放资源 |
| 98 | ⬜ 未读 | RadixCache | [cache\_finished\_req](python/sglang/srt/mem_cache/radix_cache.py#L473) | 完成请求的缓存写回与释放 |
| 99 | ⬜ 未读 | RadixCache | [insert](python/sglang/srt/mem_cache/radix_cache.py#L451) | 插入可复用 token 前缀 |
| 100 | ⬜ 未读 | RadixCache | [\_insert\_helper](python/sglang/srt/mem_cache/radix_cache.py#L761) | 维护前缀树节点与 KV 所有权 |
| 101 | ⬜ 未读 | RadixCache | [dec\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L661) | 释放缓存引用保护 |
| 102 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [free](python/sglang/srt/mem_cache/allocator/paged.py#L261) | 归还 KV 页面 |
| 103 | ⬜ 未读 | ReqToTokenPool | [free](python/sglang/srt/mem_cache/memory_pool.py#L370) | 归还请求映射表行 |
| 104 | ☑️ 推定 · C | Scheduler | [launch\_batch\_sample\_if\_needed](python/sglang/srt/managers/scheduler.py#L4386) | 上一批处理后按需采样本批并交接结果 |

## 请求未结束 → 准备下一轮 Decode

**已覆盖 2/10（20.0%），待覆盖 8。**

这些操作位于后续轮次的 get_next_batch_to_run 中；显存不足时才淘汰/撤回。准备后回到提交前向阶段，直到结束。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 105 | ⬜ 未读 | ScheduleBatch | [merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3406) | 合并完成 prefill 的请求与运行批次 |
| 106 | ✅ 已读 | Scheduler | [update\_running\_batch](python/sglang/srt/managers/scheduler.py#L3911) | 维护继续生成的请求集合 |
| 107 | ⬜ 未读 | ScheduleBatch | [filter\_batch](python/sglang/srt/managers/schedule_batch.py#L3319) | 移除结束请求并对齐批次张量 |
| 108 | ⬜ 未读 | ScheduleBatch | [check\_decode\_mem](python/sglang/srt/managers/schedule_batch.py#L2960) | 检查下一轮 KV 容量 |
| 109 | ⬜ 未读 | RadixCache | [evict](python/sglang/srt/mem_cache/radix_cache.py#L607) | 淘汰未保护前缀以腾出 KV |
| 110 | ⬜ 未读 | ScheduleBatch | [retract\_decode](python/sglang/srt/managers/schedule_batch.py#L2967) | 显存不足时撤回部分请求 |
| 111 | ⬜ 未读 | Req | [reset\_for\_retract](python/sglang/srt/managers/schedule_batch.py#L1716) | 重置被撤回请求以便重新调度 |
| 112 | ✅ 已读 | ScheduleBatch | [prepare\_for\_decode](python/sglang/srt/managers/schedule_batch.py#L3223) | 为每个请求准备下一 token 的执行状态 |
| 113 | ⬜ 未读 | — | [alloc\_for\_decode](python/sglang/srt/mem_cache/allocation.py#L551) | 为下一轮 decode token 分配 KV |
| 114 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [alloc\_decode](python/sglang/srt/mem_cache/allocator/paged.py#L222) | 处理 decode 跨页分配 |

## 输出 token → 解码文本 → 完整 HTTP 响应

**已覆盖 11/11（100.0%），待覆盖 0。**

结果处理内部发出输出，跨 ZMQ 到 Detokenizer，再回 TokenizerManager；后台 handle_loop 唤醒 _wait_one_response。非流式请求也复用 stream_output，最终由最初的 _handle_non_streaming_request 返回完整响应。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 115 | ✅ 已读 | SchedulerOutputStreamer | [stream\_output](python/sglang/srt/managers/scheduler_components/output_streamer.py#L118) | 组织生成结果并选择输出类型 |
| 116 | ☑️ 推定 · C | SchedulerOutputStreamer | [\_stream\_output\_generation](python/sglang/srt/managers/scheduler_components/output_streamer.py#L146) | 按流式间隔构造 token 输出批次 |
| 117 | ✅ 已读 | DetokenizerManager | [event\_loop](python/sglang/srt/managers/detokenizer_manager.py#L207) | 接收 token 输出并分发解码 |
| 118 | ✅ 已读 | DetokenizerManager | [handle\_batch\_token\_id\_out](python/sglang/srt/managers/detokenizer_manager.py#L506) | 解码并发送文本输出消息 |
| 119 | ✅ 已读 | DetokenizerManager | [\_decode\_batch\_token\_id\_output](python/sglang/srt/managers/detokenizer_manager.py#L360) | 维护请求的增量解码状态 |
| 120 | ✅ 已读 | DetokenizerManager | [\_grouped\_batch\_decode](python/sglang/srt/managers/detokenizer_manager.py#L282) | 将批量 token 增量转为字符串 |
| 121 | ☑️ 推定 · B | DetokenizerManager | [trim\_matched\_stop](python/sglang/srt/managers/detokenizer_manager.py#L230) | 按停止规则裁剪返回文本 |
| 122 | ✅ 已读 | TokenizerManager | [handle\_loop](python/sglang/srt/managers/tokenizer_manager.py#L2341) | 接收返回消息并按类型分发 |
| 123 | ✅ 已读 | TokenizerManager | [\_handle\_batch\_output](python/sglang/srt/managers/tokenizer_manager.py#L2362) | 更新 ReqState 并唤醒等待者 |
| 124 | ✅ 已读 | TokenizerManager | [\_wait\_one\_response](python/sglang/srt/managers/tokenizer_manager.py#L1841) | 等待并产出请求结果 |
| 125 | ☑️ 推定 · B | OpenAIServingChat | [\_build\_chat\_response](python/sglang/srt/entrypoints/openai/serving_chat.py#L1862) | 将完整生成结果包装成 OpenAI 响应 |

## 条件分支：Decode CUDA Graph

**已覆盖 0/2（0.0%），待覆盖 2。**

这是 _forward_raw 中替代 Eager 的路径，不是 HTTP 回包之后执行。capture 在图准备阶段，不属于本次运行中请求路线，因此不计入本路线。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 126 | ⬜ 未读 | DecodeCudaGraphRunner | [can\_run\_graph](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L676) | 判断批次是否满足图重放条件 |
| 127 | ⬜ 未读 | DecodeCudaGraphRunner | [execute](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1449) | 重放已捕获的执行图 |

## 重算与后续查询

以后询问“学习路线”“学习现状”“学了多少”“接下来读什么”时，默认使用此模型准备与 Overlap 路线。先从当前源码重算，再报告百分比、阶段表和逐方法清单；保持类名/方法名分列、一个方法一行。全核心总览见 LEARN.methods.md，不能混用两套范围的百分比。

```bash
python quick_learn/update_overlap_progress.py          # 只预览阶段进度
python quick_learn/update_overlap_progress.py --route  # 预览完整方法路线
python quick_learn/update_overlap_progress.py --write  # 更新本文
python quick_learn/update_overlap_progress.py --check  # 检查是否与源码一致
```

本文件自动生成；修改阅读范围、顺序或重点请编辑 [路线范围](quick_learn/LEARN.overlap.scope.json)，调整统计范围时递增版本。手动改文档不会产生已读证据。

<details>
<summary>展开逐方法判定依据与来源</summary>

| 序号 | 判定 | 依据 |
| ---: | --- | --- |
| 1 | D | 新增中文说明 L1164,1167,1171,1177,1178,1180,1182,1183,1185,1190,1217,1235,1238,1285,1288 |
| 2 | D | 新增中文说明 L279,282,288,289,291,293,294,296,321,341 |
| 3 | D | 新增中文说明 L969,982,984,988,990,992,994,995,997,999,1000 |
| 4 | D | 新增中文说明 L277,285,291,292,294,296,297,299,302,303,325,327,334,335 |
| 5 | D | 新增中文说明 L1919 |
| 6 | D | 新增中文说明 L1265,1303,1317,1318,1336 |
| 7 | D | 新增中文说明 L1456,1497,1503,1506,1507,1509,1510 |
| 8 | C | chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| 9 | C | chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| 10 | C | chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| 11 | U | 无直接证据，且未满足补全规则 |
| 12 | D | 新增中文说明 L247 |
| 13 | U | 无直接证据，且未满足补全规则 |
| 14 | U | 无直接证据，且未满足补全规则 |
| 15 | U | 无直接证据，且未满足补全规则 |
| 16 | D | 新增中文说明 L1058 |
| 17 | B | branch=model_loading;direct=5;covered_before_branch=5/6 |
| 18 | D | 新增中文说明 L1834,1835,1836,1837,1838,1841,1842,1846 |
| 19 | D | 新增中文说明 L856,859,860,865,888,891,895,898,902,904,905,906,911,915,917,918,920,921,922,923,924,928,930,940 |
| 20 | D | 新增中文说明 L3588,3604,3606,3623,3628,3629 |
| 21 | D | 新增中文说明 L1081,1085,1110,1113,1116,1241,1243,1244,1245,1246,1247,1248,1249 |
| 22 | D | 新增中文说明 L1001,1037 |
| 23 | D | 新增中文说明 L1288,1289,1290 |
| 24 | D | 新增中文说明 L253 |
| 25 | D | 新增中文说明 L1463,1466,1467,1468,1471,1472,1475,1478,1481,1482,1492,1505,1506,1507,1517,1525 |
| 26 | D | 新增中文说明 L1708,1721 |
| 27 | D | 新增中文说明 L1888,1891,1893,1894,1895,1896,1897,1901,1902,1905,1916,1922,1923,1926,1932,1934,1935,1937,1942,1943,1944,1946,1947,1949,1950,1951,1952,1953,1954,1956,1978,1979,1988 |
| 28 | D | 新增中文说明 L92,102,103,109,110,111,114,116,118 |
| 29 | D | 新增中文说明 L127,131,148,158,167,168 |
| 30 | B | branch=scheduler_ingress;direct=6;covered_before_branch=6/7 |
| 31 | D | 新增中文说明 L2059,2060,2061,2065,2070,2082,2083 |
| 32 | D | 新增中文说明 L2613,2618,2621,2626,2638,2640,2687,2702,2703,2704,2770,2772,2940 |
| 33 | D | 新增中文说明 L1025,1029,1043,1047,1142 |
| 34 | D | 新增中文说明 L3001,3008,3011 |
| 35 | D | 新增中文说明 L3304,3306,3307,3308,3309,3310,3311,3312,3313,3314,3315,3316,3317,3318,3319,3320,3321,3322,3323,3324,3325,3350,3351,3352,3365,3366,3367,3394,3400,3405,3422,3423,3441,3442,3444,3447,3449,3450,3452 |
| 36 | D | 新增中文说明 L3513,3521,3531,3532 |
| 37 | D | 新增中文说明 L3541,3557,3558,3561,3562,3563,3597,3606,3608,3609,3611,3612,3618,3619,3632,3633,3636,3638,3639,3640,3641,3642,3644,3647,3648,3657,3658,3679,3703,3711,3713,3717,3720,3723,3743,3789,3793,3811,3813,3814,3823,3824,3825,3830,3837 |
| 38 | D | 新增中文说明 L245,246,256,260 |
| 39 | D | 新增中文说明 L536,546,547,551,557,558,571 |
| 40 | U | 无直接证据，且未满足补全规则 |
| 41 | U | 无直接证据，且未满足补全规则 |
| 42 | U | 无直接证据，且未满足补全规则 |
| 43 | U | 无直接证据，且未满足补全规则 |
| 44 | D | 新增中文说明 L1250,1271,1272,1274,1275,1276,1277,1279,1280,1285,1294,1297,1301,1336,1339,1340,1393,1395,1396,1426,1427,1435,1441,1443,1445,1446,1447,1449,1450,1452,1457,1489,1508 |
| 45 | D | 新增中文说明 L638,642,643,645,647,648,649,650,651,652,655,656,657 |
| 46 | D | 新增中文说明 L1074 |
| 47 | D | 新增中文说明 L891,898,908,909,911,912,913,916,917,935 |
| 48 | D | 新增中文说明 L2281,2283,2291,2293,2296,2297,2301,2304,2305,2308,2309,2310,2312,2313 |
| 49 | D | 新增中文说明 L2466,2468,2477,2479,2481,2483,2485,2487,2489,2492,2493,2524,2544,2682,2743 |
| 50 | D | 新增中文说明 L292,293,294,295,296,297,298,301,302,304,307,310,332,342,355,359,377,391,412 |
| 51 | D | 新增中文说明 L319,341 |
| 52 | U | 无直接证据，且未满足补全规则 |
| 53 | U | 无直接证据，且未满足补全规则 |
| 54 | U | 无直接证据，且未满足补全规则 |
| 55 | D | 新增中文说明 L4060,4064,4075,4092,4094,4101,4102,4103,4104,4106,4113,4137,4138,4142,4238,4241,4247,4251,4253,4255,4261,4277 |
| 56 | D | 新增中文说明 L109,110,111 |
| 57 | U | 无直接证据，且未满足补全规则 |
| 58 | D | 新增中文说明 L605,614,621,623,642,645,663,688,691 |
| 59 | D | 新增中文说明 L804,807,809,811,813,815,817 |
| 60 | D | 新增中文说明 L1647,1653,1655,1667,1670,1671,1673,1676,1677,1678,1680,1681,1720 |
| 61 | D | 新增中文说明 L1822,1828,1841,1858,1914 |
| 62 | C | chain=eager_extend_model;anchors=raw_forward->eager_extend \| chain=eager_decode_model;anchors=raw_forward->eager_decode \| chain=eager_extend_linear_model;anchors=raw_forward->eager_extend \| chain=eager_decode_linear_model;anchors=raw_forward->eager_decode |
| 63 | D | 新增中文说明 L292,296,389,392,393 |
| 64 | D | 新增中文说明 L255,256,257,258,259,260,261,280,281,282,284 |
| 65 | B | branch=attention_backend;direct=2;covered_before_branch=4/5 |
| 66 | D | 新增中文说明 L1449,1451,1456,1472,1473,1475,1476,1477,1478,1511,1514,1515,1517,1521,1525 |
| 67 | C | chain=qwen_text_embedding;anchors=causal_lm->qwen_model |
| 68 | D | 新增中文说明 L1543,1545,1546,1563,1568 |
| 69 | D | 新增中文说明 L1288,1290,1295,1297,1308,1315,1332,1340 |
| 70 | D | 新增中文说明 L1240,1244,1261 |
| 71 | U | 无直接证据，且未满足补全规则 |
| 72 | U | 无直接证据，且未满足补全规则 |
| 73 | D | 新增中文说明 L167,175,182,292 |
| 74 | C | chain=eager_extend_model;anchors=radix_attention->store_kv |
| 75 | C | chain=eager_decode_model;anchors=radix_attention->store_kv |
| 76 | D | 新增中文说明 L2434,2435,2436,2443 |
| 77 | U | 无直接证据，且未满足补全规则 |
| 78 | U | 无直接证据，且未满足补全规则 |
| 79 | U | 无直接证据，且未满足补全规则 |
| 80 | U | 无直接证据，且未满足补全规则 |
| 81 | D | 新增中文说明 L346,353,354,355,356,360 |
| 82 | U | 无直接证据，且未满足补全规则 |
| 83 | U | 无直接证据，且未满足补全规则 |
| 84 | C | chain=worker_sample;anchors=worker_forward->sample \| chain=overlap_delayed_sample;anchors=overlap_loop->sample |
| 85 | U | 无直接证据，且未满足补全规则 |
| 86 | D | 新增中文说明 L108 |
| 87 | U | 无直接证据，且未满足补全规则 |
| 88 | C | chain=overlap_relay;anchors=run_batch->future_stash |
| 89 | D | 新增中文说明 L558 |
| 90 | U | 无直接证据，且未满足补全规则 |
| 91 | U | 无直接证据，且未满足补全规则 |
| 92 | D | 新增中文说明 L4429 |
| 93 | D | 新增中文说明 L245 |
| 94 | D | 新增中文说明 L875 |
| 95 | U | 无直接证据，且未满足补全规则 |
| 96 | U | 无直接证据，且未满足补全规则 |
| 97 | U | 无直接证据，且未满足补全规则 |
| 98 | U | 无直接证据，且未满足补全规则 |
| 99 | U | 无直接证据，且未满足补全规则 |
| 100 | U | 无直接证据，且未满足补全规则 |
| 101 | U | 无直接证据，且未满足补全规则 |
| 102 | U | 无直接证据，且未满足补全规则 |
| 103 | U | 无直接证据，且未满足补全规则 |
| 104 | C | chain=overlap_delayed_sample;anchors=overlap_loop->sample |
| 105 | U | 无直接证据，且未满足补全规则 |
| 106 | D | 新增中文说明 L3913,3993 |
| 107 | U | 无直接证据，且未满足补全规则 |
| 108 | U | 无直接证据，且未满足补全规则 |
| 109 | U | 无直接证据，且未满足补全规则 |
| 110 | U | 无直接证据，且未满足补全规则 |
| 111 | U | 无直接证据，且未满足补全规则 |
| 112 | D | 新增中文说明 L3262 |
| 113 | U | 无直接证据，且未满足补全规则 |
| 114 | U | 无直接证据，且未满足补全规则 |
| 115 | D | 新增中文说明 L125 |
| 116 | C | chain=token_message;anchors=stream_output->detokenizer_loop |
| 117 | D | 新增中文说明 L209,210,211,215,220,221,225 |
| 118 | D | 新增中文说明 L517,521,522,528 |
| 119 | D | 新增中文说明 L361,366,370,373,374,380,383,397,399,434 |
| 120 | D | 新增中文说明 L319,325 |
| 121 | B | branch=detokenize;direct=4;covered_before_branch=4/5 |
| 122 | D | 新增中文说明 L2344,2348,2349,2355 |
| 123 | D | 新增中文说明 L2364,2370,2378,2380,2392,2505,2517,2521,2522,2652,2660,2666,2679 |
| 124 | D | 新增中文说明 L1847,1854,1855,1856,1861,1878,1879,1880,1887,1894,1895,1909,1919,1938,1947,1952,1961 |
| 125 | B | branch=response;direct=4;covered_before_branch=4/5 |
| 126 | U | 无直接证据，且未满足补全规则 |
| 127 | U | 无直接证据，且未满足补全规则 |

</details>

- 路线版本：overlap_nonstream v3；固定分母：127。
- 统计对象：当前工作区，包含未提交修改；日期不作为已读证据。
- HEAD：`81827dda8cfb46762579b96ee04608661927cdb9`。
- 固定对照基线：`6388b6cfb1d93c253714a408f1d66a093302acd7`。
- 源码指纹：`4b7cc727adbe11248577310c97e620118a23d1f2b6f1f9db266dbc68e302bf49`。
- 范围指纹：`86c8ac64bfb4df570e949a1540536669e6ebc02b8fe2148abb3ca99c3a6462c7`。
