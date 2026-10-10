# SGLang Overlap 学习路线与现状

先读 Qwen3_5ForConditionalGeneration 的模型初始化与权重加载，再跟非流式文本生成请求：从 OpenAIServingChat._handle_non_streaming_request 到完整响应；调度只沿 event_loop_overlap，包含共享的 Prefill/Decode、KV、Qwen 3.5 dense 推理、采样与回包。补入 QKV 投影及父类的核心构造与未量化前向方法。 不计入已排除的 5 个 Overlap 协作方法。 main 合并后的接口映射：旧 cache_unfinished_req/cache_finished_req 已移除，缓存写回改为 checkpoint → _insert_cache，结束释放仍由 release_kv_cache 负责；方法总数不变。

**Overlap 路线阅读覆盖：95/122（77.9%），待覆盖 27 个。**

直接注释 D：79；链路补全 C：9；分支补全 B：7。

| 阅读阶段 | 已覆盖 / 总数 | 待覆盖 | 进度 |
| --- | ---: | ---: | ---: |
| 启动准备：Qwen 3.5 初始化 → 权重加载 | 17/17 | 0 | 100.0% |
| 接收请求 → 分词 → 发给 Scheduler | 9/9 | 0 | 100.0% |
| Overlap 循环 → 收请求 → 入队 | 8/8 | 0 | 100.0% |
| 选择 Prefill 请求 → 匹配前缀 → 建立批次 | 11/15 | 4 | 73.3% |
| 为 Prefill 分配 KV 空间 | 2/4 | 2 | 50.0% |
| 解析输入 → 提交前向 | 4/4 | 0 | 100.0% |
| 模型前向 → Qwen 3.5 → logits 与采样 | 24/27 | 3 | 88.9% |
| 保存本批结果 → 交接下一轮输入 | 2/2 | 0 | 100.0% |
| 处理上一批结果 → 结束判断 → 缓存与释放 | 4/13 | 9 | 30.8% |
| 请求未结束 → 准备下一轮 Decode | 3/10 | 7 | 30.0% |
| 输出 token → 解码文本 → 完整 HTTP 响应 | 11/11 | 0 | 100.0% |
| 条件分支：Decode CUDA Graph | 0/2 | 2 | 0.0% |

## 统计口径与阅读约定

路线 v4（2026-10-08）：在 v3 的 127 项基础上移除 5 个 Overlap 协作方法，共 122 项：Scheduler.is_disable_overlap_for_batch、Scheduler._forward_isolation、FutureMap.publish、GenerationBatchResult.copy_to_cpu、Scheduler._apply_war_barrier。保留 Overlap 主循环与其余核心执行方法；统计基线与 D/C/B 规则保持不变。分母缩小带来的覆盖率变化不代表新增学习，也不与全核心总览混算。

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
- Overlap 协作细节：Scheduler.is_disable_overlap_for_batch、Scheduler._forward_isolation、FutureMap.publish、GenerationBatchResult.copy_to_cpu、Scheduler._apply_war_barrier；仅保留必要时序说明，不纳入方法计数

## 接下来重点看

按固定优先级从当前未覆盖方法中选择，再按路线顺序展示；学习后重算会自动跳过已覆盖项。序号对应下方完整路线。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 40 | ⬜ 未读 | Req | [init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1619) | 按已缓存前缀准备下轮输入 |
| 41 | ⬜ 未读 | RadixCache | [match\_prefix](python/sglang/srt/mem_cache/radix_cache.py#L373) | 查找可复用前缀 |
| 53 | ⬜ 未读 | — | [assign\_req\_to\_token\_pool](python/sglang/srt/mem_cache/allocation.py#L713) | 将逻辑 token 位置映射到 KV 槽位 |
| 80 | ⬜ 未读 | LogitsProcessor | [\_get\_logits](python/sglang/srt/layers/logits_processor.py#L876) | 隐藏状态经过 LM head 得到词表分数 |
| 81 | ⬜ 未读 | LogitsProcessor | [\_compute\_lm\_head](python/sglang/srt/layers/logits_processor.py#L978) | 执行 LM head 词表投影，普通权重路径计算 hidden_states × weight.T |
| 83 | ⬜ 未读 | ModelRunner | [\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L2096) | 惩罚项、掩码与 logit 偏置 |
| 90 | ⬜ 未读 | Req | [update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1908) | 判断长度、终止 token 和停止字符串 |
| 92 | ⬜ 未读 | — | [release\_kv\_cache](python/sglang/srt/mem_cache/common.py#L379) | 结束请求时转交缓存并释放资源 |
| 93 | ⬜ 未读 | RadixCache | [\_insert\_cache](python/sglang/srt/mem_cache/radix_cache.py#L459) | 将页对齐 KV 前缀写入树并释放重复槽位 |
| 100 | ⬜ 未读 | ScheduleBatch | [merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3800) | 合并完成 prefill 的请求与运行批次 |

## 启动准备：Qwen 3.5 初始化 → 权重加载

**已覆盖 17/17（100.0%），待覆盖 0。**

发生在请求执行前，一次初始化服务后续批次。DefaultModelLoader.load_model 先调用 _initialize_model 构造模型树，构造返回后再加载权重。入口子类调用父类并注入文本主干；主干按 layers_block_type 创建两类解码层。编号是阅读顺序，两类层和共用 MLP 不是串行调用。详细对象关系见 [LEARN.md 第 3.1 节](LEARN.md#31-固定模型qwen3_5forconditionalgeneration)。QKV 投影沿继承链初始化，父类构造返回后才创建权重参数；两条调用链分开判定。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 1 | ✅ 已读 | ModelRunner | [load\_model](python/sglang/srt/model_executor/model_runner.py#L1245) | 启动期加载模型并保存 loader 与外层模型实例 |
| 2 | ✅ 已读 | — | [load\_model\_with\_memory\_saver](python/sglang/srt/model_executor/model_runner_components/load_model_utils.py#L256) | 在权重内存区域中选择 loader 并加载模型 |
| 3 | ✅ 已读 | DefaultModelLoader | [load\_model](python/sglang/srt/model_loader/loader.py#L1009) | 先创建模型结构，再加载权重并进入 eval 模式 |
| 4 | ✅ 已读 | — | [\_initialize\_model](python/sglang/srt/model_loader/loader.py#L280) | 按模型架构解析具体类并调用构造函数 |
| 5 | ✅ 已读 | Qwen3_5ForConditionalGeneration | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L2409) | 注入 Qwen3_5ForCausalLM，调用父类构造并设置位置编码信息 |
| 6 | ✅ 已读 | Qwen3VLForConditionalGeneration | [\_\_init\_\_](python/sglang/srt/models/qwen3_vl.py#L1315) | 创建文本主干、LM head、LogitsProcessor 与可选视觉模块 |
| 7 | ✅ 已读 | Qwen3_5ForCausalLM | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1790) | 创建 embedding、按配置选择两类解码层并创建最终归一化 |
| 8 | ✅ 已读 | Qwen3_5AttentionDecoderLayer | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1223) | 创建 QKV 投影、RoPE、RadixAttention、dense MLP 与归一化 |
| 9 | ✅ 已读 | QKVParallelLinear | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L1113) | 确定 QKV 合并投影的维度、头划分与权重布局 |
| 10 | ☑️ 推定 · C | ColumnParallelLinear | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L418) | 配置列并行线性层，创建权重并绑定加载回调 |
| 11 | ✅ 已读 | LinearBase | [\_\_init\_\_](python/sglang/srt/layers/linear.py#L249) | 保存线性层配置并选择 quant_method 计算实现 |
| 12 | ✅ 已读 | UnquantizedLinearMethod | [create\_weights](python/sglang/srt/layers/quantization/unquant.py#L474) | 创建并注册未量化线性层的 weight 参数及加载属性 |
| 13 | ☑️ 推定 · C | Qwen3_5LinearDecoderLayer | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1092) | 创建 GatedDeltaNet、dense MLP、归一化与层间通信组件 |
| 14 | ✅ 已读 | Qwen3_5GatedDeltaNet | [\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L333) | 创建 GDN 输入投影、卷积参数、RadixLinearAttention、门控归一化与输出投影 |
| 15 | ☑️ 推定 · B | Qwen2MoeMLP | [\_\_init\_\_](python/sglang/srt/models/qwen2_moe.py#L172) | 创建 dense gate/up、SiLU 门控与 down 投影 |
| 16 | ☑️ 推定 · B | DefaultModelLoader | [load\_weights\_and\_postprocess](python/sglang/srt/model_loader/loader.py#L1061) | 调用模型的 load_weights 并执行加载后处理 |
| 17 | ☑️ 推定 · B | Qwen3_5ForConditionalGeneration | [load\_weights](python/sglang/srt/models/qwen3_5.py#L2477) | 映射 checkpoint 名称，将 QKV、MLP 和 GDN 权重装入参数 |

## 接收请求 → 分词 → 发给 Scheduler

**已覆盖 9/9（100.0%），待覆盖 0。**

以单个非流式生成请求为主；generate_request 调用 _wait_one_response 后挂起，结果到达时才继续 yield。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 18 | ✅ 已读 | OpenAIServingChat | [\_handle\_non\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L2304) | 收集完整结果并构造 chat 响应 |
| 19 | ✅ 已读 | TokenizerManager | [generate\_request](python/sglang/srt/managers/tokenizer_manager.py#L942) | 统一请求生命周期入口 |
| 20 | ✅ 已读 | TokenizerManager | [\_init\_req\_state](python/sglang/srt/managers/tokenizer_manager.py#L3871) | 以 rid 登记请求状态 |
| 21 | ✅ 已读 | TokenizerManager | [\_tokenize\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1165) | 构造单个 tokenized 请求 |
| 22 | ✅ 已读 | TokenizerManager | [\_tokenize\_texts](python/sglang/srt/managers/tokenizer_manager.py#L1087) | 文本与 token 输入的分词路径 |
| 23 | ✅ 已读 | TokenizerManager | [\_validate\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1388) | 上下文长度与生成预算检查 |
| 24 | ✅ 已读 | SamplingParams | [normalize](python/sglang/srt/sampling/sampling_params.py#L302) | 将生成参数规范化为采样语义 |
| 25 | ✅ 已读 | TokenizerManager | [\_create\_tokenized\_object](python/sglang/srt/managers/tokenizer_manager.py#L1620) | 生成跨进程请求对象 |
| 26 | ✅ 已读 | TokenizerManager | [\_send\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1867) | 提交请求并关联返回状态 |

## Overlap 循环 → 收请求 → 入队

**已覆盖 8/8（100.0%），待覆盖 0。**

Scheduler 循环已在后台运行，TokenizerManager 通过 ZMQ 发送请求；多 rank 广播按配置执行。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 27 | ✅ 已读 | Scheduler | [event\_loop\_overlap](python/sglang/srt/managers/scheduler.py#L2004) | CPU 调度和 GPU 执行交错推进 |
| 28 | ✅ 已读 | SchedulerRequestReceiver | [recv\_requests](python/sglang/srt/managers/scheduler_components/request_receiver.py#L92) | 接收本轮请求并做 rank 同步 |
| 29 | ✅ 已读 | SchedulerRequestReceiver | [\_pull\_raw\_reqs](python/sglang/srt/managers/scheduler_components/request_receiver.py#L129) | 从主进程管道接收消息 |
| 30 | ☑️ 推定 · B | SchedulerRequestReceiver | [\_broadcast\_reqs\_across\_ranks](python/sglang/srt/managers/scheduler_components/request_receiver.py#L185) | 将请求分发到参与执行的 rank |
| 31 | ✅ 已读 | Scheduler | [process\_input\_requests](python/sglang/srt/managers/scheduler.py#L2210) | 按消息类型分派处理 |
| 32 | ✅ 已读 | Scheduler | [handle\_generate\_request](python/sglang/srt/managers/scheduler.py#L2883) | 将 tokenized 输入变为调度请求 |
| 33 | ✅ 已读 | Req | [\_\_init\_\_](python/sglang/srt/managers/schedule_batch.py#L1015) | 建立请求、生成和缓存状态 |
| 34 | ✅ 已读 | Scheduler | [\_add\_request\_to\_queue](python/sglang/srt/managers/scheduler.py#L3408) | 进入等待队列 |

## 选择 Prefill 请求 → 匹配前缀 → 建立批次

**已覆盖 11/15（73.3%），待覆盖 4。**

展示父方法后再下钻；分块 Prefill、前缀匹配属于条件路径。RadixCache 是学习代表实现，实际缓存由模型和配置选择。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 35 | ✅ 已读 | Scheduler | [get\_next\_batch\_to\_run](python/sglang/srt/managers/scheduler.py#L3793) | 选择下一轮 prefill/decode 批次 |
| 36 | ✅ 已读 | Scheduler | [get\_new\_batch\_prefill](python/sglang/srt/managers/scheduler.py#L3995) | 建立 prefill 批次入口 |
| 37 | ✅ 已读 | Scheduler | [\_get\_new\_batch\_prefill\_raw](python/sglang/srt/managers/scheduler.py#L4027) | 选择可接纳的等待请求 |
| 38 | ✅ 已读 | SchedulePolicy | [calc\_priority](python/sglang/srt/managers/schedule_policy.py#L215) | 确定等待队列处理顺序 |
| 39 | ✅ 已读 | PrefillAdder | [\_\_init\_\_](python/sglang/srt/managers/schedule_policy.py#L596) | 建立 token 与显存接纳预算 |
| 40 | ⬜ 未读 | Req | [init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1619) | 按已缓存前缀准备下轮输入 |
| 41 | ⬜ 未读 | RadixCache | [match\_prefix](python/sglang/srt/mem_cache/radix_cache.py#L373) | 查找可复用前缀 |
| 42 | ⬜ 未读 | RadixCache | [\_match\_prefix\_helper](python/sglang/srt/mem_cache/radix_cache.py#L674) | 沿树匹配 token 序列 |
| 43 | ⬜ 未读 | RadixCache | [\_split\_node](python/sglang/srt/mem_cache/radix_cache.py#L700) | 部分命中时拆分树节点 |
| 44 | ✅ 已读 | PrefillAdder | [add\_one\_req](python/sglang/srt/managers/schedule_policy.py#L1327) | 检查预算并接纳请求 |
| 45 | ✅ 已读 | RadixCache | [inc\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L596) | 保护活跃请求使用的缓存 |
| 46 | ✅ 已读 | PrefillAdder | [add\_chunked\_req](python/sglang/srt/managers/schedule_policy.py#L1099) | 继续处理分块 prefill 请求 |
| 47 | ✅ 已读 | PrefillAdder | [\_update\_prefill\_budget](python/sglang/srt/managers/schedule_policy.py#L858) | 扣减本轮 prefill 预算 |
| 48 | ✅ 已读 | ScheduleBatch | [init\_new](python/sglang/srt/managers/schedule_batch.py#L2567) | 从请求集合建立调度批次 |
| 49 | ✅ 已读 | ScheduleBatch | [prepare\_for\_extend](python/sglang/srt/managers/schedule_batch.py#L2749) | 准备 prefill token、长度和缓存位置 |

## 为 Prefill 分配 KV 空间

**已覆盖 2/4（50.0%），待覆盖 2。**

关注请求映射表行、逻辑 token 位置和物理 KV 槽位的关系；paged allocator 为代表实现。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 50 | ✅ 已读 | — | [alloc\_for\_extend](python/sglang/srt/mem_cache/allocation.py#L365) | 为新增 prefill token 分配 KV |
| 51 | ✅ 已读 | ReqToTokenPool | [alloc](python/sglang/srt/mem_cache/memory_pool.py#L438) | 为活跃请求分配映射表行 |
| 52 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [alloc\_extend](python/sglang/srt/mem_cache/allocator/paged.py#L204) | 按页接纳 prefill 的 KV 增量 |
| 53 | ⬜ 未读 | — | [assign\_req\_to\_token\_pool](python/sglang/srt/mem_cache/allocation.py#L713) | 将逻辑 token 位置映射到 KV 槽位 |

## 解析输入 → 提交前向

**已覆盖 4/4（100.0%），待覆盖 0。**

run_batch 只读 enable_overlap 分支；resolve_forward_inputs 在 worker 调用和 ForwardBatch.init_new 之前。若需关闭本轮重叠，先处理队列中的上一批结果。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 54 | ✅ 已读 | Scheduler | [run\_batch](python/sglang/srt/managers/scheduler.py#L4606) | 提交一轮模型计算并接收结果 |
| 55 | ✅ 已读 | — | [resolve\_forward\_inputs](python/sglang/srt/managers/overlap_utils.py#L87) | 将 overlap 的未来 token 解析为实际输入 |
| 56 | ✅ 已读 | TpModelWorker | [forward\_batch\_generation](python/sglang/srt/managers/tp_worker.py#L612) | 组织 ForwardBatch、前向与采样 |
| 57 | ✅ 已读 | ForwardBatch | [init\_new](python/sglang/srt/model_executor/forward_batch_info.py#L972) | 调度视图转换为 GPU 批次视图 |

## 模型前向 → Qwen 3.5 → logits 与采样

**已覆盖 24/27（88.9%），待覆盖 3。**

外层实例是 Qwen3_5ForConditionalGeneration，执行继承的 Qwen3VLForConditionalGeneration.forward。language_model_only=True 时直接调用 self.model；否则纯文本也先经 general_mm_embed_routine，再进入 qwen3_5.py 的 Qwen3_5ForCausalLM.forward。主干返回 hidden states，外层调用 LogitsProcessor 得到 logits，ModelRunner.sample 再采样。Prefill/Decode、普通 Attention/GDN、Eager/CUDA Graph 分别是不同分支；FlashAttention 是普通 Attention 后端代表。满足延迟采样条件时，sample 移到上一批结果处理之后。先读 QKV 线性投影，再看注意力缓存读写；这是阅读顺序，投影方法不调用 RadixAttention。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 58 | ✅ 已读 | ModelRunner | [forward](python/sglang/srt/model_executor/model_runner.py#L1811) | 模型前向总入口 |
| 59 | ✅ 已读 | ModelRunner | [\_forward\_raw](python/sglang/srt/model_executor/model_runner.py#L1988) | 选择图执行或 eager 路径 |
| 60 | ☑️ 推定 · C | EagerRunner | [execute](python/sglang/srt/model_executor/runner/eager_runner.py#L215) | 按 ForwardMode 分派执行 |
| 61 | ✅ 已读 | EagerRunner | [\_execute\_extend](python/sglang/srt/model_executor/runner/eager_runner.py#L289) | Prefill 的模型执行入口 |
| 62 | ✅ 已读 | EagerRunner | [\_execute\_decode](python/sglang/srt/model_executor/runner/eager_runner.py#L248) | Decode 的模型执行入口 |
| 63 | ✅ 已读 | FlashAttentionBackend | [init\_forward\_metadata](python/sglang/srt/layers/attention/flashattention_backend.py#L866) | 准备批次和 KV 索引元数据 |
| 64 | ✅ 已读 | Qwen3VLForConditionalGeneration | [forward](python/sglang/srt/models/qwen3_vl.py#L1618) | Qwen 3.5 继承的入口：执行文本主干并产出 logits |
| 65 | ☑️ 推定 · C | — | [general\_mm\_embed\_routine](python/sglang/srt/managers/mm_utils.py#L804) | 非 language_model_only 配置下，纯文本也经此完成 embedding 并调用主干 |
| 66 | ✅ 已读 | Qwen3_5ForCausalLM | [forward](python/sglang/srt/models/qwen3_5.py#L1929) | 嵌入、混合类型解码层逐层执行与最终归一化 |
| 67 | ✅ 已读 | Qwen3_5AttentionDecoderLayer | [forward](python/sglang/srt/models/qwen3_5.py#L1656) | 普通 Attention 层的注意力、MLP 与残差连接 |
| 68 | ✅ 已读 | Qwen3_5AttentionDecoderLayer | [self\_attention](python/sglang/srt/models/qwen3_5.py#L1624) | QKV 与位置编码准备、RadixAttention 和输出门控 |
| 69 | ✅ 已读 | ColumnParallelLinear | [forward](python/sglang/srt/layers/linear.py#L610) | QKVParallelLinear 继承的前向入口，调用 quant_method.apply 执行投影 |
| 70 | ✅ 已读 | UnquantizedLinearMethod | [apply](python/sglang/srt/layers/quantization/unquant.py#L501) | 未量化线性计算的普通分支，以 F.linear 执行输入与权重的投影 |
| 71 | ✅ 已读 | RadixAttention | [forward](python/sglang/srt/layers/radix_attention.py#L163) | 模型层接入 SGLang 注意力后端 |
| 72 | ✅ 已读 | FlashAttentionBackend | [forward\_extend](python/sglang/srt/layers/attention/flashattention_backend.py#L1449) | Prefill 注意力与 KV 写入 |
| 73 | ✅ 已读 | FlashAttentionBackend | [forward\_decode](python/sglang/srt/layers/attention/flashattention_backend.py#L2018) | 读取历史 KV 并计算 decode 注意力 |
| 74 | ✅ 已读 | MHATokenToKVPool | [set\_kv\_buffer](python/sglang/srt/mem_cache/memory_pool.py#L2850) | 将新 token 的 K/V 写入槽位 |
| 75 | ✅ 已读 | Qwen3_5LinearDecoderLayer | [forward](python/sglang/srt/models/qwen3_5.py#L1181) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 |
| 76 | ☑️ 推定 · C | Qwen3_5GatedDeltaNet | [forward](python/sglang/srt/models/qwen3_5.py#L955) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 |
| 77 | ✅ 已读 | RadixLinearAttention | [forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 |
| 78 | ☑️ 推定 · B | Qwen2MoeMLP | [forward](python/sglang/srt/models/qwen2_moe.py#L242) | Qwen 3.5 dense 复用的门控前馈网络 |
| 79 | ✅ 已读 | LogitsProcessor | [forward](python/sglang/srt/layers/logits_processor.py#L511) | 选择需要输出的 token 并形成 logits |
| 80 | ⬜ 未读 | LogitsProcessor | [\_get\_logits](python/sglang/srt/layers/logits_processor.py#L876) | 隐藏状态经过 LM head 得到词表分数 |
| 81 | ⬜ 未读 | LogitsProcessor | [\_compute\_lm\_head](python/sglang/srt/layers/logits_processor.py#L978) | 执行 LM head 词表投影，普通权重路径计算 hidden_states × weight.T |
| 82 | ☑️ 推定 · C | ModelRunner | [sample](python/sglang/srt/model_executor/model_runner.py#L2124) | 调用 sampler 并返回 token ID |
| 83 | ⬜ 未读 | ModelRunner | [\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L2096) | 惩罚项、掩码与 logit 偏置 |
| 84 | ✅ 已读 | Sampler | [forward](python/sglang/srt/layers/sampler.py#L142) | 贪心与概率采样的核心分派 |

## 保存本批结果 → 交接下一轮输入

**已覆盖 2/2（100.0%），待覆盖 0。**

publish 发布序列长度等元数据，stash 保存生成 token 等下一轮输入。copy_to_cpu 安排异步复制；随后将 batch 快照与结果放入 result_queue。若延迟采样，relay/stash/copy 在延迟采样后执行。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 85 | ☑️ 推定 · C | Scheduler | [\_relay\_forward\_payload](python/sglang/srt/managers/scheduler.py#L4965) | 将采样结果交给下一轮输入通道 |
| 86 | ✅ 已读 | FutureMap | [stash](python/sglang/srt/managers/overlap_utils.py#L622) | 保存生成 token 等下一轮输入数据 |

## 处理上一批结果 → 结束判断 → 缓存与释放

**已覆盖 4/13（30.8%），待覆盖 9。**

event_loop_overlap 内的 pop_and_process 从队头取出上一批并调用 process_batch_result；此处通常与本批 GPU 工作重叠。完成/未完成请求走不同分支；最后 launch_batch_sample_if_needed 仅在存在 delay_sample_func 时实际采样。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 87 | ✅ 已读 | Scheduler | [process\_batch\_result](python/sglang/srt/managers/scheduler.py#L5056) | 按批次类型处理前向结果 |
| 88 | ✅ 已读 | SchedulerBatchResultProcessor | [process\_batch\_result\_prefill](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L258) | 处理 prefill 输出和请求状态 |
| 89 | ✅ 已读 | SchedulerBatchResultProcessor | [process\_batch\_result\_decode](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L923) | 接收新 token 并推进 decode 状态 |
| 90 | ⬜ 未读 | Req | [update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1908) | 判断长度、终止 token 和停止字符串 |
| 91 | ⬜ 未读 | RadixCache | [checkpoint](python/sglang/srt/mem_cache/radix_cache.py#L519) | 为请求提交已计算 KV 前缀检查点 |
| 92 | ⬜ 未读 | — | [release\_kv\_cache](python/sglang/srt/mem_cache/common.py#L379) | 结束请求时转交缓存并释放资源 |
| 93 | ⬜ 未读 | RadixCache | [\_insert\_cache](python/sglang/srt/mem_cache/radix_cache.py#L459) | 将页对齐 KV 前缀写入树并释放重复槽位 |
| 94 | ⬜ 未读 | RadixCache | [insert](python/sglang/srt/mem_cache/radix_cache.py#L437) | 插入可复用 token 前缀 |
| 95 | ⬜ 未读 | RadixCache | [\_insert\_helper](python/sglang/srt/mem_cache/radix_cache.py#L728) | 维护前缀树节点与 KV 所有权 |
| 96 | ⬜ 未读 | RadixCache | [dec\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L620) | 释放缓存引用保护 |
| 97 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [free](python/sglang/srt/mem_cache/allocator/paged.py#L328) | 归还 KV 页面 |
| 98 | ⬜ 未读 | ReqToTokenPool | [free](python/sglang/srt/mem_cache/memory_pool.py#L487) | 归还请求映射表行 |
| 99 | ☑️ 推定 · C | Scheduler | [launch\_batch\_sample\_if\_needed](python/sglang/srt/managers/scheduler.py#L5018) | 上一批处理后按需采样本批并交接结果 |

## 请求未结束 → 准备下一轮 Decode

**已覆盖 3/10（30.0%），待覆盖 7。**

这些操作位于后续轮次的 get_next_batch_to_run 中；显存不足时才淘汰/撤回。准备后回到提交前向阶段，直到结束。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 100 | ⬜ 未读 | ScheduleBatch | [merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3800) | 合并完成 prefill 的请求与运行批次 |
| 101 | ✅ 已读 | Scheduler | [update\_running\_batch](python/sglang/srt/managers/scheduler.py#L4448) | 维护继续生成的请求集合 |
| 102 | ⬜ 未读 | ScheduleBatch | [filter\_batch](python/sglang/srt/managers/schedule_batch.py#L3712) | 移除结束请求并对齐批次张量 |
| 103 | ⬜ 未读 | ScheduleBatch | [check\_decode\_mem](python/sglang/srt/managers/schedule_batch.py#L3334) | 检查下一轮 KV 容量 |
| 104 | ✅ 已读 | RadixCache | [evict](python/sglang/srt/mem_cache/radix_cache.py#L561) | 淘汰未保护前缀以腾出 KV |
| 105 | ⬜ 未读 | ScheduleBatch | [retract\_decode](python/sglang/srt/managers/schedule_batch.py#L3353) | 显存不足时撤回部分请求 |
| 106 | ⬜ 未读 | Req | [reset\_for\_retract](python/sglang/srt/managers/schedule_batch.py#L1950) | 重置被撤回请求以便重新调度 |
| 107 | ✅ 已读 | ScheduleBatch | [prepare\_for\_decode](python/sglang/srt/managers/schedule_batch.py#L3613) | 为每个请求准备下一 token 的执行状态 |
| 108 | ⬜ 未读 | — | [alloc\_for\_decode](python/sglang/srt/mem_cache/allocation.py#L750) | 为下一轮 decode token 分配 KV |
| 109 | ⬜ 未读 | PagedTokenToKVPoolAllocator | [alloc\_decode](python/sglang/srt/mem_cache/allocator/paged.py#L279) | 处理 decode 跨页分配 |

## 输出 token → 解码文本 → 完整 HTTP 响应

**已覆盖 11/11（100.0%），待覆盖 0。**

结果处理内部发出输出，跨 ZMQ 到 Detokenizer，再回 TokenizerManager；后台 handle_loop 唤醒 _wait_one_response。非流式请求也复用 stream_output，最终由最初的 _handle_non_streaming_request 返回完整响应。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 110 | ✅ 已读 | SchedulerOutputStreamer | [stream\_output](python/sglang/srt/managers/scheduler_components/output_streamer.py#L119) | 组织生成结果并选择输出类型 |
| 111 | ☑️ 推定 · C | SchedulerOutputStreamer | [\_stream\_output\_generation](python/sglang/srt/managers/scheduler_components/output_streamer.py#L147) | 按流式间隔构造 token 输出批次 |
| 112 | ✅ 已读 | DetokenizerManager | [event\_loop](python/sglang/srt/managers/detokenizer_manager.py#L203) | 接收 token 输出并分发解码 |
| 113 | ✅ 已读 | DetokenizerManager | [handle\_batch\_token\_id\_out](python/sglang/srt/managers/detokenizer_manager.py#L490) | 解码并发送文本输出消息 |
| 114 | ✅ 已读 | DetokenizerManager | [\_decode\_batch\_token\_id\_output](python/sglang/srt/managers/detokenizer_manager.py#L344) | 维护请求的增量解码状态 |
| 115 | ✅ 已读 | DetokenizerManager | [\_grouped\_batch\_decode](python/sglang/srt/managers/detokenizer_manager.py#L278) | 将批量 token 增量转为字符串 |
| 116 | ☑️ 推定 · B | DetokenizerManager | [trim\_matched\_stop](python/sglang/srt/managers/detokenizer_manager.py#L226) | 按停止规则裁剪返回文本 |
| 117 | ✅ 已读 | TokenizerManager | [handle\_loop](python/sglang/srt/managers/tokenizer_manager.py#L2563) | 接收返回消息并按类型分发 |
| 118 | ✅ 已读 | TokenizerManager | [\_handle\_batch\_output](python/sglang/srt/managers/tokenizer_manager.py#L2584) | 更新 ReqState 并唤醒等待者 |
| 119 | ✅ 已读 | TokenizerManager | [\_wait\_one\_response](python/sglang/srt/managers/tokenizer_manager.py#L2028) | 等待并产出请求结果 |
| 120 | ☑️ 推定 · B | OpenAIServingChat | [\_build\_chat\_response](python/sglang/srt/entrypoints/openai/serving_chat.py#L2338) | 将完整生成结果包装成 OpenAI 响应 |

## 条件分支：Decode CUDA Graph

**已覆盖 0/2（0.0%），待覆盖 2。**

这是 _forward_raw 中替代 Eager 的路径，不是 HTTP 回包之后执行。capture 在图准备阶段，不属于本次运行中请求路线，因此不计入本路线。

| 序号 | 状态 | 类名 | 方法名 | 作用 |
| ---: | --- | --- | --- | --- |
| 121 | ⬜ 未读 | DecodeCudaGraphRunner | [can\_run\_graph](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L702) | 判断批次是否满足图重放条件 |
| 122 | ⬜ 未读 | DecodeCudaGraphRunner | [execute](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1529) | 重放已捕获的执行图 |

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
| 1 | D | 新增中文说明 L1246,1249,1253,1259,1260,1262,1264,1265,1267,1272,1311,1313,1361,1364 |
| 2 | D | 新增中文说明 L269,272,278,279,281,283,284,286,343,363 |
| 3 | D | 新增中文说明 L1016,1029,1031,1035,1037,1039,1041,1042,1044,1046,1047 |
| 4 | D | 新增中文说明 L286,294,300,301,303,305,306,308,311,312,334 |
| 5 | D | 新增中文说明 L2414 |
| 6 | D | 新增中文说明 L1320,1358,1372,1373,1391 |
| 7 | D | 新增中文说明 L1804,1831,1874,1877,1878 |
| 8 | D | 新增中文说明 L1226,1235,1249,1250,1251,1279,1291,1295,1296,1300 |
| 9 | D | 新增中文说明 L1122 |
| 10 | C | chain=qwen_construct_qkv_base;anchors=qkv_projection_init->linear_base_init \| chain=qwen_construct_qkv_weights;anchors=qkv_projection_init->unquant_linear_weights |
| 11 | D | 新增中文说明 L268 |
| 12 | D | 新增中文说明 L476,493 |
| 13 | C | chain=qwen_construct_linear;anchors=qwen_model_init->qwen_gdn_init |
| 14 | D | 新增中文说明 L467 |
| 15 | B | branch=model_construction;direct=5;covered_before_branch=6/7 |
| 16 | B | branch=model_loading;direct=4;covered_before_branch=4/6 |
| 17 | B | branch=model_loading;direct=4;covered_before_branch=4/6 |
| 18 | D | 新增中文说明 L2312,2313,2314,2315,2316,2319,2320 |
| 19 | D | 新增中文说明 L948,951,952,957,980,987,991,994,998,1000,1001,1002,1007,1011,1013,1015,1019,1032 |
| 20 | D | 新增中文说明 L3877,3893,3895,3912,3917,3918 |
| 21 | D | 新增中文说明 L1178,1182,1219,1222,1346,1348,1349,1350,1351,1352,1353,1354 |
| 22 | D | 新增中文说明 L1093,1133 |
| 23 | D | 新增中文说明 L1393,1394,1395 |
| 24 | D | 新增中文说明 L353 |
| 25 | D | 新增中文说明 L1621,1624,1625,1626,1629,1630,1633,1636,1639,1640,1650,1663,1664,1665,1675,1683 |
| 26 | D | 新增中文说明 L1871,1886 |
| 27 | D | 新增中文说明 L2003,2006,2008,2009,2010,2011,2012,2017,2018,2021,2032,2038,2045,2047,2048,2050,2055,2056,2057,2059,2060,2061,2063,2064,2065,2066,2067,2068,2070,2092,2093,2102 |
| 28 | D | 新增中文说明 L96,109,110,116,120,122 |
| 29 | D | 新增中文说明 L130,133,149,159,168 |
| 30 | B | branch=scheduler_ingress;direct=6;covered_before_branch=6/7 |
| 31 | D | 新增中文说明 L2212,2213,2214,2219,2238 |
| 32 | D | 新增中文说明 L2890,2897,2900,2904,2911,2913,2963,2978,2979,2980,3072,3074,3257 |
| 33 | D | 新增中文说明 L1194,1197,1214,1218,1326 |
| 34 | D | 新增中文说明 L3409,3422,3426 |
| 35 | D | 新增中文说明 L3797,3799,3800,3801,3802,3803,3804,3805,3806,3807,3808,3809,3810,3811,3812,3813,3814,3815,3816,3817,3818,3837,3838,3839,3851,3852,3853,3880,3886,3891,3908,3909,3928,3929,3931,3934,3936 |
| 36 | D | 新增中文说明 L4005,4013,4017,4023,4024 |
| 37 | D | 新增中文说明 L4033,4046,4047,4050,4051,4052,4086,4099,4101,4102,4104,4105,4109,4123,4124,4127,4129,4130,4131,4132,4133,4135,4138,4139,4148,4149,4177,4210,4217,4224,4249,4297,4301,4323,4325,4326,4335,4340,4347 |
| 38 | D | 新增中文说明 L221,222,230,234 |
| 39 | D | 新增中文说明 L605,615,616,620,626,627,650 |
| 40 | U | 无直接证据，且未满足补全规则 |
| 41 | U | 无直接证据，且未满足补全规则 |
| 42 | U | 无直接证据，且未满足补全规则 |
| 43 | U | 无直接证据，且未满足补全规则 |
| 44 | D | 新增中文说明 L1330,1344,1345,1347,1358,1486 |
| 45 | D | 新增中文说明 L597,601,602,604,606,607,608,609,610,611,614,615,616 |
| 46 | D | 新增中文说明 L1100,1150 |
| 47 | D | 新增中文说明 L861,870,906 |
| 48 | D | 新增中文说明 L2570,2572,2580,2582,2585,2586,2590,2592,2593,2594,2597,2598,2599,2601,2602 |
| 49 | D | 新增中文说明 L2750,2752,2761,2763,2765,2767,2769,2771,2773,2776,2827,2830,2848,2980,2983,2992,2993,2996,2999,3060 |
| 50 | D | 新增中文说明 L375,376,377,378,379,380,381,384,385,387,390,393,412,424,438,442,460,474,493 |
| 51 | D | 新增中文说明 L440,455 |
| 52 | U | 无直接证据，且未满足补全规则 |
| 53 | U | 无直接证据，且未满足补全规则 |
| 54 | D | 新增中文说明 L4608,4612,4632,4649,4651,4658,4659,4660,4661,4663,4670,4694,4695,4699,4873,4876,4882,4886,4888,4890,4896,4911 |
| 55 | D | 新增中文说明 L118,119,120 |
| 56 | D | 新增中文说明 L614,624,639,641,660,663,682,708,711 |
| 57 | D | 新增中文说明 L984,1039,1042,1044,1046,1048,1050,1052 |
| 58 | D | 新增中文说明 L1813,1819,1821,1833,1836,1837,1839,1842,1843,1844,1846,1847,1884 |
| 59 | D | 新增中文说明 L1990,1996,2009,2026,2083 |
| 60 | C | chain=eager_extend_model;anchors=raw_forward->eager_extend \| chain=eager_decode_model;anchors=raw_forward->eager_decode \| chain=eager_extend_linear_model;anchors=raw_forward->eager_extend \| chain=eager_decode_linear_model;anchors=raw_forward->eager_decode |
| 61 | D | 新增中文说明 L291,295,400,401 |
| 62 | D | 新增中文说明 L254,255,256,257,258,259,260,279,280,281,283 |
| 63 | D | 新增中文说明 L874,878,880,886,887 |
| 64 | D | 新增中文说明 L1620,1622,1627,1643,1644,1646,1647,1648,1649,1682,1685,1686,1688,1692,1696 |
| 65 | C | chain=qwen_text_embedding;anchors=causal_lm->qwen_model |
| 66 | D | 新增中文说明 L1948,1966,1971 |
| 67 | D | 新增中文说明 L1664,1680,1685 |
| 68 | D | 新增中文说明 L1627,1632,1651,1652 |
| 69 | D | 新增中文说明 L624,626,632,633,634 |
| 70 | D | 新增中文说明 L508,547 |
| 71 | D | 新增中文说明 L169,177,184,299 |
| 72 | D | 新增中文说明 L2010 |
| 73 | D | 新增中文说明 L2040,2043,2044,2045,2046,2047,2048,2051,2052,2053,2054,2055,2057,2058,2060,2061,2065,2067,2068,2069,2070,2071,2073,2074,2075,2077,2078,2092,2093,2094,2095,2097,2098,2108,2197,2245,2266,2268,2270,2272,2279,2281,2299,2300,2301,2302,2304,2436 |
| 74 | D | 新增中文说明 L2853,2854,2855,2856,2857,2858,2859,2862,2864,2870,2890,2891,2894,2900,2939 |
| 75 | D | 新增中文说明 L1202,1208 |
| 76 | C | chain=eager_extend_linear_model;anchors=qwen_linear_layer->radix_linear_attention \| chain=eager_decode_linear_model;anchors=qwen_linear_layer->radix_linear_attention |
| 77 | D | 新增中文说明 L142,143 |
| 78 | B | branch=dense_model;direct=6;covered_before_branch=8/9 |
| 79 | D | 新增中文说明 L514,520,522,523,524,525,526 |
| 80 | U | 无直接证据，且未满足补全规则 |
| 81 | U | 无直接证据，且未满足补全规则 |
| 82 | C | chain=worker_sample;anchors=worker_forward->sample \| chain=overlap_delayed_sample;anchors=overlap_loop->sample |
| 83 | U | 无直接证据，且未满足补全规则 |
| 84 | D | 新增中文说明 L152 |
| 85 | C | chain=overlap_relay;anchors=run_batch->future_stash |
| 86 | D | 新增中文说明 L623 |
| 87 | D | 新增中文说明 L5061 |
| 88 | D | 新增中文说明 L263 |
| 89 | D | 新增中文说明 L928 |
| 90 | U | 无直接证据，且未满足补全规则 |
| 91 | U | 无直接证据，且未满足补全规则 |
| 92 | U | 无直接证据，且未满足补全规则 |
| 93 | U | 无直接证据，且未满足补全规则 |
| 94 | U | 无直接证据，且未满足补全规则 |
| 95 | U | 无直接证据，且未满足补全规则 |
| 96 | U | 无直接证据，且未满足补全规则 |
| 97 | U | 无直接证据，且未满足补全规则 |
| 98 | U | 无直接证据，且未满足补全规则 |
| 99 | C | chain=overlap_delayed_sample;anchors=overlap_loop->sample |
| 100 | U | 无直接证据，且未满足补全规则 |
| 101 | D | 新增中文说明 L4450,4541 |
| 102 | U | 无直接证据，且未满足补全规则 |
| 103 | U | 无直接证据，且未满足补全规则 |
| 104 | D | 新增中文说明 L562,567,569,571,579 |
| 105 | U | 无直接证据，且未满足补全规则 |
| 106 | U | 无直接证据，且未满足补全规则 |
| 107 | D | 新增中文说明 L3636,3656,3664 |
| 108 | U | 无直接证据，且未满足补全规则 |
| 109 | U | 无直接证据，且未满足补全规则 |
| 110 | D | 新增中文说明 L126 |
| 111 | C | chain=token_message;anchors=stream_output->detokenizer_loop |
| 112 | D | 新增中文说明 L205,206,207,211,216,217,221 |
| 113 | D | 新增中文说明 L501,505,506,512 |
| 114 | D | 新增中文说明 L345,350,354,357,358,364,367,381,383,418 |
| 115 | D | 新增中文说明 L309,315 |
| 116 | B | branch=detokenize;direct=4;covered_before_branch=4/5 |
| 117 | D | 新增中文说明 L2566,2570,2571,2577 |
| 118 | D | 新增中文说明 L2586,2592,2600,2602,2614,2726,2738,2742,2743,2884,2890,2896,2909 |
| 119 | D | 新增中文说明 L2033 |
| 120 | B | branch=response;direct=4;covered_before_branch=4/5 |
| 121 | U | 无直接证据，且未满足补全规则 |
| 122 | U | 无直接证据，且未满足补全规则 |

</details>

- 路线版本：overlap_nonstream v5；固定分母：122。
- 统计对象：当前工作区，包含未提交修改；日期不作为已读证据。
- HEAD：`41e029e21bbd7c2552f1cee21792ed04e84638f4`。
- 固定对照基线：`6388b6cfb1d93c253714a408f1d66a093302acd7`。
- 源码指纹：`389f81c3df1d0c0b18eab3fc3e5718c53bd45d0c456a9cb3c7764eedcddbd500`。
- 范围指纹：`1511df5b4dab2f09af9359a2c997d123534a504e67afca7194de650bf1cdfdb4`。
