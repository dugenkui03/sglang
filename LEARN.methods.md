# SGLang 核心方法阅读清单

当前模型固定为 `Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`；初始化、权重加载和文本推理的对象关系见 [LEARN.md 第 3.1 节](LEARN.md#31-固定模型qwen3_5forconditionalgeneration)。本页是全核心总览；当前默认的模型准备与非流式 Overlap 路线见 [LEARN.overlap.md](LEARN.overlap.md)，两者独立计分。

**已读 119/160（74.4%），未读 41 个。**

直接注释 101 个；链路补全 11 个；分支补全 7 个。方法按核心阶段和概念分支排列，可点击方法名跳转源码。

| 阅读标记 | 判定依据 | 含义 |
| --- | --- | --- |
| ✅ 已读 | D · 直接注释 | 方法已有新增中文学习说明。 |
| ✅ 已读（推定） | C · 链路补全 | 位于同一链路的两个直接证据点之间。 |
| ✅ 已读（推定） | B · 分支补全 | 所属分支满足覆盖阈值，按规则视为已读。 |
| ⬜ 未读 | U · 待覆盖 | 当前未找到直接证据，也未满足补全规则。 |

这里的“未读”表示统计尚未覆盖；已读包含按约定推定的节点。每个方法等权计 1。
分支补全要求至少 2 个 D，且 (D+C)/分支方法数 ≥ 2/3；C/B 不作为新的链路端点。完整规则见 [LEARN.md](LEARN.md)，固定范围见 [LEARN.scope.json](quick_learn/LEARN.scope.json)。

本清单由 [统计脚本](quick_learn/update_learn_progress.py) 自动生成。在源码保留学习注释后，运行 `python quick_learn/update_learn_progress.py --write` 更新；手动修改本清单不会作为阅读证据，重算时会被覆盖。

## 阶段总览

| 核心阶段 | 已读 | 未读 | 覆盖率 |
| --- | ---: | ---: | ---: |
| 1 启动与进程 | 27/31 | 4 | 87.1% |
| 2 配置发布 | 2/4 | 2 | 50.0% |
| 3 API 与请求转换 | 6/6 | 0 | 100.0% |
| 4 分词与请求提交 | 10/10 | 0 | 100.0% |
| 5 调度与组批 | 23/29 | 6 | 79.3% |
| 6 执行桥接与 overlap 数据交接 | 7/7 | 0 | 100.0% |
| 7 模型前向与 Attention | 15/24 | 9 | 62.5% |
| 8 KV 分配与前缀缓存 | 9/24 | 15 | 37.5% |
| 9 采样、结果处理与回包 | 20/25 | 5 | 80.0% |

## 1. 启动与进程

### 启动三类进程及 HTTP 服务

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [launch\_server](python/sglang/srt/entrypoints/http_server.py#L2811) | 服务启动总入口 | D · 直接注释：新增中文说明 L2812,2814,2815,2817,2818,2820,2822,2824,2833,2844,2845,2853,2874 |
| ✅ 已读 | [Engine.\_launch\_subprocesses](python/sglang/srt/entrypoints/engine.py#L1051) | 编排进程启动与就绪 | D · 直接注释：新增中文说明 L1067,1068,1069,1070,1071,1072,1083,1092,1102,1103,1111,1126,1128,1153,1154,1163,1166,1174,1175,1176,1180,1182,1233,1239,1246,1247,1260,1266,1276,1283,1284,1285,1289 |
| ✅ 已读 | [Engine.\_launch\_scheduler\_processes](python/sglang/srt/entrypoints/engine.py#L838) | 按 rank 拉起 GPU worker | D · 直接注释：新增中文说明 L875,876,878,879,880,886,887,888,894,896,900,914,960 |
| ✅ 已读 | [Engine.\_launch\_detokenizer\_subprocesses](python/sglang/srt/entrypoints/engine.py#L964) | 拉起解码回包进程 | D · 直接注释：新增中文说明 L984,986 |
| ✅ 已读 | [init\_tokenizer\_manager](python/sglang/srt/entrypoints/engine.py#L156) | 建立主进程请求入口 | D · 直接注释：新增中文说明 L159,161,162,163,164,168,170,171,173,174,175 |
| ✅ 已读 | [run\_scheduler\_process](python/sglang/srt/managers/scheduler.py#L5614) | Scheduler 子进程入口 | D · 直接注释：新增中文说明 L5630,5635,5640,5671,5688 |
| ✅ 已读 | [run\_detokenizer\_process](python/sglang/srt/managers/detokenizer_manager.py#L610) | Detokenizer 子进程入口 | D · 直接注释：新增中文说明 L625 |
| ✅ 已读 | [\_setup\_and\_run\_http\_server](python/sglang/srt/entrypoints/http_server.py#L2549) | 将 Engine 组件接入 HTTP 服务 | D · 直接注释：新增中文说明 L2565,2566,2697 |

### GPU 执行组件初始化与模型加载

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.\_\_init\_\_](python/sglang/srt/managers/scheduler.py#L413) | 调度器资源与组件初始化 | D · 直接注释：新增中文说明 L542,543,550,551,587,672 |
| ✅ 已读 | [Scheduler.init\_model\_worker](python/sglang/srt/managers/scheduler.py#L1060) | 编排权重、缓存、后端和图初始化 | D · 直接注释：新增中文说明 L1061,1062,1063,1064,1065,1068,1076,1081,1139,1140 |
| ✅ 已读 | [Scheduler.init\_tp\_model\_worker](python/sglang/srt/managers/scheduler.py#L946) | 建立调度到 GPU 的执行桥 | D · 直接注释：新增中文说明 L947,948,949,950,961,968 |
| ✅ 已读 | [TpModelWorker.\_\_init\_\_](python/sglang/srt/managers/tp_worker.py#L318) | 建立模型配置与 ModelRunner | D · 直接注释：新增中文说明 L355,356 |
| ✅ 已读 | [ModelRunner.\_\_init\_\_](python/sglang/srt/model_executor/model_runner.py#L314) | GPU 执行状态与设备配置 | D · 直接注释：新增中文说明 L329,430,464,472,478,479,481,483,484,486 |
| ✅ 已读 | [ModelRunner.initialize](python/sglang/srt/model_executor/model_runner.py#L670) | 模型执行环境初始化顺序 | D · 直接注释：新增中文说明 L671,680,682,699,700 |
| ✅ 已读 | [ModelRunner.load\_model](python/sglang/srt/model_executor/model_runner.py#L1163) | 将模型权重与结构加载到执行器 | D · 直接注释：新增中文说明 L1164,1167,1171,1177,1178,1180,1182,1183,1185,1190,1217,1235,1238,1285,1288 |

### Qwen 3.5 模型构造入口与权重加载

已读 **5/5**，未读 **0**。

代表路径为常规 DefaultModelLoader；load_model 内先调用 _initialize_model 完成整个对象树构造，再调用 load_weights_and_postprocess。构造与加载是两个阶段，不把构造方法和 load_weights 串成父子调用。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [load\_model\_with\_memory\_saver](python/sglang/srt/model_executor/model_runner_components/load_model_utils.py#L265) | 在权重内存区域中选择 loader 并加载模型 | D · 直接注释：新增中文说明 L279,282,288,289,291,293,294,296,321,341 |
| ✅ 已读 | [DefaultModelLoader.load\_model](python/sglang/srt/model_loader/loader.py#L962) | 先创建模型结构，再加载权重并进入 eval 模式 | D · 直接注释：新增中文说明 L969,982,984,988,990,992,994,995,997,999,1000 |
| ✅ 已读 | [\_initialize\_model](python/sglang/srt/model_loader/loader.py#L271) | 按模型架构解析具体类并调用构造函数 | D · 直接注释：新增中文说明 L277,285,291,292,294,296,297,299,302,303,325,327,334,335 |
| ✅ 已读 | [DefaultModelLoader.load\_weights\_and\_postprocess](python/sglang/srt/model_loader/loader.py#L1014) | 调用模型的 load_weights 并执行加载后处理 | D · 直接注释：新增中文说明 L1058 |
| ✅ 已读（推定） | [Qwen3\_5ForConditionalGeneration.load\_weights](python/sglang/srt/models/qwen3_5.py#L1972) | 映射 checkpoint 名称，将 QKV、MLP 和 GDN 权重装入参数 | B · 分支补全：branch=model_loading;direct=4;covered_before_branch=4/5 |

### Qwen3_5ForConditionalGeneration 的对象初始化

已读 **4/7**，未读 **3**。

入口把 language_model_cls=Qwen3_5ForCausalLM 传给父类；父类用 config.text_config 创建 self.model，并持有 lm_head 与 logits_processor。两类层按 layers_block_type 分支创建，dense MLP 被两类层复用。这里只认识视觉模块的创建条件，不展开视觉计算。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Qwen3\_5ForConditionalGeneration.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1914) | 注入 Qwen3_5ForCausalLM，调用父类构造并设置位置编码信息 | D · 直接注释：新增中文说明 L1919 |
| ✅ 已读 | [Qwen3VLForConditionalGeneration.\_\_init\_\_](python/sglang/srt/models/qwen3_vl.py#L1260) | 创建文本主干、LM head、LogitsProcessor 与可选视觉模块 | D · 直接注释：新增中文说明 L1265,1303,1317,1318,1336 |
| ✅ 已读 | [Qwen3\_5ForCausalLM.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1442) | 创建 embedding、按配置选择两类解码层并创建最终归一化 | D · 直接注释：新增中文说明 L1456,1497,1503,1506,1507,1509,1510 |
| ✅ 已读（推定） | [Qwen3\_5AttentionDecoderLayer.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L938) | 创建 QKV 投影、RoPE、RadixAttention、dense MLP 与归一化 | C · 链路补全：chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| ⬜ 未读 | [Qwen3\_5LinearDecoderLayer.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L795) | 创建 GatedDeltaNet、dense MLP、归一化与层间通信组件 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5GatedDeltaNet.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L267) | 创建 GDN 输入投影、卷积参数、RadixLinearAttention、门控归一化与输出投影 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen2MoeMLP.\_\_init\_\_](python/sglang/srt/models/qwen2_moe.py#L177) | 创建 dense gate/up、SiLU 门控与 down 投影 | U · 待覆盖：无直接证据，且未满足补全规则 |

### QKV 投影与线性层权重创建

已读 **3/4**，未读 **1**。

QKVParallelLinear 继承 ColumnParallelLinear 和 LinearBase。ColumnParallelLinear.__init__ 先调用父类构造选择计算实现，返回后再调用 create_weights；两者是独立调用分支。这里只读未量化代表路径，共用方法只计一次。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读（推定） | [QKVParallelLinear.\_\_init\_\_](python/sglang/srt/layers/linear.py#L970) | 确定 QKV 合并投影的维度、头划分与权重布局 | C · 链路补全：chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| ✅ 已读（推定） | [ColumnParallelLinear.\_\_init\_\_](python/sglang/srt/layers/linear.py#L334) | 配置列并行线性层，创建权重并绑定加载回调 | C · 链路补全：chain=qwen_construct_qkv_weights;anchors=qwen_model_init->unquant_linear_weights |
| ⬜ 未读 | [LinearBase.\_\_init\_\_](python/sglang/srt/layers/linear.py#L167) | 保存线性层配置并选择 quant_method 计算实现 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [UnquantizedLinearMethod.create\_weights](python/sglang/srt/layers/quantization/unquant.py#L228) | 创建并注册未量化线性层的 weight 参数及加载属性 | D · 直接注释：新增中文说明 L247 |


## 2. 配置发布

### 从启动参数到本进程配置

已读 **2/4**，未读 **2**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ServerArgs.resolve\_once](python/sglang/srt/server_args.py#L3675) | 一次性解析启动配置 | D · 直接注释：新增中文说明 L3687,3690 |
| ✅ 已读 | [publish](python/sglang/srt/runtime_context.py#L1323) | 向本进程发布运行配置 | D · 直接注释：新增中文说明 L1339 |
| ⬜ 未读 | [RuntimeContext.set\_server\_args](python/sglang/srt/runtime_context.py#L807) | 构造可读取的配置分组 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PortArgs.init\_new](python/sglang/srt/server_args.py#L11197) | 分配组件间通信地址 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 3. API 与请求转换

### 生成 API 到内部请求

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [generate\_request](python/sglang/srt/entrypoints/http_server.py#L918) | 原生生成入口与流式选择 | D · 直接注释：新增中文说明 L920,924,961 |
| ✅ 已读 | [openai\_v1\_chat\_completions](python/sglang/srt/entrypoints/http_server.py#L1753) | OpenAI chat 入口 | D · 直接注释：新增中文说明 L1750,1751,1754,1755,1759,1760,1763 |
| ✅ 已读 | [OpenAIServingBase.handle\_request](python/sglang/srt/entrypoints/openai/serving_base.py#L73) | 校验、转换和响应路径选择 | D · 直接注释：新增中文说明 L79,95,96,97,98,99,101,104,109,119 |
| ✅ 已读 | [OpenAIServingChat.\_convert\_to\_internal\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L972) | Chat 请求转换为 GenerateReqInput | D · 直接注释：新增中文说明 L978,980,981,983,986 |
| ✅ 已读（推定） | [OpenAIServingChat.\_process\_messages](python/sglang/srt/entrypoints/openai/serving_chat.py#L1117) | 会话消息转换为模型输入 | B · 分支补全：branch=api;direct=4;covered_before_branch=4/6 |
| ✅ 已读（推定） | [OpenAIServingChat.\_apply\_jinja\_template](python/sglang/srt/entrypoints/openai/serving_chat.py#L1223) | 应用模型会话模板 | B · 分支补全：branch=api;direct=4;covered_before_branch=4/6 |


## 4. 分词与请求提交

### 主进程请求处理组件

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.\_\_init\_\_](python/sglang/srt/managers/tokenizer_manager.py#L427) | 建立请求处理状态与组件 | D · 直接注释：新增中文说明 L435,453,455,459,460,464,465,469,470,474,478,482,486,490,494,499 |
| ✅ 已读 | [TokenizerManager.init\_tokenizer\_and\_processor](python/sglang/srt/managers/tokenizer_manager.py#L521) | 初始化模型分词器 | D · 直接注释：新增中文说明 L523,530,549,553,554,560,564,566,567,568,569 |
| ✅ 已读 | [TokenizerManager.init\_ipc\_channels](python/sglang/srt/managers/tokenizer_manager.py#L604) | 建立请求与回包管道 | D · 直接注释：新增中文说明 L606,610,611,612,613,615,616,617,618,621,622,623,625,630,631,632,633,638 |

### 请求登记、分词与提交

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.generate\_request](python/sglang/srt/managers/tokenizer_manager.py#L850) | 统一请求生命周期入口 | D · 直接注释：新增中文说明 L856,859,860,865,888,891,895,898,902,904,905,906,911,915,917,918,920,921,922,923,924,928,930,940 |
| ✅ 已读 | [TokenizerManager.\_init\_req\_state](python/sglang/srt/managers/tokenizer_manager.py#L3582) | 以 rid 登记请求状态 | D · 直接注释：新增中文说明 L3588,3604,3606,3623,3628,3629 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1069) | 构造单个 tokenized 请求 | D · 直接注释：新增中文说明 L1081,1085,1110,1113,1116,1241,1243,1244,1245,1246,1247,1248,1249 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_texts](python/sglang/srt/managers/tokenizer_manager.py#L995) | 文本与 token 输入的分词路径 | D · 直接注释：新增中文说明 L1001,1037 |
| ✅ 已读 | [TokenizerManager.\_validate\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1283) | 上下文长度与生成预算检查 | D · 直接注释：新增中文说明 L1288,1289,1290 |
| ✅ 已读 | [TokenizerManager.\_create\_tokenized\_object](python/sglang/srt/managers/tokenizer_manager.py#L1462) | 生成跨进程请求对象 | D · 直接注释：新增中文说明 L1463,1466,1467,1468,1471,1472,1475,1478,1481,1482,1492,1505,1506,1507,1517,1525 |
| ✅ 已读 | [TokenizerManager.\_send\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1704) | 提交请求并关联返回状态 | D · 直接注释：新增中文说明 L1708,1721 |


## 5. 调度与组批

### 接收、分发与入队

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SchedulerRequestReceiver.recv\_requests](python/sglang/srt/managers/scheduler_components/request_receiver.py#L88) | 接收本轮请求并做 rank 同步 | D · 直接注释：新增中文说明 L92,102,103,109,110,111,114,116,118 |
| ✅ 已读 | [SchedulerRequestReceiver.\_pull\_raw\_reqs](python/sglang/srt/managers/scheduler_components/request_receiver.py#L125) | 从主进程管道接收消息 | D · 直接注释：新增中文说明 L127,131,148,158,167,168 |
| ✅ 已读（推定） | [SchedulerRequestReceiver.\_broadcast\_reqs\_across\_ranks](python/sglang/srt/managers/scheduler_components/request_receiver.py#L183) | 将请求分发到参与执行的 rank | B · 分支补全：branch=scheduler_ingress;direct=6;covered_before_branch=6/7 |
| ✅ 已读 | [Scheduler.process\_input\_requests](python/sglang/srt/managers/scheduler.py#L2057) | 按消息类型分派处理 | D · 直接注释：新增中文说明 L2059,2060,2061,2065,2070,2082,2083 |
| ✅ 已读 | [Scheduler.handle\_generate\_request](python/sglang/srt/managers/scheduler.py#L2608) | 将 tokenized 输入变为调度请求 | D · 直接注释：新增中文说明 L2613,2618,2621,2626,2638,2640,2687,2702,2703,2704,2770,2772,2940 |
| ✅ 已读 | [Req.\_\_init\_\_](python/sglang/srt/managers/schedule_batch.py#L839) | 建立请求、生成和缓存状态 | D · 直接注释：新增中文说明 L1025,1029,1043,1047,1142 |
| ✅ 已读 | [Scheduler.\_add\_request\_to\_queue](python/sglang/srt/managers/scheduler.py#L3000) | 进入等待队列 | D · 直接注释：新增中文说明 L3001,3008,3011 |

### 普通与重叠调度主循环

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [dispatch\_event\_loop](python/sglang/srt/managers/scheduler.py#L5499) | 选择运行主循环 | D · 直接注释：新增中文说明 L5501,5506,5510,5513,5517,5518,5519,5520,5523,5524,5525,5526,5529,5537 |
| ✅ 已读 | [Scheduler.run\_event\_loop](python/sglang/srt/managers/scheduler.py#L1791) | 按服务模式选择事件循环 | D · 直接注释：新增中文说明 L1794,1795,1806,1831 |
| ✅ 已读 | [Scheduler.event\_loop\_normal](python/sglang/srt/managers/scheduler.py#L1848) | 收请求、组批、前向、结果处理 | D · 直接注释：新增中文说明 L1855,1863,1868,1870 |
| ✅ 已读 | [Scheduler.event\_loop\_overlap](python/sglang/srt/managers/scheduler.py#L1889) | CPU 调度和 GPU 执行交错推进 | D · 直接注释：新增中文说明 L1888,1891,1893,1894,1895,1896,1897,1901,1902,1905,1916,1922,1923,1926,1932,1934,1935,1937,1942,1943,1944,1946,1947,1949,1950,1951,1952,1953,1954,1956,1978,1979,1988 |

### Prefill 选择与预算控制

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.get\_next\_batch\_to\_run](python/sglang/srt/managers/scheduler.py#L3300) | 选择下一轮 prefill/decode 批次 | D · 直接注释：新增中文说明 L3304,3306,3307,3308,3309,3310,3311,3312,3313,3314,3315,3316,3317,3318,3319,3320,3321,3322,3323,3324,3325,3350,3351,3352,3365,3366,3367,3394,3400,3405,3422,3423,3441,3442,3444,3447,3449,3450,3452 |
| ✅ 已读 | [Scheduler.get\_new\_batch\_prefill](python/sglang/srt/managers/scheduler.py#L3503) | 建立 prefill 批次入口 | D · 直接注释：新增中文说明 L3513,3521,3531,3532 |
| ✅ 已读 | [Scheduler.\_get\_new\_batch\_prefill\_raw](python/sglang/srt/managers/scheduler.py#L3535) | 选择可接纳的等待请求 | D · 直接注释：新增中文说明 L3541,3557,3558,3561,3562,3563,3597,3606,3608,3609,3611,3612,3618,3619,3632,3633,3636,3638,3639,3640,3641,3642,3644,3647,3648,3657,3658,3679,3703,3711,3713,3717,3720,3723,3743,3789,3793,3811,3813,3814,3823,3824,3825,3830,3837 |
| ✅ 已读 | [SchedulePolicy.calc\_priority](python/sglang/srt/managers/schedule_policy.py#L242) | 确定等待队列处理顺序 | D · 直接注释：新增中文说明 L245,246,256,260 |
| ✅ 已读 | [PrefillAdder.\_\_init\_\_](python/sglang/srt/managers/schedule_policy.py#L527) | 建立 token 与显存接纳预算 | D · 直接注释：新增中文说明 L536,546,547,551,557,558,571 |
| ✅ 已读 | [PrefillAdder.add\_one\_req](python/sglang/srt/managers/schedule_policy.py#L1246) | 检查预算并接纳请求 | D · 直接注释：新增中文说明 L1250,1271,1272,1274,1275,1276,1277,1279,1280,1285,1294,1297,1301,1336,1339,1340,1393,1395,1396,1426,1427,1435,1441,1443,1445,1446,1447,1449,1450,1452,1457,1489,1508 |
| ✅ 已读 | [PrefillAdder.add\_chunked\_req](python/sglang/srt/managers/schedule_policy.py#L1038) | 继续处理分块 prefill 请求 | D · 直接注释：新增中文说明 L1074 |
| ✅ 已读 | [PrefillAdder.\_update\_prefill\_budget](python/sglang/srt/managers/schedule_policy.py#L888) | 扣减本轮 prefill 预算 | D · 直接注释：新增中文说明 L891,898,908,909,911,912,913,916,917,935 |

### Prefill 与 decode 批次状态

已读 **4/7**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [Req.init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1343) | 按已缓存前缀准备下轮输入 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [ScheduleBatch.init\_new](python/sglang/srt/managers/schedule_batch.py#L2278) | 从请求集合建立调度批次 | D · 直接注释：新增中文说明 L2281,2283,2291,2293,2296,2297,2301,2304,2305,2308,2309,2310,2312,2313 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_extend](python/sglang/srt/managers/schedule_batch.py#L2465) | 准备 prefill token、长度和缓存位置 | D · 直接注释：新增中文说明 L2466,2468,2477,2479,2481,2483,2485,2487,2489,2492,2493,2524,2544,2682,2743 |
| ✅ 已读 | [Scheduler.update\_running\_batch](python/sglang/srt/managers/scheduler.py#L3911) | 维护继续生成的请求集合 | D · 直接注释：新增中文说明 L3913,3993 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_decode](python/sglang/srt/managers/schedule_batch.py#L3223) | 为每个请求准备下一 token 的执行状态 | D · 直接注释：新增中文说明 L3262 |
| ⬜ 未读 | [ScheduleBatch.filter\_batch](python/sglang/srt/managers/schedule_batch.py#L3319) | 移除结束请求并对齐批次张量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3406) | 合并完成 prefill 的请求与运行批次 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Decode 显存不足与回退

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [ScheduleBatch.check\_decode\_mem](python/sglang/srt/managers/schedule_batch.py#L2960) | 检查下一轮 KV 容量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.retract\_decode](python/sglang/srt/managers/schedule_batch.py#L2967) | 显存不足时撤回部分请求 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Req.reset\_for\_retract](python/sglang/srt/managers/schedule_batch.py#L1716) | 重置被撤回请求以便重新调度 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 6. 执行桥接与 overlap 数据交接

### 调度批次进入 GPU 执行

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.run\_batch](python/sglang/srt/managers/scheduler.py#L4058) | 提交一轮模型计算并接收结果 | D · 直接注释：新增中文说明 L4060,4064,4075,4092,4094,4101,4102,4103,4104,4106,4113,4137,4138,4142,4238,4241,4247,4251,4253,4255,4261,4277 |
| ✅ 已读 | [TpModelWorker.forward\_batch\_generation](python/sglang/srt/managers/tp_worker.py#L603) | 组织 ForwardBatch、前向与采样 | D · 直接注释：新增中文说明 L605,614,621,623,642,645,663,688,691 |
| ✅ 已读 | [ForwardBatch.init\_new](python/sglang/srt/model_executor/forward_batch_info.py#L744) | 调度视图转换为 GPU 批次视图 | D · 直接注释：新增中文说明 L804,807,809,811,813,815,817 |
| ✅ 已读 | [resolve\_forward\_inputs](python/sglang/srt/managers/overlap_utils.py#L87) | 将 overlap 的未来 token 解析为实际输入 | D · 直接注释：新增中文说明 L109,110,111 |

### Overlap 结果与后续输入衔接

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [FutureMap.\_\_init\_\_](python/sglang/srt/managers/overlap_utils.py#L260) | 建立未来输出槽位映射 | D · 直接注释：新增中文说明 L288,289,290,291,292,293,297,298,299,300,301 |
| ✅ 已读（推定） | [FutureMap.publish](python/sglang/srt/managers/overlap_utils.py#L532) | 发布 GPU 输出供后续批次使用 | B · 分支补全：branch=future_map;direct=2;covered_before_branch=2/3 |
| ✅ 已读 | [FutureMap.stash](python/sglang/srt/managers/overlap_utils.py#L557) | 保存可复用的异步执行结果 | D · 直接注释：新增中文说明 L558 |


## 7. 模型前向与 Attention

### 模型执行路径分派

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ModelRunner.forward](python/sglang/srt/model_executor/model_runner.py#L1645) | 模型前向总入口 | D · 直接注释：新增中文说明 L1647,1653,1655,1667,1670,1671,1673,1676,1677,1678,1680,1681,1720 |
| ✅ 已读 | [ModelRunner.\_forward\_raw](python/sglang/srt/model_executor/model_runner.py#L1820) | 选择图执行或 eager 路径 | D · 直接注释：新增中文说明 L1822,1828,1841,1858,1914 |
| ✅ 已读（推定） | [EagerRunner.execute](python/sglang/srt/model_executor/runner/eager_runner.py#L216) | 按 ForwardMode 分派执行 | C · 链路补全：chain=eager_extend_model;anchors=raw_forward->eager_extend；chain=eager_decode_model;anchors=raw_forward->eager_decode；chain=eager_extend_linear_model;anchors=raw_forward->eager_extend；chain=eager_decode_linear_model;anchors=raw_forward->eager_decode |
| ✅ 已读 | [EagerRunner.\_execute\_extend](python/sglang/srt/model_executor/runner/eager_runner.py#L290) | Prefill 的模型执行入口 | D · 直接注释：新增中文说明 L292,296,389,392,393 |
| ✅ 已读 | [EagerRunner.\_execute\_decode](python/sglang/srt/model_executor/runner/eager_runner.py#L249) | Decode 的模型执行入口 | D · 直接注释：新增中文说明 L255,256,257,258,259,260,261,280,281,282,284 |

### 基础 decode CUDA Graph

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [DecodeCudaGraphRunner.can\_run\_graph](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L676) | 判断批次是否满足图重放条件 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.capture](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1027) | 捕获固定布局的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.execute](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1449) | 重放已捕获的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Qwen 3.5 dense 文本模型的前向主干

已读 **5/9**，未读 **4**。

前向继承父类，但执行的是构造时注入的 Qwen 3.5 主干。language_model_only=True 直接调用 self.model；否则经 general_mm_embed_routine 的文本路径。普通 Attention 与 GDN 是不同层类型；完整调用关系见 LEARN.md 第 3.1 节。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Qwen3VLForConditionalGeneration.forward](python/sglang/srt/models/qwen3_vl.py#L1447) | Qwen 3.5 继承的入口：执行文本主干并产出 logits | D · 直接注释：新增中文说明 L1449,1451,1456,1472,1473,1475,1476,1477,1478,1511,1514,1515,1517,1521,1525 |
| ✅ 已读（推定） | [general\_mm\_embed\_routine](python/sglang/srt/managers/mm_utils.py#L608) | 非 language_model_only 配置下，纯文本也经此完成 embedding 并调用主干 | C · 链路补全：chain=qwen_text_embedding;anchors=causal_lm->qwen_model |
| ✅ 已读 | [Qwen3\_5ForCausalLM.forward](python/sglang/srt/models/qwen3_5.py#L1534) | 嵌入、混合类型解码层逐层执行与最终归一化 | D · 直接注释：新增中文说明 L1543,1545,1546,1563,1568 |
| ✅ 已读 | [Qwen3\_5AttentionDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L1285) | 普通 Attention 层的注意力、MLP 与残差连接 | D · 直接注释：新增中文说明 L1288,1290,1295,1297,1308,1315,1332,1340 |
| ✅ 已读 | [Qwen3\_5AttentionDecoderLayer.self\_attention](python/sglang/srt/models/qwen3_5.py#L1237) | QKV 与位置编码准备、RadixAttention 和输出门控 | D · 直接注释：新增中文说明 L1240,1244,1261 |
| ⬜ 未读 | [Qwen3\_5LinearDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L875) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5GatedDeltaNet.forward](python/sglang/srt/models/qwen3_5.py#L708) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixLinearAttention.forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen2MoeMLP.forward](python/sglang/srt/models/qwen2_moe.py#L249) | Qwen 3.5 dense 复用的门控前馈网络 | U · 待覆盖：无直接证据，且未满足补全规则 |

### QKV 线性投影的前向计算

已读 **0/2**，未读 **2**。

self_attention 经 QKV 准备方法调用 qkv_proj 对象，实际执行继承的 ColumnParallelLinear.forward，再由 UnquantizedLinearMethod.apply 计算。投影与 RadixAttention 是 self_attention 内的不同调用分支；QKVParallelLinear 没有独立的 forward 定义。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [ColumnParallelLinear.forward](python/sglang/srt/layers/linear.py#L492) | QKVParallelLinear 继承的前向入口，调用 quant_method.apply 执行投影 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [UnquantizedLinearMethod.apply](python/sglang/srt/layers/quantization/unquant.py#L255) | 未量化线性计算的普通分支，以 F.linear 执行输入与权重的投影 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Attention 与 KV 读写

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [RadixAttention.forward](python/sglang/srt/layers/radix_attention.py#L161) | 模型层接入 SGLang 注意力后端 | D · 直接注释：新增中文说明 L167,175,182,292 |
| ✅ 已读（推定） | [FlashAttentionBackend.init\_forward\_metadata](python/sglang/srt/layers/attention/flashattention_backend.py#L683) | 准备批次和 KV 索引元数据 | B · 分支补全：branch=attention_backend;direct=2;covered_before_branch=4/5 |
| ✅ 已读（推定） | [FlashAttentionBackend.forward\_extend](python/sglang/srt/layers/attention/flashattention_backend.py#L1228) | Prefill 注意力与 KV 写入 | C · 链路补全：chain=eager_extend_model;anchors=radix_attention->store_kv |
| ✅ 已读（推定） | [FlashAttentionBackend.forward\_decode](python/sglang/srt/layers/attention/flashattention_backend.py#L1804) | 读取历史 KV 并计算 decode 注意力 | C · 链路补全：chain=eager_decode_model;anchors=radix_attention->store_kv |
| ✅ 已读 | [MHATokenToKVPool.set\_kv\_buffer](python/sglang/srt/mem_cache/memory_pool.py#L2431) | 将新 token 的 K/V 写入槽位 | D · 直接注释：新增中文说明 L2434,2435,2436,2443 |


## 8. KV 分配与前缀缓存

### 请求池和 KV 池的建立

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.init\_memory\_pools](python/sglang/srt/managers/scheduler.py#L1028) | 建立调度器持有的缓存池 | D · 直接注释：新增中文说明 L1030,1038 |
| ✅ 已读 | [Scheduler.init\_target\_memory\_pool](python/sglang/srt/managers/scheduler.py#L1004) | 连接模型执行器的目标缓存池 | D · 直接注释：新增中文说明 L1007,1011,1025 |
| ✅ 已读 | [TpModelWorker.alloc\_memory\_pool](python/sglang/srt/managers/tp_worker.py#L409) | 向执行器请求 KV 池 | D · 直接注释：新增中文说明 L422,423,426 |
| ✅ 已读 | [ModelRunner.alloc\_memory\_pool](python/sglang/srt/model_executor/model_runner.py#L896) | 按容量创建并发布缓存资源 | D · 直接注释：新增中文说明 L899,905,909,911,913,915 |
| ✅ 已读 | [ReqToTokenPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L274) | 建立请求槽位到 token 槽位的映射表 | D · 直接注释：新增中文说明 L276,277,282,283,292,297,299,305 |
| ✅ 已读 | [MHATokenToKVPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L1839) | 建立各层 K/V 的实际存储 | D · 直接注释：新增中文说明 L1863,1864,1865,1866,1867,1868,1869,1871,1872,1875,1876 |

### 请求与 KV 槽位分配

已读 **2/8**，未读 **6**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ReqToTokenPool.alloc](python/sglang/srt/mem_cache/memory_pool.py#L317) | 为活跃请求分配映射表行 | D · 直接注释：新增中文说明 L319,341 |
| ⬜ 未读 | [ReqToTokenPool.free](python/sglang/srt/mem_cache/memory_pool.py#L370) | 归还请求映射表行 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [alloc\_for\_extend](python/sglang/srt/mem_cache/allocation.py#L282) | 为新增 prefill token 分配 KV | D · 直接注释：新增中文说明 L292,293,294,295,296,297,298,301,302,304,307,310,332,342,355,359,377,391,412 |
| ⬜ 未读 | [alloc\_for\_decode](python/sglang/srt/mem_cache/allocation.py#L551) | 为下一轮 decode token 分配 KV | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [assign\_req\_to\_token\_pool](python/sglang/srt/mem_cache/allocation.py#L617) | 将逻辑 token 位置映射到 KV 槽位 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.alloc\_extend](python/sglang/srt/mem_cache/allocator/paged.py#L172) | 按页接纳 prefill 的 KV 增量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.alloc\_decode](python/sglang/srt/mem_cache/allocator/paged.py#L222) | 处理 decode 跨页分配 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.free](python/sglang/srt/mem_cache/allocator/paged.py#L261) | 归还 KV 页面 | U · 待覆盖：无直接证据，且未满足补全规则 |

### 前缀匹配与引用保护

已读 **1/5**，未读 **4**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [RadixCache.match\_prefix](python/sglang/srt/mem_cache/radix_cache.py#L391) | 查找可复用前缀 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_match\_prefix\_helper](python/sglang/srt/mem_cache/radix_cache.py#L702) | 沿树匹配 token 序列 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_split\_node](python/sglang/srt/mem_cache/radix_cache.py#L728) | 部分命中时拆分树节点 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [RadixCache.inc\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L637) | 保护活跃请求使用的缓存 | D · 直接注释：新增中文说明 L638,642,643,645,647,648,649,650,651,652,655,656,657 |
| ⬜ 未读 | [RadixCache.dec\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L661) | 释放缓存引用保护 | U · 待覆盖：无直接证据，且未满足补全规则 |

### 缓存写回与淘汰

已读 **0/5**，未读 **5**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [RadixCache.insert](python/sglang/srt/mem_cache/radix_cache.py#L451) | 插入可复用 token 前缀 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_insert\_helper](python/sglang/srt/mem_cache/radix_cache.py#L761) | 维护前缀树节点与 KV 所有权 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.cache\_unfinished\_req](python/sglang/srt/mem_cache/radix_cache.py#L530) | 缓存继续运行请求的前缀 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.cache\_finished\_req](python/sglang/srt/mem_cache/radix_cache.py#L473) | 完成请求的缓存写回与释放 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.evict](python/sglang/srt/mem_cache/radix_cache.py#L607) | 淘汰未保护前缀以腾出 KV | U · 待覆盖：无直接证据，且未满足补全规则 |


## 9. 采样、结果处理与回包

### 从隐藏状态到下一个 token

已读 **4/7**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SamplingParams.normalize](python/sglang/srt/sampling/sampling_params.py#L218) | 将生成参数规范化为采样语义 | D · 直接注释：新增中文说明 L253 |
| ✅ 已读 | [LogitsProcessor.forward](python/sglang/srt/layers/logits_processor.py#L343) | 选择需要输出的 token 并形成 logits | D · 直接注释：新增中文说明 L346,353,354,355,356,360 |
| ⬜ 未读 | [LogitsProcessor.\_get\_logits](python/sglang/srt/layers/logits_processor.py#L668) | 隐藏状态经过 LM head 得到词表分数 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [LogitsProcessor.\_compute\_lm\_head](python/sglang/srt/layers/logits_processor.py#L722) | 执行 LM head 词表投影，普通权重路径计算 hidden_states × weight.T | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ModelRunner.\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L1927) | 惩罚项、掩码与 logit 偏置 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读（推定） | [ModelRunner.sample](python/sglang/srt/model_executor/model_runner.py#L1955) | 调用 sampler 并返回 token ID | C · 链路补全：chain=worker_sample;anchors=worker_forward->sample |
| ✅ 已读 | [Sampler.forward](python/sglang/srt/layers/sampler.py#L98) | 贪心与概率采样的核心分派 | D · 直接注释：新增中文说明 L108 |

### 前向结果进入请求状态

已读 **3/5**，未读 **2**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.process\_batch\_result](python/sglang/srt/managers/scheduler.py#L4424) | 按批次类型处理前向结果 | D · 直接注释：新增中文说明 L4429 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_prefill](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L240) | 处理 prefill 输出和请求状态 | D · 直接注释：新增中文说明 L245 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_decode](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L870) | 接收新 token 并推进 decode 状态 | D · 直接注释：新增中文说明 L875 |
| ⬜ 未读 | [Req.update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1674) | 判断长度、终止 token 和停止字符串 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [release\_kv\_cache](python/sglang/srt/mem_cache/common.py#L202) | 结束请求时转交缓存并释放资源 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Scheduler 输出 token 消息

已读 **2/2**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SchedulerOutputStreamer.stream\_output](python/sglang/srt/managers/scheduler_components/output_streamer.py#L118) | 组织生成结果并选择输出类型 | D · 直接注释：新增中文说明 L125 |
| ✅ 已读（推定） | [SchedulerOutputStreamer.\_stream\_output\_generation](python/sglang/srt/managers/scheduler_components/output_streamer.py#L146) | 按流式间隔构造 token 输出批次 | C · 链路补全：chain=token_message;anchors=stream_output->detokenizer_loop |

### Token 消息增量解码为文本

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [DetokenizerManager.event\_loop](python/sglang/srt/managers/detokenizer_manager.py#L207) | 接收 token 输出并分发解码 | D · 直接注释：新增中文说明 L209,210,211,215,220,221,225 |
| ✅ 已读 | [DetokenizerManager.handle\_batch\_token\_id\_out](python/sglang/srt/managers/detokenizer_manager.py#L506) | 解码并发送文本输出消息 | D · 直接注释：新增中文说明 L517,521,522,528 |
| ✅ 已读 | [DetokenizerManager.\_decode\_batch\_token\_id\_output](python/sglang/srt/managers/detokenizer_manager.py#L360) | 维护请求的增量解码状态 | D · 直接注释：新增中文说明 L361,366,370,373,374,380,383,397,399,434 |
| ✅ 已读 | [DetokenizerManager.\_grouped\_batch\_decode](python/sglang/srt/managers/detokenizer_manager.py#L282) | 将批量 token 增量转为字符串 | D · 直接注释：新增中文说明 L319,325 |
| ✅ 已读（推定） | [DetokenizerManager.trim\_matched\_stop](python/sglang/srt/managers/detokenizer_manager.py#L230) | 按停止规则裁剪返回文本 | B · 分支补全：branch=detokenize;direct=4;covered_before_branch=4/5 |

### 按 rid 回送流式与完整响应

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.handle\_loop](python/sglang/srt/managers/tokenizer_manager.py#L2341) | 接收返回消息并按类型分发 | D · 直接注释：新增中文说明 L2344,2348,2349,2355 |
| ✅ 已读 | [TokenizerManager.\_handle\_batch\_output](python/sglang/srt/managers/tokenizer_manager.py#L2362) | 更新 ReqState 并唤醒等待者 | D · 直接注释：新增中文说明 L2364,2370,2378,2380,2392,2505,2517,2521,2522,2652,2660,2666,2679 |
| ✅ 已读 | [TokenizerManager.\_wait\_one\_response](python/sglang/srt/managers/tokenizer_manager.py#L1841) | 等待并产出请求结果 | D · 直接注释：新增中文说明 L1847,1854,1855,1856,1861,1878,1879,1880,1887,1894,1895,1909,1919,1938,1947,1952,1961 |
| ✅ 已读 | [OpenAIServingChat.\_handle\_non\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1826) | 收集完整结果并构造 chat 响应 | D · 直接注释：新增中文说明 L1834,1835,1836,1837,1838,1841,1842,1846 |
| ✅ 已读（推定） | [OpenAIServingChat.\_handle\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1532) | 建立 chat 流式响应 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |
| ✅ 已读（推定） | [OpenAIServingChat.\_generate\_chat\_stream](python/sglang/srt/entrypoints/openai/serving_chat.py#L1560) | 将增量结果转换为 SSE 消息 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |

## 统计范围与来源

SGLang 常规文本生成核心链路：启动、请求、调度、KV、前向、采样、回包；包含 normal/overlap、chunked prefill、基础 decode CUDA Graph，以及 Qwen3_5ForConditionalGeneration 的初始化、权重加载和 dense 文本推理。补入 QKV 投影及父类的核心构造与未量化前向方法。

模型固定为 Qwen3_5ForConditionalGeneration（dense）：构造函数在 qwen3_5.py，forward 继承 qwen3_vl.py 的 Qwen3VLForConditionalGeneration；self.model 是 qwen3_5.py 的 Qwen3_5ForCausalLM。qwen3_5_text.py 中的同名类是另一个文本模型入口，不计入这条路线。初始化取 DefaultModelLoader 的常规加载路径；普通 Attention 后端取 FlashAttention，线性注意力跟到 GatedDeltaNet 与 RadixLinearAttention 接口；前缀缓存取基础 RadixCache，KV 槽位取 paged allocator。同类实现不重复扩展分母。QKV 线性投影补读 QKVParallelLinear、ColumnParallelLinear、LinearBase 的构造，以及 UnquantizedLinearMethod 的参数创建和普通 F.linear 前向分支；共用方法只计一次。

- 范围版本：v4。
- 统计对象：当前工作区，包含未提交修改。
- HEAD：`81827dda8cfb46762579b96ee04608661927cdb9`。
- 固定对照基线：`6388b6cfb1d93c253714a408f1d66a093302acd7`。
- 源码指纹：`3d4616def05bfd13c546918e894ba36da0182c68be7b631a36c2edaaf46c07f4`。
- 范围指纹：`7bf85107846c00576be9fdd48ea7bc643ede30debb9e867be9899796be26ef90`。
