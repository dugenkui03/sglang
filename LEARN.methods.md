# SGLang 核心方法阅读清单

当前模型固定为 `Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`；初始化、权重加载和文本推理的对象关系见 [LEARN.md 第 3.1 节](LEARN.md#31-固定模型qwen3_5forconditionalgeneration)。本页是全核心总览；当前默认的模型准备与非流式 Overlap 路线见 [LEARN.overlap.md](LEARN.overlap.md)，两者独立计分。

**已读 128/160（80.0%），未读 32 个。**

直接注释 112 个；链路补全 9 个；分支补全 7 个。方法按核心阶段和概念分支排列，可点击方法名跳转源码。

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
| 1 启动与进程 | 29/31 | 2 | 93.5% |
| 2 配置发布 | 2/4 | 2 | 50.0% |
| 3 API 与请求转换 | 6/6 | 0 | 100.0% |
| 4 分词与请求提交 | 10/10 | 0 | 100.0% |
| 5 调度与组批 | 23/29 | 6 | 79.3% |
| 6 执行桥接与 overlap 数据交接 | 7/7 | 0 | 100.0% |
| 7 模型前向与 Attention | 21/24 | 3 | 87.5% |
| 8 KV 分配与前缀缓存 | 10/24 | 14 | 41.7% |
| 9 采样、结果处理与回包 | 20/25 | 5 | 80.0% |

## 1. 启动与进程

### 启动三类进程及 HTTP 服务

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [launch\_server](python/sglang/srt/entrypoints/http_server.py#L2934) | 服务启动总入口 | D · 直接注释：新增中文说明 L2935,2937,2938,2940,2941,2943,2945,2947,2956,2967,2968,2976,2996 |
| ✅ 已读 | [Engine.\_launch\_subprocesses](python/sglang/srt/entrypoints/engine.py#L1110) | 编排进程启动与就绪 | D · 直接注释：新增中文说明 L1126,1127,1128,1129,1130,1131,1142,1151,1161,1162,1170,1185,1187,1212,1213,1222,1225,1233,1234,1240,1326,1332,1339,1352,1358,1368,1375,1376,1377,1381 |
| ✅ 已读 | [Engine.\_launch\_scheduler\_processes](python/sglang/srt/entrypoints/engine.py#L900) | 按 rank 拉起 GPU worker | D · 直接注释：新增中文说明 L932,933,935,936,937,945,947,951,962,1019 |
| ✅ 已读 | [Engine.\_launch\_detokenizer\_subprocesses](python/sglang/srt/entrypoints/engine.py#L1023) | 拉起解码回包进程 | D · 直接注释：新增中文说明 L1043,1045 |
| ✅ 已读 | [init\_tokenizer\_manager](python/sglang/srt/entrypoints/engine.py#L185) | 建立主进程请求入口 | D · 直接注释：新增中文说明 L188,190,191,192,193,197,199,200,202,203,204 |
| ✅ 已读 | [run\_scheduler\_process](python/sglang/srt/managers/scheduler.py#L6200) | Scheduler 子进程入口 | D · 直接注释：新增中文说明 L6213,6220,6232,6257,6263 |
| ✅ 已读 | [run\_detokenizer\_process](python/sglang/srt/managers/detokenizer_manager.py#L593) | Detokenizer 子进程入口 | D · 直接注释：新增中文说明 L609 |
| ✅ 已读 | [\_setup\_and\_run\_http\_server](python/sglang/srt/entrypoints/http_server.py#L2654) | 将 Engine 组件接入 HTTP 服务 | D · 直接注释：新增中文说明 L2670,2671 |

### GPU 执行组件初始化与模型加载

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.\_\_init\_\_](python/sglang/srt/managers/scheduler.py#L460) | 调度器资源与组件初始化 | D · 直接注释：新增中文说明 L563,564,571,572,598,599,698 |
| ✅ 已读 | [Scheduler.init\_model\_worker](python/sglang/srt/managers/scheduler.py#L1116) | 编排权重、缓存、后端和图初始化 | D · 直接注释：新增中文说明 L1117,1118,1119,1120,1121,1124,1132,1137,1205,1206,1238 |
| ✅ 已读 | [Scheduler.init\_tp\_model\_worker](python/sglang/srt/managers/scheduler.py#L993) | 建立调度到 GPU 的执行桥 | D · 直接注释：新增中文说明 L994,995,996,997,1007,1014 |
| ✅ 已读 | [TpModelWorker.\_\_init\_\_](python/sglang/srt/managers/tp_worker.py#L284) | 建立模型配置与 ModelRunner | D · 直接注释：新增中文说明 L320,321 |
| ✅ 已读 | [ModelRunner.\_\_init\_\_](python/sglang/srt/model_executor/model_runner.py#L328) | GPU 执行状态与设备配置 | D · 直接注释：新增中文说明 L342,445,482,490,496,497,499,501,502,504 |
| ✅ 已读 | [ModelRunner.initialize](python/sglang/srt/model_executor/model_runner.py#L692) | 模型执行环境初始化顺序 | D · 直接注释：新增中文说明 L693,702,704,716,717 |
| ✅ 已读 | [ModelRunner.load\_model](python/sglang/srt/model_executor/model_runner.py#L1245) | 将模型权重与结构加载到执行器 | D · 直接注释：新增中文说明 L1246,1249,1253,1259,1260,1262,1264,1265,1267,1272,1311,1313,1361,1364 |

### Qwen 3.5 模型构造入口与权重加载

已读 **3/5**，未读 **2**。

代表路径为常规 DefaultModelLoader；load_model 内先调用 _initialize_model 完成整个对象树构造，再调用 load_weights_and_postprocess。构造与加载是两个阶段，不把构造方法和 load_weights 串成父子调用。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [load\_model\_with\_memory\_saver](python/sglang/srt/model_executor/model_runner_components/load_model_utils.py#L256) | 在权重内存区域中选择 loader 并加载模型 | D · 直接注释：新增中文说明 L269,272,278,279,281,283,284,286,343,363 |
| ✅ 已读 | [DefaultModelLoader.load\_model](python/sglang/srt/model_loader/loader.py#L1009) | 先创建模型结构，再加载权重并进入 eval 模式 | D · 直接注释：新增中文说明 L1016,1029,1031,1035,1037,1039,1041,1042,1044,1046,1047 |
| ✅ 已读 | [\_initialize\_model](python/sglang/srt/model_loader/loader.py#L280) | 按模型架构解析具体类并调用构造函数 | D · 直接注释：新增中文说明 L286,294,300,301,303,305,306,308,311,312,334 |
| ⬜ 未读 | [DefaultModelLoader.load\_weights\_and\_postprocess](python/sglang/srt/model_loader/loader.py#L1061) | 调用模型的 load_weights 并执行加载后处理 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5ForConditionalGeneration.load\_weights](python/sglang/srt/models/qwen3_5.py#L2477) | 映射 checkpoint 名称，将 QKV、MLP 和 GDN 权重装入参数 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Qwen3_5ForConditionalGeneration 的对象初始化

已读 **7/7**，未读 **0**。

入口把 language_model_cls=Qwen3_5ForCausalLM 传给父类；父类用 config.text_config 创建 self.model，并持有 lm_head 与 logits_processor。两类层按 layers_block_type 分支创建，dense MLP 被两类层复用。这里只认识视觉模块的创建条件，不展开视觉计算。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Qwen3\_5ForConditionalGeneration.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L2409) | 注入 Qwen3_5ForCausalLM，调用父类构造并设置位置编码信息 | D · 直接注释：新增中文说明 L2414 |
| ✅ 已读 | [Qwen3VLForConditionalGeneration.\_\_init\_\_](python/sglang/srt/models/qwen3_vl.py#L1315) | 创建文本主干、LM head、LogitsProcessor 与可选视觉模块 | D · 直接注释：新增中文说明 L1320,1358,1372,1373,1391 |
| ✅ 已读 | [Qwen3\_5ForCausalLM.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1790) | 创建 embedding、按配置选择两类解码层并创建最终归一化 | D · 直接注释：新增中文说明 L1804,1831,1874,1877,1878 |
| ✅ 已读 | [Qwen3\_5AttentionDecoderLayer.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1223) | 创建 QKV 投影、RoPE、RadixAttention、dense MLP 与归一化 | D · 直接注释：新增中文说明 L1226,1235,1249,1250,1251,1279,1291,1295,1296,1300 |
| ✅ 已读（推定） | [Qwen3\_5LinearDecoderLayer.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L1092) | 创建 GatedDeltaNet、dense MLP、归一化与层间通信组件 | C · 链路补全：chain=qwen_construct_linear;anchors=qwen_model_init->qwen_gdn_init |
| ✅ 已读 | [Qwen3\_5GatedDeltaNet.\_\_init\_\_](python/sglang/srt/models/qwen3_5.py#L333) | 创建 GDN 输入投影、卷积参数、RadixLinearAttention、门控归一化与输出投影 | D · 直接注释：新增中文说明 L467 |
| ✅ 已读（推定） | [Qwen2MoeMLP.\_\_init\_\_](python/sglang/srt/models/qwen2_moe.py#L172) | 创建 dense gate/up、SiLU 门控与 down 投影 | B · 分支补全：branch=model_construction;direct=5;covered_before_branch=6/7 |

### QKV 投影与线性层权重创建

已读 **4/4**，未读 **0**。

QKVParallelLinear 继承 ColumnParallelLinear 和 LinearBase。ColumnParallelLinear.__init__ 先调用父类构造选择计算实现，返回后再调用 create_weights；两者是独立调用分支。这里只读未量化代表路径，共用方法只计一次。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [QKVParallelLinear.\_\_init\_\_](python/sglang/srt/layers/linear.py#L1113) | 确定 QKV 合并投影的维度、头划分与权重布局 | D · 直接注释：新增中文说明 L1122 |
| ✅ 已读（推定） | [ColumnParallelLinear.\_\_init\_\_](python/sglang/srt/layers/linear.py#L418) | 配置列并行线性层，创建权重并绑定加载回调 | C · 链路补全：chain=qwen_construct_qkv_base;anchors=qkv_projection_init->linear_base_init；chain=qwen_construct_qkv_weights;anchors=qkv_projection_init->unquant_linear_weights |
| ✅ 已读 | [LinearBase.\_\_init\_\_](python/sglang/srt/layers/linear.py#L249) | 保存线性层配置并选择 quant_method 计算实现 | D · 直接注释：新增中文说明 L268 |
| ✅ 已读 | [UnquantizedLinearMethod.create\_weights](python/sglang/srt/layers/quantization/unquant.py#L474) | 创建并注册未量化线性层的 weight 参数及加载属性 | D · 直接注释：新增中文说明 L476,493 |


## 2. 配置发布

### 从启动参数到本进程配置

已读 **2/4**，未读 **2**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ServerArgs.resolve\_once](python/sglang/srt/server_args.py#L243) | 一次性解析启动配置 | D · 直接注释：新增中文说明 L249,252 |
| ✅ 已读 | [publish](python/sglang/srt/runtime_context.py#L1482) | 向本进程发布运行配置 | D · 直接注释：新增中文说明 L1508 |
| ⬜ 未读 | [RuntimeContext.set\_server\_args](python/sglang/srt/runtime_context.py#L1024) | 构造可读取的配置分组 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PortArgs.init\_new](python/sglang/srt/server_args.py#L745) | 分配组件间通信地址 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 3. API 与请求转换

### 生成 API 到内部请求

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [generate\_request](python/sglang/srt/entrypoints/http_server.py#L931) | 原生生成入口与流式选择 | D · 直接注释：新增中文说明 L933 |
| ✅ 已读 | [openai\_v1\_chat\_completions](python/sglang/srt/entrypoints/http_server.py#L1826) | OpenAI chat 入口 | D · 直接注释：新增中文说明 L1823,1824,1827,1828,1832,1833,1836 |
| ✅ 已读 | [OpenAIServingBase.handle\_request](python/sglang/srt/entrypoints/openai/serving_base.py#L72) | 校验、转换和响应路径选择 | D · 直接注释：新增中文说明 L78,94,95,96,97,98,100,103,108,118 |
| ✅ 已读 | [OpenAIServingChat.\_convert\_to\_internal\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1199) | Chat 请求转换为 GenerateReqInput | D · 直接注释：新增中文说明 L1205,1207,1208,1210,1213 |
| ✅ 已读（推定） | [OpenAIServingChat.\_process\_messages](python/sglang/srt/entrypoints/openai/serving_chat.py#L1352) | 会话消息转换为模型输入 | B · 分支补全：branch=api;direct=4;covered_before_branch=4/6 |
| ✅ 已读（推定） | [OpenAIServingChat.\_apply\_jinja\_template](python/sglang/srt/entrypoints/openai/serving_chat.py#L1521) | 应用模型会话模板 | B · 分支补全：branch=api;direct=4;covered_before_branch=4/6 |


## 4. 分词与请求提交

### 主进程请求处理组件

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.\_\_init\_\_](python/sglang/srt/managers/tokenizer_manager.py#L507) | 建立请求处理状态与组件 | D · 直接注释：新增中文说明 L515,533,535,539,540,544,545,549,550,554,558,562,566,570,574,579 |
| ✅ 已读 | [TokenizerManager.init\_tokenizer\_and\_processor](python/sglang/srt/managers/tokenizer_manager.py#L601) | 初始化模型分词器 | D · 直接注释：新增中文说明 L603,610,629,633,634,640,644,646,647,648,649 |
| ✅ 已读 | [TokenizerManager.init\_ipc\_channels](python/sglang/srt/managers/tokenizer_manager.py#L684) | 建立请求与回包管道 | D · 直接注释：新增中文说明 L686,690,691,692,693,695,696,697,698,701,702,703,705,710,711,712,713,718 |

### 请求登记、分词与提交

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.generate\_request](python/sglang/srt/managers/tokenizer_manager.py#L942) | 统一请求生命周期入口 | D · 直接注释：新增中文说明 L948,951,952,957,980,987,991,994,998,1000,1001,1002,1007,1011,1013,1015,1019,1032 |
| ✅ 已读 | [TokenizerManager.\_init\_req\_state](python/sglang/srt/managers/tokenizer_manager.py#L3871) | 以 rid 登记请求状态 | D · 直接注释：新增中文说明 L3877,3893,3895,3912,3917,3918 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1165) | 构造单个 tokenized 请求 | D · 直接注释：新增中文说明 L1178,1182,1219,1222,1346,1348,1349,1350,1351,1352,1353,1354 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_texts](python/sglang/srt/managers/tokenizer_manager.py#L1087) | 文本与 token 输入的分词路径 | D · 直接注释：新增中文说明 L1093,1133 |
| ✅ 已读 | [TokenizerManager.\_validate\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1388) | 上下文长度与生成预算检查 | D · 直接注释：新增中文说明 L1393,1394,1395 |
| ✅ 已读 | [TokenizerManager.\_create\_tokenized\_object](python/sglang/srt/managers/tokenizer_manager.py#L1620) | 生成跨进程请求对象 | D · 直接注释：新增中文说明 L1621,1624,1625,1626,1629,1630,1633,1636,1639,1640,1650,1663,1664,1665,1675,1683 |
| ✅ 已读 | [TokenizerManager.\_send\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1867) | 提交请求并关联返回状态 | D · 直接注释：新增中文说明 L1871,1886 |


## 5. 调度与组批

### 接收、分发与入队

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SchedulerRequestReceiver.recv\_requests](python/sglang/srt/managers/scheduler_components/request_receiver.py#L92) | 接收本轮请求并做 rank 同步 | D · 直接注释：新增中文说明 L96,109,110,116,120,122 |
| ✅ 已读 | [SchedulerRequestReceiver.\_pull\_raw\_reqs](python/sglang/srt/managers/scheduler_components/request_receiver.py#L129) | 从主进程管道接收消息 | D · 直接注释：新增中文说明 L130,133,149,159,168 |
| ✅ 已读（推定） | [SchedulerRequestReceiver.\_broadcast\_reqs\_across\_ranks](python/sglang/srt/managers/scheduler_components/request_receiver.py#L185) | 将请求分发到参与执行的 rank | B · 分支补全：branch=scheduler_ingress;direct=6;covered_before_branch=6/7 |
| ✅ 已读 | [Scheduler.process\_input\_requests](python/sglang/srt/managers/scheduler.py#L2210) | 按消息类型分派处理 | D · 直接注释：新增中文说明 L2212,2213,2214,2219,2238 |
| ✅ 已读 | [Scheduler.handle\_generate\_request](python/sglang/srt/managers/scheduler.py#L2883) | 将 tokenized 输入变为调度请求 | D · 直接注释：新增中文说明 L2890,2897,2900,2904,2911,2913,2963,2978,2979,2980,3072,3074,3257 |
| ✅ 已读 | [Req.\_\_init\_\_](python/sglang/srt/managers/schedule_batch.py#L1015) | 建立请求、生成和缓存状态 | D · 直接注释：新增中文说明 L1194,1197,1214,1218,1326 |
| ✅ 已读 | [Scheduler.\_add\_request\_to\_queue](python/sglang/srt/managers/scheduler.py#L3408) | 进入等待队列 | D · 直接注释：新增中文说明 L3409,3422,3426 |

### 普通与重叠调度主循环

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [dispatch\_event\_loop](python/sglang/srt/managers/scheduler.py#L6084) | 选择运行主循环 | D · 直接注释：新增中文说明 L6085 |
| ✅ 已读 | [Scheduler.run\_event\_loop](python/sglang/srt/managers/scheduler.py#L1907) | 按服务模式选择事件循环 | D · 直接注释：新增中文说明 L1910,1911,1921,1946 |
| ✅ 已读 | [Scheduler.event\_loop\_normal](python/sglang/srt/managers/scheduler.py#L1963) | 收请求、组批、前向、结果处理 | D · 直接注释：新增中文说明 L1970,1978,1983,1985 |
| ✅ 已读 | [Scheduler.event\_loop\_overlap](python/sglang/srt/managers/scheduler.py#L2004) | CPU 调度和 GPU 执行交错推进 | D · 直接注释：新增中文说明 L2003,2006,2008,2009,2010,2011,2012,2017,2018,2021,2032,2038,2045,2047,2048,2050,2055,2056,2057,2059,2060,2061,2063,2064,2065,2066,2067,2068,2070,2092,2093,2102 |

### Prefill 选择与预算控制

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.get\_next\_batch\_to\_run](python/sglang/srt/managers/scheduler.py#L3793) | 选择下一轮 prefill/decode 批次 | D · 直接注释：新增中文说明 L3797,3799,3800,3801,3802,3803,3804,3805,3806,3807,3808,3809,3810,3811,3812,3813,3814,3815,3816,3817,3818,3837,3838,3839,3851,3852,3853,3880,3886,3891,3908,3909,3928,3929,3931,3934,3936 |
| ✅ 已读 | [Scheduler.get\_new\_batch\_prefill](python/sglang/srt/managers/scheduler.py#L3995) | 建立 prefill 批次入口 | D · 直接注释：新增中文说明 L4005,4013,4017,4023,4024 |
| ✅ 已读 | [Scheduler.\_get\_new\_batch\_prefill\_raw](python/sglang/srt/managers/scheduler.py#L4027) | 选择可接纳的等待请求 | D · 直接注释：新增中文说明 L4033,4046,4047,4050,4051,4052,4086,4099,4101,4102,4104,4105,4109,4123,4124,4127,4129,4130,4131,4132,4133,4135,4138,4139,4148,4149,4177,4210,4217,4224,4249,4297,4301,4323,4325,4326,4335,4340,4347 |
| ✅ 已读 | [SchedulePolicy.calc\_priority](python/sglang/srt/managers/schedule_policy.py#L215) | 确定等待队列处理顺序 | D · 直接注释：新增中文说明 L221,222,230,234 |
| ✅ 已读 | [PrefillAdder.\_\_init\_\_](python/sglang/srt/managers/schedule_policy.py#L596) | 建立 token 与显存接纳预算 | D · 直接注释：新增中文说明 L605,615,616,620,626,627,650 |
| ✅ 已读 | [PrefillAdder.add\_one\_req](python/sglang/srt/managers/schedule_policy.py#L1327) | 检查预算并接纳请求 | D · 直接注释：新增中文说明 L1330,1344,1345,1347,1358,1486 |
| ✅ 已读 | [PrefillAdder.add\_chunked\_req](python/sglang/srt/managers/schedule_policy.py#L1099) | 继续处理分块 prefill 请求 | D · 直接注释：新增中文说明 L1100,1150 |
| ✅ 已读 | [PrefillAdder.\_update\_prefill\_budget](python/sglang/srt/managers/schedule_policy.py#L858) | 扣减本轮 prefill 预算 | D · 直接注释：新增中文说明 L861,870,906 |

### Prefill 与 decode 批次状态

已读 **4/7**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [Req.init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1619) | 按已缓存前缀准备下轮输入 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [ScheduleBatch.init\_new](python/sglang/srt/managers/schedule_batch.py#L2567) | 从请求集合建立调度批次 | D · 直接注释：新增中文说明 L2570,2572,2580,2582,2585,2586,2590,2592,2593,2594,2597,2598,2599,2601,2602 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_extend](python/sglang/srt/managers/schedule_batch.py#L2749) | 准备 prefill token、长度和缓存位置 | D · 直接注释：新增中文说明 L2750,2752,2761,2763,2765,2767,2769,2771,2773,2776,2827,2830,2848,2980,2983,2992,2993,2996,2999,3060 |
| ✅ 已读 | [Scheduler.update\_running\_batch](python/sglang/srt/managers/scheduler.py#L4448) | 维护继续生成的请求集合 | D · 直接注释：新增中文说明 L4450,4541 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_decode](python/sglang/srt/managers/schedule_batch.py#L3613) | 为每个请求准备下一 token 的执行状态 | D · 直接注释：新增中文说明 L3636,3656,3664 |
| ⬜ 未读 | [ScheduleBatch.filter\_batch](python/sglang/srt/managers/schedule_batch.py#L3712) | 移除结束请求并对齐批次张量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3800) | 合并完成 prefill 的请求与运行批次 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Decode 显存不足与回退

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [ScheduleBatch.check\_decode\_mem](python/sglang/srt/managers/schedule_batch.py#L3334) | 检查下一轮 KV 容量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.retract\_decode](python/sglang/srt/managers/schedule_batch.py#L3353) | 显存不足时撤回部分请求 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Req.reset\_for\_retract](python/sglang/srt/managers/schedule_batch.py#L1950) | 重置被撤回请求以便重新调度 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 6. 执行桥接与 overlap 数据交接

### 调度批次进入 GPU 执行

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.run\_batch](python/sglang/srt/managers/scheduler.py#L4606) | 提交一轮模型计算并接收结果 | D · 直接注释：新增中文说明 L4608,4612,4632,4649,4651,4658,4659,4660,4661,4663,4670,4694,4695,4699,4873,4876,4882,4886,4888,4890,4896,4911 |
| ✅ 已读 | [TpModelWorker.forward\_batch\_generation](python/sglang/srt/managers/tp_worker.py#L612) | 组织 ForwardBatch、前向与采样 | D · 直接注释：新增中文说明 L614,624,639,641,660,663,682,708,711 |
| ✅ 已读 | [ForwardBatch.init\_new](python/sglang/srt/model_executor/forward_batch_info.py#L972) | 调度视图转换为 GPU 批次视图 | D · 直接注释：新增中文说明 L984,1039,1042,1044,1046,1048,1050,1052 |
| ✅ 已读 | [resolve\_forward\_inputs](python/sglang/srt/managers/overlap_utils.py#L87) | 将 overlap 的未来 token 解析为实际输入 | D · 直接注释：新增中文说明 L118,119,120 |

### Overlap 结果与后续输入衔接

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [FutureMap.\_\_init\_\_](python/sglang/srt/managers/overlap_utils.py#L267) | 建立未来输出槽位映射 | D · 直接注释：新增中文说明 L297,298,299,300,301,302,306,307,308,309,310 |
| ✅ 已读（推定） | [FutureMap.publish](python/sglang/srt/managers/overlap_utils.py#L597) | 发布 GPU 输出供后续批次使用 | B · 分支补全：branch=future_map;direct=2;covered_before_branch=2/3 |
| ✅ 已读 | [FutureMap.stash](python/sglang/srt/managers/overlap_utils.py#L622) | 保存可复用的异步执行结果 | D · 直接注释：新增中文说明 L623 |


## 7. 模型前向与 Attention

### 模型执行路径分派

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ModelRunner.forward](python/sglang/srt/model_executor/model_runner.py#L1811) | 模型前向总入口 | D · 直接注释：新增中文说明 L1813,1819,1821,1833,1836,1837,1839,1842,1843,1844,1846,1847,1884 |
| ✅ 已读 | [ModelRunner.\_forward\_raw](python/sglang/srt/model_executor/model_runner.py#L1988) | 选择图执行或 eager 路径 | D · 直接注释：新增中文说明 L1990,1996,2009,2026,2083 |
| ✅ 已读（推定） | [EagerRunner.execute](python/sglang/srt/model_executor/runner/eager_runner.py#L215) | 按 ForwardMode 分派执行 | C · 链路补全：chain=eager_extend_model;anchors=raw_forward->eager_extend；chain=eager_decode_model;anchors=raw_forward->eager_decode；chain=eager_extend_linear_model;anchors=raw_forward->eager_extend；chain=eager_decode_linear_model;anchors=raw_forward->eager_decode |
| ✅ 已读 | [EagerRunner.\_execute\_extend](python/sglang/srt/model_executor/runner/eager_runner.py#L289) | Prefill 的模型执行入口 | D · 直接注释：新增中文说明 L291,295,400,401 |
| ✅ 已读 | [EagerRunner.\_execute\_decode](python/sglang/srt/model_executor/runner/eager_runner.py#L248) | Decode 的模型执行入口 | D · 直接注释：新增中文说明 L254,255,256,257,258,259,260,279,280,281,283 |

### 基础 decode CUDA Graph

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [DecodeCudaGraphRunner.can\_run\_graph](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L702) | 判断批次是否满足图重放条件 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.capture](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1063) | 捕获固定布局的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.execute](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1529) | 重放已捕获的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Qwen 3.5 dense 文本模型的前向主干

已读 **9/9**，未读 **0**。

前向继承父类，但执行的是构造时注入的 Qwen 3.5 主干。language_model_only=True 直接调用 self.model；否则经 general_mm_embed_routine 的文本路径。普通 Attention 与 GDN 是不同层类型；完整调用关系见 LEARN.md 第 3.1 节。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Qwen3VLForConditionalGeneration.forward](python/sglang/srt/models/qwen3_vl.py#L1618) | Qwen 3.5 继承的入口：执行文本主干并产出 logits | D · 直接注释：新增中文说明 L1620,1622,1627,1643,1644,1646,1647,1648,1649,1682,1685,1686,1688,1692,1696 |
| ✅ 已读（推定） | [general\_mm\_embed\_routine](python/sglang/srt/managers/mm_utils.py#L804) | 非 language_model_only 配置下，纯文本也经此完成 embedding 并调用主干 | C · 链路补全：chain=qwen_text_embedding;anchors=causal_lm->qwen_model |
| ✅ 已读 | [Qwen3\_5ForCausalLM.forward](python/sglang/srt/models/qwen3_5.py#L1929) | 嵌入、混合类型解码层逐层执行与最终归一化 | D · 直接注释：新增中文说明 L1948,1966,1971 |
| ✅ 已读 | [Qwen3\_5AttentionDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L1656) | 普通 Attention 层的注意力、MLP 与残差连接 | D · 直接注释：新增中文说明 L1664,1680,1685 |
| ✅ 已读 | [Qwen3\_5AttentionDecoderLayer.self\_attention](python/sglang/srt/models/qwen3_5.py#L1624) | QKV 与位置编码准备、RadixAttention 和输出门控 | D · 直接注释：新增中文说明 L1627,1632,1651,1652 |
| ✅ 已读 | [Qwen3\_5LinearDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L1181) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 | D · 直接注释：新增中文说明 L1202,1208 |
| ✅ 已读（推定） | [Qwen3\_5GatedDeltaNet.forward](python/sglang/srt/models/qwen3_5.py#L955) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 | C · 链路补全：chain=eager_extend_linear_model;anchors=qwen_linear_layer->radix_linear_attention；chain=eager_decode_linear_model;anchors=qwen_linear_layer->radix_linear_attention |
| ✅ 已读 | [RadixLinearAttention.forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 | D · 直接注释：新增中文说明 L142,143 |
| ✅ 已读（推定） | [Qwen2MoeMLP.forward](python/sglang/srt/models/qwen2_moe.py#L242) | Qwen 3.5 dense 复用的门控前馈网络 | B · 分支补全：branch=dense_model;direct=6;covered_before_branch=8/9 |

### QKV 线性投影的前向计算

已读 **2/2**，未读 **0**。

self_attention 经 QKV 准备方法调用 qkv_proj 对象，实际执行继承的 ColumnParallelLinear.forward，再由 UnquantizedLinearMethod.apply 计算。投影与 RadixAttention 是 self_attention 内的不同调用分支；QKVParallelLinear 没有独立的 forward 定义。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ColumnParallelLinear.forward](python/sglang/srt/layers/linear.py#L610) | QKVParallelLinear 继承的前向入口，调用 quant_method.apply 执行投影 | D · 直接注释：新增中文说明 L624,626,632,633,634 |
| ✅ 已读 | [UnquantizedLinearMethod.apply](python/sglang/srt/layers/quantization/unquant.py#L501) | 未量化线性计算的普通分支，以 F.linear 执行输入与权重的投影 | D · 直接注释：新增中文说明 L508,547 |

### Attention 与 KV 读写

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [RadixAttention.forward](python/sglang/srt/layers/radix_attention.py#L163) | 模型层接入 SGLang 注意力后端 | D · 直接注释：新增中文说明 L169,177,184,299 |
| ✅ 已读 | [FlashAttentionBackend.init\_forward\_metadata](python/sglang/srt/layers/attention/flashattention_backend.py#L866) | 准备批次和 KV 索引元数据 | D · 直接注释：新增中文说明 L874,878,880,886,887 |
| ✅ 已读 | [FlashAttentionBackend.forward\_extend](python/sglang/srt/layers/attention/flashattention_backend.py#L1449) | Prefill 注意力与 KV 写入 | D · 直接注释：新增中文说明 L2010 |
| ✅ 已读 | [FlashAttentionBackend.forward\_decode](python/sglang/srt/layers/attention/flashattention_backend.py#L2018) | 读取历史 KV 并计算 decode 注意力 | D · 直接注释：新增中文说明 L2040,2043,2044,2045,2046,2047,2048,2051,2052,2053,2054,2055,2057,2058,2060,2061,2065,2067,2068,2069,2070,2071,2073,2074,2075,2077,2078,2092,2093,2094,2095,2097,2098,2108,2197,2245,2266,2268,2270,2272,2279,2281,2299,2300,2301,2302,2304,2436 |
| ✅ 已读 | [MHATokenToKVPool.set\_kv\_buffer](python/sglang/srt/mem_cache/memory_pool.py#L2850) | 将新 token 的 K/V 写入槽位 | D · 直接注释：新增中文说明 L2853,2854,2855,2856,2857,2858,2859,2862,2864,2870,2890,2891,2894,2900,2939 |


## 8. KV 分配与前缀缓存

### 请求池和 KV 池的建立

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.init\_memory\_pools](python/sglang/srt/managers/scheduler.py#L1084) | 建立调度器持有的缓存池 | D · 直接注释：新增中文说明 L1086,1094 |
| ✅ 已读 | [Scheduler.init\_target\_memory\_pool](python/sglang/srt/managers/scheduler.py#L1060) | 连接模型执行器的目标缓存池 | D · 直接注释：新增中文说明 L1063,1067,1081 |
| ✅ 已读 | [TpModelWorker.alloc\_memory\_pool](python/sglang/srt/managers/tp_worker.py#L396) | 向执行器请求 KV 池 | D · 直接注释：新增中文说明 L409,410,413 |
| ✅ 已读 | [ModelRunner.alloc\_memory\_pool](python/sglang/srt/model_executor/model_runner.py#L946) | 按容量创建并发布缓存资源 | D · 直接注释：新增中文说明 L949,955,959,961,963,965 |
| ✅ 已读 | [ReqToTokenPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L395) | 建立请求槽位到 token 槽位的映射表 | D · 直接注释：新增中文说明 L397,398,403,404,413,418,420,426 |
| ✅ 已读 | [MHATokenToKVPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L2201) | 建立各层 K/V 的实际存储 | D · 直接注释：新增中文说明 L2225,2226,2227,2228,2229,2230,2231,2233,2234,2237,2238 |

### 请求与 KV 槽位分配

已读 **2/8**，未读 **6**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ReqToTokenPool.alloc](python/sglang/srt/mem_cache/memory_pool.py#L438) | 为活跃请求分配映射表行 | D · 直接注释：新增中文说明 L440,455 |
| ⬜ 未读 | [ReqToTokenPool.free](python/sglang/srt/mem_cache/memory_pool.py#L487) | 归还请求映射表行 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [alloc\_for\_extend](python/sglang/srt/mem_cache/allocation.py#L365) | 为新增 prefill token 分配 KV | D · 直接注释：新增中文说明 L375,376,377,378,379,380,381,384,385,387,390,393,412,424,438,442,460,474,493 |
| ⬜ 未读 | [alloc\_for\_decode](python/sglang/srt/mem_cache/allocation.py#L750) | 为下一轮 decode token 分配 KV | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [assign\_req\_to\_token\_pool](python/sglang/srt/mem_cache/allocation.py#L713) | 将逻辑 token 位置映射到 KV 槽位 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.alloc\_extend](python/sglang/srt/mem_cache/allocator/paged.py#L204) | 按页接纳 prefill 的 KV 增量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.alloc\_decode](python/sglang/srt/mem_cache/allocator/paged.py#L279) | 处理 decode 跨页分配 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [PagedTokenToKVPoolAllocator.free](python/sglang/srt/mem_cache/allocator/paged.py#L328) | 归还 KV 页面 | U · 待覆盖：无直接证据，且未满足补全规则 |

### 前缀匹配与引用保护

已读 **1/5**，未读 **4**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [RadixCache.match\_prefix](python/sglang/srt/mem_cache/radix_cache.py#L373) | 查找可复用前缀 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_match\_prefix\_helper](python/sglang/srt/mem_cache/radix_cache.py#L674) | 沿树匹配 token 序列 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_split\_node](python/sglang/srt/mem_cache/radix_cache.py#L700) | 部分命中时拆分树节点 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [RadixCache.inc\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L596) | 保护活跃请求使用的缓存 | D · 直接注释：新增中文说明 L597,601,602,604,606,607,608,609,610,611,614,615,616 |
| ⬜ 未读 | [RadixCache.dec\_lock\_ref](python/sglang/srt/mem_cache/radix_cache.py#L620) | 释放缓存引用保护 | U · 待覆盖：无直接证据，且未满足补全规则 |

### 缓存写回与淘汰

已读 **1/5**，未读 **4**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [RadixCache.insert](python/sglang/srt/mem_cache/radix_cache.py#L437) | 插入可复用 token 前缀 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_insert\_helper](python/sglang/srt/mem_cache/radix_cache.py#L728) | 维护前缀树节点与 KV 所有权 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.checkpoint](python/sglang/srt/mem_cache/radix_cache.py#L519) | 为请求提交已计算 KV 前缀检查点 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixCache.\_insert\_cache](python/sglang/srt/mem_cache/radix_cache.py#L459) | 将页对齐 KV 前缀写入树并释放重复槽位 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [RadixCache.evict](python/sglang/srt/mem_cache/radix_cache.py#L561) | 淘汰未保护前缀以腾出 KV | D · 直接注释：新增中文说明 L562,567,569,571,579 |


## 9. 采样、结果处理与回包

### 从隐藏状态到下一个 token

已读 **4/7**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SamplingParams.normalize](python/sglang/srt/sampling/sampling_params.py#L302) | 将生成参数规范化为采样语义 | D · 直接注释：新增中文说明 L353 |
| ✅ 已读 | [LogitsProcessor.forward](python/sglang/srt/layers/logits_processor.py#L511) | 选择需要输出的 token 并形成 logits | D · 直接注释：新增中文说明 L514,520,522,523,524,525,526 |
| ⬜ 未读 | [LogitsProcessor.\_get\_logits](python/sglang/srt/layers/logits_processor.py#L876) | 隐藏状态经过 LM head 得到词表分数 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [LogitsProcessor.\_compute\_lm\_head](python/sglang/srt/layers/logits_processor.py#L978) | 执行 LM head 词表投影，普通权重路径计算 hidden_states × weight.T | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ModelRunner.\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L2096) | 惩罚项、掩码与 logit 偏置 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读（推定） | [ModelRunner.sample](python/sglang/srt/model_executor/model_runner.py#L2124) | 调用 sampler 并返回 token ID | C · 链路补全：chain=worker_sample;anchors=worker_forward->sample |
| ✅ 已读 | [Sampler.forward](python/sglang/srt/layers/sampler.py#L142) | 贪心与概率采样的核心分派 | D · 直接注释：新增中文说明 L152 |

### 前向结果进入请求状态

已读 **3/5**，未读 **2**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.process\_batch\_result](python/sglang/srt/managers/scheduler.py#L5056) | 按批次类型处理前向结果 | D · 直接注释：新增中文说明 L5061 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_prefill](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L258) | 处理 prefill 输出和请求状态 | D · 直接注释：新增中文说明 L263 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_decode](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L923) | 接收新 token 并推进 decode 状态 | D · 直接注释：新增中文说明 L928 |
| ⬜ 未读 | [Req.update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1908) | 判断长度、终止 token 和停止字符串 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [release\_kv\_cache](python/sglang/srt/mem_cache/common.py#L379) | 结束请求时转交缓存并释放资源 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Scheduler 输出 token 消息

已读 **2/2**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SchedulerOutputStreamer.stream\_output](python/sglang/srt/managers/scheduler_components/output_streamer.py#L119) | 组织生成结果并选择输出类型 | D · 直接注释：新增中文说明 L126 |
| ✅ 已读（推定） | [SchedulerOutputStreamer.\_stream\_output\_generation](python/sglang/srt/managers/scheduler_components/output_streamer.py#L147) | 按流式间隔构造 token 输出批次 | C · 链路补全：chain=token_message;anchors=stream_output->detokenizer_loop |

### Token 消息增量解码为文本

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [DetokenizerManager.event\_loop](python/sglang/srt/managers/detokenizer_manager.py#L203) | 接收 token 输出并分发解码 | D · 直接注释：新增中文说明 L205,206,207,211,216,217,221 |
| ✅ 已读 | [DetokenizerManager.handle\_batch\_token\_id\_out](python/sglang/srt/managers/detokenizer_manager.py#L490) | 解码并发送文本输出消息 | D · 直接注释：新增中文说明 L501,505,506,512 |
| ✅ 已读 | [DetokenizerManager.\_decode\_batch\_token\_id\_output](python/sglang/srt/managers/detokenizer_manager.py#L344) | 维护请求的增量解码状态 | D · 直接注释：新增中文说明 L345,350,354,357,358,364,367,381,383,418 |
| ✅ 已读 | [DetokenizerManager.\_grouped\_batch\_decode](python/sglang/srt/managers/detokenizer_manager.py#L278) | 将批量 token 增量转为字符串 | D · 直接注释：新增中文说明 L309,315 |
| ✅ 已读（推定） | [DetokenizerManager.trim\_matched\_stop](python/sglang/srt/managers/detokenizer_manager.py#L226) | 按停止规则裁剪返回文本 | B · 分支补全：branch=detokenize;direct=4;covered_before_branch=4/5 |

### 按 rid 回送流式与完整响应

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.handle\_loop](python/sglang/srt/managers/tokenizer_manager.py#L2563) | 接收返回消息并按类型分发 | D · 直接注释：新增中文说明 L2566,2570,2571,2577 |
| ✅ 已读 | [TokenizerManager.\_handle\_batch\_output](python/sglang/srt/managers/tokenizer_manager.py#L2584) | 更新 ReqState 并唤醒等待者 | D · 直接注释：新增中文说明 L2586,2592,2600,2602,2614,2726,2738,2742,2743,2884,2890,2896,2909 |
| ✅ 已读 | [TokenizerManager.\_wait\_one\_response](python/sglang/srt/managers/tokenizer_manager.py#L2028) | 等待并产出请求结果 | D · 直接注释：新增中文说明 L2033 |
| ✅ 已读 | [OpenAIServingChat.\_handle\_non\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L2304) | 收集完整结果并构造 chat 响应 | D · 直接注释：新增中文说明 L2312,2313,2314,2315,2316,2319,2320 |
| ✅ 已读（推定） | [OpenAIServingChat.\_handle\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1938) | 建立 chat 流式响应 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |
| ✅ 已读（推定） | [OpenAIServingChat.\_generate\_chat\_stream](python/sglang/srt/entrypoints/openai/serving_chat.py#L1966) | 将增量结果转换为 SSE 消息 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |

## 统计范围与来源

SGLang 常规文本生成核心链路：启动、请求、调度、KV、前向、采样、回包；包含 normal/overlap、chunked prefill、基础 decode CUDA Graph，以及 Qwen3_5ForConditionalGeneration 的初始化、权重加载和 dense 文本推理。补入 QKV 投影及父类的核心构造与未量化前向方法。 main 合并后的接口映射：旧 cache_unfinished_req/cache_finished_req 已移除，缓存写回改为 checkpoint → _insert_cache，结束释放仍由 release_kv_cache 负责；方法总数不变。

模型固定为 Qwen3_5ForConditionalGeneration（dense）：构造函数在 qwen3_5.py，forward 继承 qwen3_vl.py 的 Qwen3VLForConditionalGeneration；self.model 是 qwen3_5.py 的 Qwen3_5ForCausalLM。qwen3_5_text.py 中的同名类是另一个文本模型入口，不计入这条路线。初始化取 DefaultModelLoader 的常规加载路径；普通 Attention 后端取 FlashAttention，线性注意力跟到 GatedDeltaNet 与 RadixLinearAttention 接口；前缀缓存取基础 RadixCache，KV 槽位取 paged allocator。同类实现不重复扩展分母。QKV 线性投影补读 QKVParallelLinear、ColumnParallelLinear、LinearBase 的构造，以及 UnquantizedLinearMethod 的参数创建和普通 F.linear 前向分支；共用方法只计一次。

- 范围版本：v5。
- 统计对象：当前工作区，包含未提交修改。
- HEAD：`41e029e21bbd7c2552f1cee21792ed04e84638f4`。
- 固定对照基线：`6388b6cfb1d93c253714a408f1d66a093302acd7`。
- 源码指纹：`cdde38c6488c0d8a76d959e348d1d508e84cdd4ef0abe80b8e6afbfc8d41a439`。
- 范围指纹：`c5aeae3bf58deca7ee1a4141c3a037fa703e56cda080a2467a3fc3d11b487b60`。
