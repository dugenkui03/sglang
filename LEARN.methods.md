# SGLang 核心方法阅读清单

**已读 93/140（66.4%），未读 47 个。**

直接注释 82 个；链路补全 5 个；分支补全 6 个。方法按核心阶段和概念分支排列，可点击方法名跳转源码。

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
| 1 启动与进程 | 15/15 | 0 | 100.0% |
| 2 配置发布 | 2/4 | 2 | 50.0% |
| 3 API 与请求转换 | 3/6 | 3 | 50.0% |
| 4 分词与请求提交 | 10/10 | 0 | 100.0% |
| 5 调度与组批 | 23/29 | 6 | 79.3% |
| 6 执行桥接与 overlap 数据交接 | 7/7 | 0 | 100.0% |
| 7 模型前向与 Attention | 6/21 | 15 | 28.6% |
| 8 KV 分配与前缀缓存 | 8/24 | 16 | 33.3% |
| 9 采样、结果处理与回包 | 19/24 | 5 | 79.2% |

## 1. 启动与进程

### 启动三类进程及 HTTP 服务

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [launch\_server](python/sglang/srt/entrypoints/http_server.py#L2790) | 服务启动总入口 | D · 直接注释：新增中文说明 L2791,2793,2795,2797,2799,2801,2810,2827 |
| ✅ 已读 | [Engine.\_launch\_subprocesses](python/sglang/srt/entrypoints/engine.py#L1051) | 编排进程启动与就绪 | D · 直接注释：新增中文说明 L1067,1068,1069,1070,1071,1072,1083,1092,1102,1103,1111,1126,1128,1153,1154,1163,1166,1174,1175,1176,1180,1182,1233,1239,1246,1247,1260,1266,1276,1283,1284,1285,1289 |
| ✅ 已读 | [Engine.\_launch\_scheduler\_processes](python/sglang/srt/entrypoints/engine.py#L838) | 按 rank 拉起 GPU worker | D · 直接注释：新增中文说明 L875,876,878,879,880,886,887,888,894,896,900,914,960 |
| ✅ 已读 | [Engine.\_launch\_detokenizer\_subprocesses](python/sglang/srt/entrypoints/engine.py#L964) | 拉起解码回包进程 | D · 直接注释：新增中文说明 L984,986 |
| ✅ 已读 | [init\_tokenizer\_manager](python/sglang/srt/entrypoints/engine.py#L156) | 建立主进程请求入口 | D · 直接注释：新增中文说明 L159,161,162,163,164,168,170,171,173,174,175 |
| ✅ 已读 | [run\_scheduler\_process](python/sglang/srt/managers/scheduler.py#L5557) | Scheduler 子进程入口 | D · 直接注释：新增中文说明 L5608,5625 |
| ✅ 已读 | [run\_detokenizer\_process](python/sglang/srt/managers/detokenizer_manager.py#L610) | Detokenizer 子进程入口 | D · 直接注释：新增中文说明 L625 |
| ✅ 已读 | [\_setup\_and\_run\_http\_server](python/sglang/srt/entrypoints/http_server.py#L2522) | 将 Engine 组件接入 HTTP 服务 | D · 直接注释：新增中文说明 L2533,2534,2535,2536,2542,2544,2557,2605 |

### GPU 执行组件初始化与模型加载

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.\_\_init\_\_](python/sglang/srt/managers/scheduler.py#L413) | 调度器资源与组件初始化 | D · 直接注释：新增中文说明 L542,543,550,551,585,670 |
| ✅ 已读 | [Scheduler.init\_model\_worker](python/sglang/srt/managers/scheduler.py#L1057) | 编排权重、缓存、后端和图初始化 | D · 直接注释：新增中文说明 L1058,1059,1060,1061,1062,1065,1073,1078,1136,1137 |
| ✅ 已读 | [Scheduler.init\_tp\_model\_worker](python/sglang/srt/managers/scheduler.py#L944) | 建立调度到 GPU 的执行桥 | D · 直接注释：新增中文说明 L945,946,947,948,959,965 |
| ✅ 已读 | [TpModelWorker.\_\_init\_\_](python/sglang/srt/managers/tp_worker.py#L318) | 建立模型配置与 ModelRunner | D · 直接注释：新增中文说明 L353,354 |
| ✅ 已读 | [ModelRunner.\_\_init\_\_](python/sglang/srt/model_executor/model_runner.py#L314) | GPU 执行状态与设备配置 | D · 直接注释：新增中文说明 L329,431,465 |
| ✅ 已读 | [ModelRunner.initialize](python/sglang/srt/model_executor/model_runner.py#L649) | 模型执行环境初始化顺序 | D · 直接注释：新增中文说明 L650,659,661,678,679 |
| ✅ 已读 | [ModelRunner.load\_model](python/sglang/srt/model_executor/model_runner.py#L1142) | 将模型权重与结构加载到执行器 | D · 直接注释：新增中文说明 L1143,1146,1173,1191,1239,1240 |


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

已读 **3/6**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [generate\_request](python/sglang/srt/entrypoints/http_server.py#L899) | 原生生成入口与流式选择 | D · 直接注释：新增中文说明 L901,905,942 |
| ✅ 已读 | [openai\_v1\_chat\_completions](python/sglang/srt/entrypoints/http_server.py#L1732) | OpenAI chat 入口 | D · 直接注释：新增中文说明 L1736 |
| ✅ 已读 | [OpenAIServingBase.handle\_request](python/sglang/srt/entrypoints/openai/serving_base.py#L73) | 校验、转换和响应路径选择 | D · 直接注释：新增中文说明 L79,80,91,97,98,99,104,114 |
| ⬜ 未读 | [OpenAIServingChat.\_convert\_to\_internal\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L968) | Chat 请求转换为 GenerateReqInput | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [OpenAIServingChat.\_process\_messages](python/sglang/srt/entrypoints/openai/serving_chat.py#L1102) | 会话消息转换为模型输入 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [OpenAIServingChat.\_apply\_jinja\_template](python/sglang/srt/entrypoints/openai/serving_chat.py#L1208) | 应用模型会话模板 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 4. 分词与请求提交

### 主进程请求处理组件

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.\_\_init\_\_](python/sglang/srt/managers/tokenizer_manager.py#L423) | 建立请求处理状态与组件 | D · 直接注释：新增中文说明 L431,449,451,455,456,460,461,465,466,470,474,478,482,486,490,495 |
| ✅ 已读 | [TokenizerManager.init\_tokenizer\_and\_processor](python/sglang/srt/managers/tokenizer_manager.py#L517) | 初始化模型分词器 | D · 直接注释：新增中文说明 L519,524,543,547,548,554,558,560,561,562,563 |
| ✅ 已读 | [TokenizerManager.init\_ipc\_channels](python/sglang/srt/managers/tokenizer_manager.py#L598) | 建立请求与回包管道 | D · 直接注释：新增中文说明 L600,602,603,604,605,607,608,609,610,613,614,615,617,622,623,624,625,630 |

### 请求登记、分词与提交

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [TokenizerManager.generate\_request](python/sglang/srt/managers/tokenizer_manager.py#L845) | 统一请求生命周期入口 | D · 直接注释：新增中文说明 L850,853,858,865,871,883,886,890,893,897,899,900,901,909,911,912,914,915,916,917,918,922,924,934 |
| ✅ 已读 | [TokenizerManager.\_init\_req\_state](python/sglang/srt/managers/tokenizer_manager.py#L3568) | 以 rid 登记请求状态 | D · 直接注释：新增中文说明 L3574,3590,3592,3609,3614,3615 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1064) | 构造单个 tokenized 请求 | D · 直接注释：新增中文说明 L1075,1077,1102,1105,1108,1233,1235,1237,1238,1239,1240,1242,1243 |
| ✅ 已读 | [TokenizerManager.\_tokenize\_texts](python/sglang/srt/managers/tokenizer_manager.py#L989) | 文本与 token 输入的分词路径 | D · 直接注释：新增中文说明 L995,1032 |
| ✅ 已读 | [TokenizerManager.\_validate\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1277) | 上下文长度与生成预算检查 | D · 直接注释：新增中文说明 L1282,1283,1284 |
| ✅ 已读 | [TokenizerManager.\_create\_tokenized\_object](python/sglang/srt/managers/tokenizer_manager.py#L1456) | 生成跨进程请求对象 | D · 直接注释：新增中文说明 L1457,1458,1459,1460,1461,1462,1463,1466,1469,1470,1480,1493,1494,1495,1505,1513 |
| ✅ 已读 | [TokenizerManager.\_send\_one\_request](python/sglang/srt/managers/tokenizer_manager.py#L1692) | 提交请求并关联返回状态 | D · 直接注释：新增中文说明 L1694,1707 |


## 5. 调度与组批

### 接收、分发与入队

已读 **7/7**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SchedulerRequestReceiver.recv\_requests](python/sglang/srt/managers/scheduler_components/request_receiver.py#L88) | 接收本轮请求并做 rank 同步 | D · 直接注释：新增中文说明 L92,102,103,109,110,111,114,116,118 |
| ✅ 已读 | [SchedulerRequestReceiver.\_pull\_raw\_reqs](python/sglang/srt/managers/scheduler_components/request_receiver.py#L125) | 从主进程管道接收消息 | D · 直接注释：新增中文说明 L127,131,148,158,167,168 |
| ✅ 已读（推定） | [SchedulerRequestReceiver.\_broadcast\_reqs\_across\_ranks](python/sglang/srt/managers/scheduler_components/request_receiver.py#L183) | 将请求分发到参与执行的 rank | B · 分支补全：branch=scheduler_ingress;direct=6;covered_before_branch=6/7 |
| ✅ 已读 | [Scheduler.process\_input\_requests](python/sglang/srt/managers/scheduler.py#L2042) | 按消息类型分派处理 | D · 直接注释：新增中文说明 L2044,2045,2046,2050,2055,2067,2068 |
| ✅ 已读 | [Scheduler.handle\_generate\_request](python/sglang/srt/managers/scheduler.py#L2592) | 将 tokenized 输入变为调度请求 | D · 直接注释：新增中文说明 L2597,2604,2607,2612,2624,2626,2673,2686,2687,2688,2754,2756,2924 |
| ✅ 已读 | [Req.\_\_init\_\_](python/sglang/srt/managers/schedule_batch.py#L839) | 建立请求、生成和缓存状态 | D · 直接注释：新增中文说明 L1025,1029,1043,1047,1142 |
| ✅ 已读 | [Scheduler.\_add\_request\_to\_queue](python/sglang/srt/managers/scheduler.py#L2984) | 进入等待队列 | D · 直接注释：新增中文说明 L2985,2993,2996 |

### 普通与重叠调度主循环

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [dispatch\_event\_loop](python/sglang/srt/managers/scheduler.py#L5446) | 选择运行主循环 | D · 直接注释：新增中文说明 L5450,5454,5457,5460,5461,5462,5463,5466,5467,5468,5469,5472,5480 |
| ✅ 已读 | [Scheduler.run\_event\_loop](python/sglang/srt/managers/scheduler.py#L1785) | 按服务模式选择事件循环 | D · 直接注释：新增中文说明 L1797,1822 |
| ✅ 已读 | [Scheduler.event\_loop\_normal](python/sglang/srt/managers/scheduler.py#L1839) | 收请求、组批、前向、结果处理 | D · 直接注释：新增中文说明 L1846,1854,1860,1862 |
| ✅ 已读 | [Scheduler.event\_loop\_overlap](python/sglang/srt/managers/scheduler.py#L1881) | CPU 调度和 GPU 执行交错推进 | D · 直接注释：新增中文说明 L1880,1883,1885,1886,1887,1888,1889,1893,1894,1900,1909,1915,1916,1919,1925,1928,1929,1930,1932,1933,1935,1936,1937,1938,1939,1940,1942,1964,1973 |

### Prefill 选择与预算控制

已读 **8/8**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.get\_next\_batch\_to\_run](python/sglang/srt/managers/scheduler.py#L3285) | 选择下一轮 prefill/decode 批次 | D · 直接注释：新增中文说明 L3291,3316,3317,3318,3331,3332,3333,3360,3366,3371,3383,3389,3390,3408,3409,3411,3414,3416,3417,3419 |
| ✅ 已读 | [Scheduler.get\_new\_batch\_prefill](python/sglang/srt/managers/scheduler.py#L3470) | 建立 prefill 批次入口 | D · 直接注释：新增中文说明 L3480,3488,3498,3499 |
| ✅ 已读 | [Scheduler.\_get\_new\_batch\_prefill\_raw](python/sglang/srt/managers/scheduler.py#L3502) | 选择可接纳的等待请求 | D · 直接注释：新增中文说明 L3508,3524,3525,3528,3529,3530,3565,3577,3578,3591,3592,3595,3597,3598,3599,3600,3601,3603,3606,3607,3616,3637,3665,3667,3671,3674,3677,3697,3743,3747,3765,3767,3768,3777,3778,3779,3784,3791 |
| ✅ 已读 | [SchedulePolicy.calc\_priority](python/sglang/srt/managers/schedule_policy.py#L242) | 确定等待队列处理顺序 | D · 直接注释：新增中文说明 L245,247,255,259 |
| ✅ 已读 | [PrefillAdder.\_\_init\_\_](python/sglang/srt/managers/schedule_policy.py#L525) | 建立 token 与显存接纳预算 | D · 直接注释：新增中文说明 L534,544,545,549,555,556,569 |
| ✅ 已读 | [PrefillAdder.add\_one\_req](python/sglang/srt/managers/schedule_policy.py#L1243) | 检查预算并接纳请求 | D · 直接注释：新增中文说明 L1247,1268,1269,1271,1272,1273,1274,1276,1277,1280,1289,1292,1296,1331,1334,1335,1388,1390,1391,1420,1421,1429,1436,1438,1440,1441,1442,1444,1445,1447,1452,1484,1503 |
| ✅ 已读（推定） | [PrefillAdder.add\_chunked\_req](python/sglang/srt/managers/schedule_policy.py#L1037) | 继续处理分块 prefill 请求 | B · 分支补全：branch=prefill_admission;direct=7;covered_before_branch=7/8 |
| ✅ 已读 | [PrefillAdder.\_update\_prefill\_budget](python/sglang/srt/managers/schedule_policy.py#L886) | 扣减本轮 prefill 预算 | D · 直接注释：新增中文说明 L889,896,907,908,910,911,912,915,916,934 |

### Prefill 与 decode 批次状态

已读 **4/7**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [Req.init\_next\_round\_input](python/sglang/srt/managers/schedule_batch.py#L1344) | 按已缓存前缀准备下轮输入 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [ScheduleBatch.init\_new](python/sglang/srt/managers/schedule_batch.py#L2273) | 从请求集合建立调度批次 | D · 直接注释：新增中文说明 L2276,2278,2286,2288,2291,2292,2296,2299,2300,2303,2304,2305,2307,2308 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_extend](python/sglang/srt/managers/schedule_batch.py#L2460) | 准备 prefill token、长度和缓存位置 | D · 直接注释：新增中文说明 L2461,2463,2472,2474,2476,2478,2480,2482,2484,2487,2488,2519,2539,2677,2738 |
| ✅ 已读 | [Scheduler.update\_running\_batch](python/sglang/srt/managers/scheduler.py#L3865) | 维护继续生成的请求集合 | D · 直接注释：新增中文说明 L3867,3947 |
| ✅ 已读 | [ScheduleBatch.prepare\_for\_decode](python/sglang/srt/managers/schedule_batch.py#L3218) | 为每个请求准备下一 token 的执行状态 | D · 直接注释：新增中文说明 L3257 |
| ⬜ 未读 | [ScheduleBatch.filter\_batch](python/sglang/srt/managers/schedule_batch.py#L3314) | 移除结束请求并对齐批次张量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.merge\_batch](python/sglang/srt/managers/schedule_batch.py#L3401) | 合并完成 prefill 的请求与运行批次 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Decode 显存不足与回退

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [ScheduleBatch.check\_decode\_mem](python/sglang/srt/managers/schedule_batch.py#L2955) | 检查下一轮 KV 容量 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ScheduleBatch.retract\_decode](python/sglang/srt/managers/schedule_batch.py#L2962) | 显存不足时撤回部分请求 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Req.reset\_for\_retract](python/sglang/srt/managers/schedule_batch.py#L1717) | 重置被撤回请求以便重新调度 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 6. 执行桥接与 overlap 数据交接

### 调度批次进入 GPU 执行

已读 **4/4**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.run\_batch](python/sglang/srt/managers/scheduler.py#L4012) | 提交一轮模型计算并接收结果 | D · 直接注释：新增中文说明 L4014,4018,4029,4036,4046,4047,4049,4056,4057,4058,4059,4061,4068,4092,4093,4097 |
| ✅ 已读 | [TpModelWorker.forward\_batch\_generation](python/sglang/srt/managers/tp_worker.py#L601) | 组织 ForwardBatch、前向与采样 | D · 直接注释：新增中文说明 L603,612,613,614,616,623,625,644,664,689,692 |
| ✅ 已读（推定） | [ForwardBatch.init\_new](python/sglang/srt/model_executor/forward_batch_info.py#L732) | 调度视图转换为 GPU 批次视图 | B · 分支补全：branch=forward_bridge;direct=3;covered_before_branch=3/4 |
| ✅ 已读 | [resolve\_forward\_inputs](python/sglang/srt/managers/overlap_utils.py#L87) | 将 overlap 的未来 token 解析为实际输入 | D · 直接注释：新增中文说明 L96,110 |

### Overlap 结果与后续输入衔接

已读 **3/3**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [FutureMap.\_\_init\_\_](python/sglang/srt/managers/overlap_utils.py#L259) | 建立未来输出槽位映射 | D · 直接注释：新增中文说明 L287,288,289,290,291,292,298,299,300,301,302 |
| ✅ 已读（推定） | [FutureMap.publish](python/sglang/srt/managers/overlap_utils.py#L535) | 发布 GPU 输出供后续批次使用 | B · 分支补全：branch=future_map;direct=2;covered_before_branch=2/3 |
| ✅ 已读 | [FutureMap.stash](python/sglang/srt/managers/overlap_utils.py#L560) | 保存可复用的异步执行结果 | D · 直接注释：新增中文说明 L561 |


## 7. 模型前向与 Attention

### 模型执行路径分派

已读 **5/5**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [ModelRunner.forward](python/sglang/srt/model_executor/model_runner.py#L1597) | 模型前向总入口 | D · 直接注释：新增中文说明 L1606,1645 |
| ✅ 已读 | [ModelRunner.\_forward\_raw](python/sglang/srt/model_executor/model_runner.py#L1745) | 选择图执行或 eager 路径 | D · 直接注释：新增中文说明 L1747,1753,1780,1781,1807,1823,1841 |
| ✅ 已读（推定） | [EagerRunner.execute](python/sglang/srt/model_executor/runner/eager_runner.py#L214) | 按 ForwardMode 分派执行 | C · 链路补全：chain=eager_extend_model;anchors=raw_forward->eager_extend；chain=eager_decode_model;anchors=raw_forward->eager_decode；chain=eager_extend_linear_model;anchors=raw_forward->eager_extend；chain=eager_decode_linear_model;anchors=raw_forward->eager_decode |
| ✅ 已读 | [EagerRunner.\_execute\_extend](python/sglang/srt/model_executor/runner/eager_runner.py#L289) | Prefill 的模型执行入口 | D · 直接注释：新增中文说明 L291,295,300 |
| ✅ 已读 | [EagerRunner.\_execute\_decode](python/sglang/srt/model_executor/runner/eager_runner.py#L248) | Decode 的模型执行入口 | D · 直接注释：新增中文说明 L254,255,256,257,258,259,260,279,280,281,283 |

### 基础 decode CUDA Graph

已读 **0/3**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [DecodeCudaGraphRunner.can\_run\_graph](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L676) | 判断批次是否满足图重放条件 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.capture](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1027) | 捕获固定布局的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [DecodeCudaGraphRunner.execute](python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1449) | 重放已捕获的执行图 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Qwen 3.5 dense 文本模型的前向主干

已读 **1/8**，未读 **7**。

入口为 Qwen3_5ForConditionalGeneration，forward 继承自 Qwen3VLForConditionalGeneration；其内部文本模型是 qwen3_5.py 中的 Qwen3_5ForCausalLM。普通 Attention 层和 GatedDeltaNet 线性注意力层是按配置选择的两条分支；两类层复用 Qwen2MoeMLP 的 dense 前馈实现，类名中的 Moe 不表示本清单纳入了专家路由。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Qwen3VLForConditionalGeneration.forward](python/sglang/srt/models/qwen3_vl.py#L1442) | Qwen 3.5 继承的入口：执行文本主干并产出 logits | D · 直接注释：新增中文说明 L1451,1467,1468,1469,1470,1471,1472,1473,1474,1475,1479,1513,1514,1516,1520 |
| ⬜ 未读 | [Qwen3\_5ForCausalLM.forward](python/sglang/srt/models/qwen3_5.py#L1503) | 嵌入、混合类型解码层逐层执行与最终归一化 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5AttentionDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L1280) | 普通 Attention 层的注意力、MLP 与残差连接 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5AttentionDecoderLayer.self\_attention](python/sglang/srt/models/qwen3_5.py#L1235) | QKV 与位置编码准备、RadixAttention 和输出门控 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5LinearDecoderLayer.forward](python/sglang/srt/models/qwen3_5.py#L875) | 线性注意力层的 GatedDeltaNet、MLP 与残差连接 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen3\_5GatedDeltaNet.forward](python/sglang/srt/models/qwen3_5.py#L708) | GatedDeltaNet 的输入投影、线性注意力调用与输出门控 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [RadixLinearAttention.forward](python/sglang/srt/layers/radix_linear_attention.py#L80) | 将 GatedDeltaNet 输入交给 SGLang 线性注意力后端 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [Qwen2MoeMLP.forward](python/sglang/srt/models/qwen2_moe.py#L249) | Qwen 3.5 dense 复用的门控前馈网络 | U · 待覆盖：无直接证据，且未满足补全规则 |

### Attention 与 KV 读写

已读 **0/5**，未读 **5**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [RadixAttention.forward](python/sglang/srt/layers/radix_attention.py#L157) | 模型层接入 SGLang 注意力后端 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [FlashAttentionBackend.init\_forward\_metadata](python/sglang/srt/layers/attention/flashattention_backend.py#L683) | 准备批次和 KV 索引元数据 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [FlashAttentionBackend.forward\_extend](python/sglang/srt/layers/attention/flashattention_backend.py#L1228) | Prefill 注意力与 KV 写入 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [FlashAttentionBackend.forward\_decode](python/sglang/srt/layers/attention/flashattention_backend.py#L1804) | 读取历史 KV 并计算 decode 注意力 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [MHATokenToKVPool.set\_kv\_buffer](python/sglang/srt/mem_cache/memory_pool.py#L2388) | 将新 token 的 K/V 写入槽位 | U · 待覆盖：无直接证据，且未满足补全规则 |


## 8. KV 分配与前缀缓存

### 请求池和 KV 池的建立

已读 **6/6**，未读 **0**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.init\_memory\_pools](python/sglang/srt/managers/scheduler.py#L1025) | 建立调度器持有的缓存池 | D · 直接注释：新增中文说明 L1027,1035 |
| ✅ 已读 | [Scheduler.init\_target\_memory\_pool](python/sglang/srt/managers/scheduler.py#L1001) | 连接模型执行器的目标缓存池 | D · 直接注释：新增中文说明 L1004,1008,1022 |
| ✅ 已读 | [TpModelWorker.alloc\_memory\_pool](python/sglang/srt/managers/tp_worker.py#L407) | 向执行器请求 KV 池 | D · 直接注释：新增中文说明 L420,421 |
| ✅ 已读 | [ModelRunner.alloc\_memory\_pool](python/sglang/srt/model_executor/model_runner.py#L875) | 按容量创建并发布缓存资源 | D · 直接注释：新增中文说明 L878,884,888,890,892,894 |
| ✅ 已读 | [ReqToTokenPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L264) | 建立请求槽位到 token 槽位的映射表 | D · 直接注释：新增中文说明 L272,273,282,287,289 |
| ✅ 已读（推定） | [MHATokenToKVPool.\_\_init\_\_](python/sglang/srt/mem_cache/memory_pool.py#L1817) | 建立各层 K/V 的实际存储 | B · 分支补全：branch=kv_setup;direct=5;covered_before_branch=5/6 |

### 请求与 KV 槽位分配

已读 **1/8**，未读 **7**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ⬜ 未读 | [ReqToTokenPool.alloc](python/sglang/srt/mem_cache/memory_pool.py#L304) | 为活跃请求分配映射表行 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ReqToTokenPool.free](python/sglang/srt/mem_cache/memory_pool.py#L352) | 归还请求映射表行 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读 | [alloc\_for\_extend](python/sglang/srt/mem_cache/allocation.py#L282) | 为新增 prefill token 分配 KV | D · 直接注释：新增中文说明 L292,293,294,295,296,297,298,301,302,305,308,330,342,355,359,377,391,412 |
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

已读 **3/6**，未读 **3**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [SamplingParams.normalize](python/sglang/srt/sampling/sampling_params.py#L218) | 将生成参数规范化为采样语义 | D · 直接注释：新增中文说明 L253 |
| ⬜ 未读 | [LogitsProcessor.forward](python/sglang/srt/layers/logits_processor.py#L343) | 选择需要输出的 token 并形成 logits | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [LogitsProcessor.\_get\_logits](python/sglang/srt/layers/logits_processor.py#L657) | 隐藏状态经过 LM head 得到词表分数 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ⬜ 未读 | [ModelRunner.\_preprocess\_logits](python/sglang/srt/model_executor/model_runner.py#L1855) | 惩罚项、掩码与 logit 偏置 | U · 待覆盖：无直接证据，且未满足补全规则 |
| ✅ 已读（推定） | [ModelRunner.sample](python/sglang/srt/model_executor/model_runner.py#L1883) | 调用 sampler 并返回 token ID | C · 链路补全：chain=worker_sample;anchors=worker_forward->sample |
| ✅ 已读 | [Sampler.forward](python/sglang/srt/layers/sampler.py#L98) | 贪心与概率采样的核心分派 | D · 直接注释：新增中文说明 L108 |

### 前向结果进入请求状态

已读 **3/5**，未读 **2**。

| 阅读状态 | 方法 | 核心概念 | 判定依据 |
| --- | --- | --- | --- |
| ✅ 已读 | [Scheduler.process\_batch\_result](python/sglang/srt/managers/scheduler.py#L4371) | 按批次类型处理前向结果 | D · 直接注释：新增中文说明 L4376 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_prefill](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L240) | 处理 prefill 输出和请求状态 | D · 直接注释：新增中文说明 L245 |
| ✅ 已读 | [SchedulerBatchResultProcessor.process\_batch\_result\_decode](python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L870) | 接收新 token 并推进 decode 状态 | D · 直接注释：新增中文说明 L875 |
| ⬜ 未读 | [Req.update\_finish\_state](python/sglang/srt/managers/schedule_batch.py#L1675) | 判断长度、终止 token 和停止字符串 | U · 待覆盖：无直接证据，且未满足补全规则 |
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
| ✅ 已读 | [TokenizerManager.handle\_loop](python/sglang/srt/managers/tokenizer_manager.py#L2326) | 接收返回消息并按类型分发 | D · 直接注释：新增中文说明 L2329,2333,2334,2340 |
| ✅ 已读 | [TokenizerManager.\_handle\_batch\_output](python/sglang/srt/managers/tokenizer_manager.py#L2347) | 更新 ReqState 并唤醒等待者 | D · 直接注释：新增中文说明 L2349,2355,2364,2366,2378,2491,2503,2507,2508,2638,2646,2652,2665 |
| ✅ 已读 | [TokenizerManager.\_wait\_one\_response](python/sglang/srt/managers/tokenizer_manager.py#L1827) | 等待并产出请求结果 | D · 直接注释：新增中文说明 L1833,1840,1841,1842,1848,1865,1866,1867,1874,1881,1882,1894,1904,1923,1932,1937,1946 |
| ✅ 已读 | [OpenAIServingChat.\_handle\_non\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1811) | 收集完整结果并构造 chat 响应 | D · 直接注释：新增中文说明 L1818,1821,1825 |
| ✅ 已读（推定） | [OpenAIServingChat.\_handle\_streaming\_request](python/sglang/srt/entrypoints/openai/serving_chat.py#L1517) | 建立 chat 流式响应 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |
| ✅ 已读（推定） | [OpenAIServingChat.\_generate\_chat\_stream](python/sglang/srt/entrypoints/openai/serving_chat.py#L1545) | 将增量结果转换为 SSE 消息 | C · 链路补全：chain=chat_streaming;anchors=handle_api_request->generate_request |

## 统计范围与来源

SGLang 常规文本生成核心链路：启动、请求、调度、KV、前向、采样、回包；包含 normal/overlap、chunked prefill、基础 decode CUDA Graph，以及 Qwen 3.5 dense 文本主干的两类解码层。

模型选 Qwen 3.5 dense：Qwen3_5ForConditionalGeneration 的文本路径；普通 Attention 后端取 FlashAttention，线性注意力跟到 GatedDeltaNet 与 RadixLinearAttention 接口；前缀缓存取基础 RadixCache，KV 槽位取 paged allocator。同类实现不重复扩展分母。

- 范围版本：v2。
- 统计对象：当前工作区，包含未提交修改。
- HEAD：`7191bda32a495b6617edbd47a3e50dfe018a9b7d`。
- 固定对照基线：`6388b6cfb1d93c253714a408f1d66a093302acd7`。
- 源码指纹：`f38eeffbcb567ab7b35093a9a9c2545fa14086f46aa433a6c5722c4e99b76b3c`。
- 范围指纹：`d2f38b91a6ad05a3a74296db8e23690abb75471f254735a7e5c8862acc69e27f`。
