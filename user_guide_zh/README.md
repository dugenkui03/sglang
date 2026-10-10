# SGLang 中文学习指南(Chinese Learning Guide)

围绕服务配置、并行策略、注意力与缓存、推测解码、量化与剪枝、编译与计算图，结合概念、示意图、配置与代码入口学习 SGLang 推理。

## 核心组件(Core Components)

- [核心组件一：Engine](核心组件一：Engine.md)：推理接口、组件启动时序与资源关闭。
- [核心组件二：TokenizerManager](核心组件二：TokenizerManager.md)：分词、提交请求与接收结果，控制操作转发给 Scheduler。
- [核心组件三：DetokenizerManager](核心组件三：DetokenizerManager.md)：增量解码 token ID、维护解码状态并回传文本。
- [核心组件四：Scheduler](核心组件四：Scheduler.md)：主循环四步（收请求、组批、执行、处理结果）、prefill 与 decode 的选择、overlap 调度。
- [核心组件五：TpModelWorker 与 ModelRunner](核心组件五：TpModelWorker与ModelRunner.md)：启动时加载权重、分配 KV 池、捕获 CUDA Graph；每轮构造 ForwardBatch、前向、采样。
- [核心组件六：ScheduleBatch](核心组件六：ScheduleBatch.md)：Scheduler 组好的一批请求，组批时创建、前向时转成 ForwardBatch、处理结果时再读；继承关系与核心方法。

## 核心流程(Core Workflows)

- [核心流程一：模型加载](核心流程一：模型加载.md)：从 ModelRunner 初始化到 Qwen3.5 模型创建、权重文件下载，以及 weight_loader 的绑定与执行。
- [核心流程二：模型前向](核心流程二：模型前向.md)：Qwen3.5 的 embedding、两类 Decoder 层、最终归一化与 LM Head 计算，共三张调用图。

## 核心概念(Core Concepts)

- [核心概念一：SGLang中的并行策略](核心概念一：SGLang中的并行策略.md)：TP、PP、DP、EP、CP 五种基础策略及细分变体，卡怎么分，rank 坐标与请求分发。
- [核心概念二：CPU 调度与 GPU 执行（CUDA Stream 与 Event）](<核心概念二：CPU 调度与 GPU 执行（CUDA Stream 与 Event）.md>)：三条流如何分工，event 的两种状态和等待方式，以及 overlap 调度如何靠它们并行。
- [核心概念三：显存管理（槽位、KV Cache 与 FutureMap）](<核心概念三：显存管理（槽位、KV Cache 与 FutureMap）.md>)：槽位、req_to_token 与 KV 池的映射，KV 编号何时分配、K/V 何时写入和读取，FutureMap 如何按槽位把 token 交给下一轮。

## 基础知识(Fundamentals)

- [基础知识：ZeroMQ 进程间通信](<基础知识：ZeroMQ 进程间通信.md>)：PUSH/PULL socket、bind 与 connect，以及进程之间如何收发消息。
- [基础知识二：一次推理的全过程（启动、组批、前向与采样）](<基础知识二：一次推理的全过程（启动、组批、前向与采样）.md>)：启动时加载模型、建 KV 池；请求如何经 ZMQ 提交、`_wait_one_response` 如何等待结果；每轮组批、`run_batch` 前向、`lm_head` 打分、采样出下一个 token，以及 CUDA Graph 与 eager 路径和数据形状的变化。
- [基础知识三：推理并行参数解释](<基础知识三：推理并行参数解释.md>)：每张卡一个 Scheduler 进程怎么启动、`gpu_id` 怎么算，TP、PP、DP、attn_cp、moe_dp、EP 各切什么，size 与 rank 的约束，以及一次前向里各种切法怎么配合。
- [基础知识四：Python 中的进程、线程与协程](<基础知识四：Python 中的进程、线程与协程.md>)：三者的关系与区别、GIL、协程如何在 `await` 处轮流执行、`async`/`await`/`yield` 语法速查，以及 sglang 里多进程、后台线程和 TokenizerManager 协程的对应代码。
- [基础知识五：大模型中的 GPU（显存与并行计算）](<基础知识五：大模型中的 GPU（显存与并行计算）.md>)：显存主要花在权重、KV Cache 和前向中间结果上；KV 编号从分配、使用、缓存保留到回收的生命周期；GPU 靠高显存带宽和矩阵分块并行适合大模型计算。
- [基础知识六：矩阵乘法、点积与外积](<基础知识六：矩阵乘法、点积与外积.md>)：先用简单向量理解点积和外积，再用“左边取一行、右边取一列做点积”理解矩阵乘法及特殊形状的乘法。

## 思考(Reflections)

- [思考一：关于组批请求的数据模型](思考一：关于组批请求的数据模型.md)：把组批看成调度器、数据容器和有状态元素之间的交互：调度器按规则搬运 Req、读写状态；补充请求行、KV 索引、RadixCache 与 KV 池的关系图。
- [思考二：关于推理过程的思考](思考二：关于推理过程的思考.md)：模型架构、权重与实现的关系，前向计算与缓存读写，训练和推理的区别，以及新模型首日支持。

## 1. 服务与调优(Serving and Tuning)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 1.1 | [大模型服务功能与配置(LLM Serving Features)](<1.1 大模型服务功能与配置(LLM Serving Features).md>) | 接口选择、结构化输出、工具调用与推理内容解析 |
| 1.2 | [超参数调优(Hyperparameter Tuning)](<1.2 超参数调优(Hyperparameter Tuning).md>) | 请求并发、显存分配、调度与批大小 |
| 1.3 | [并行策略与模型网关(Parallelism and Model Gateway)](<1.3 并行策略与模型网关(Parallelism and Model Gateway).md>) | DP、DPA 与 SMG 完整译文，附从请求路由到模型内部的分层总览 |
| 1.4 | [专家并行(Expert Parallelism)](<1.4 专家并行(Expert Parallelism).md>) | EP 通信与计算后端、可扩展框架、计算通信重叠与负载均衡 |
| 1.4.1 | [弹性专家并行(Elastic Expert Parallelism)](<../python/sglang/srt/elastic_ep/1.4.1 弹性专家并行(Elastic Expert Parallelism).md>) | 成员故障与扩容、专家重新布局、CPU 权重备份及核心实现时序 |
| 1.5 | [预填充与解码分离(PD Disaggregation)](<1.5 预填充与解码分离(PD Disaggregation).md>) | PD 分离原理、路由与 Responses API 限制、Mooncake/NIXL 部署、异构 TP 与 GPU 暂存缓冲区 |
| 1.5 | [LoRA 服务(LoRA Serving)](<1.5 LoRA 服务(LoRA Serving).md>) | 共享基础模型、按请求选择适配器、动态加载、GPU 常驻、DPA 与重叠加载 |
| 1.6 | [混合专家模型入门(Mixture of Experts)](<1.6 混合专家模型入门(Mixture of Experts).md>) | 专家 MLP、上下文路由、Top-k 与加权合并、词元批量计算及参数与计算量 |
| 1.6 | [编码、预填充与解码分离(EPD Disaggregation)](<1.6 编码、预填充与解码分离(EPD Disaggregation).md>) | 视觉编码独立扩展、编码结果传输、全局多模态嵌入缓存、Qwen VL 与 gRPC 部署 |
| 1.7 | [长上下文流水线并行(Pipeline Parallelism for Long Context)](<1.7 长上下文流水线并行(Pipeline Parallelism for Long Context).md>) | 异步 P2P 通信、动态分块、平滑因子调优与 NVIDIA H20 部署示例 |
| 1.8 | [SGLang 模拟器(SGLang Simulator)](<1.8 SGLang 模拟器(SGLang Simulator).md>) | 延迟预测、逻辑时间模拟与实际时间回放、工作负载发送和指标输出 |
| 1.9 | [模型加载(Model Loading)](<1.9 模型加载(Model Loading).md>) | 加载格式、加载器额外配置、多线程与预取、远程和流式权重加载 |
| 1.10 | [可观测性(Observability)](<1.10 可观测性(Observability).md>) | Prometheus 指标、日志、请求转储与回放、崩溃诊断和 CUDA 设备核心转储 |
| 1.11 | [远程权重加载(R-Fork)](<1.11 远程权重加载(R-Fork).md>) | 种子实例、GPU 间权重传输、启动加速及 NCCL、TransferEngine、ModelExpress 配置 |

## 2. 注意力与缓存(Attention and Caching)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 2.1 | [注意力机制与注意力后端(Attention Mechanisms and Backends)](<2.1 注意力机制与注意力后端(Attention Mechanisms and Backends).md>) | 模型结构、配置与计算实现的关系 |
| 2.2 | [分层稀疏注意力(Hierarchical Sparse Attention)](<2.2 分层稀疏注意力(Hierarchical Sparse Attention).md>) | 稀疏选择、锁页内存与 CPU/GPU 缓存搬运 |
| 2.3 | [会话感知基数缓存(Session-Aware Radix Cache)](<2.3 会话感知基数缓存(Session-Aware Radix Cache).md>) | 会话引用、软保护与缓存淘汰 |
| 2.4 | [解码上下文并行(Decode Context Parallelism)](<2.4 解码上下文并行(Decode Context Parallelism).md>) | MLA KV Cache 分片、局部注意力合并与组合配置 |
| 2.5 | [预填充上下文并行(Prefill Context Parallelism)](<2.5 预填充上下文并行(Prefill Context Parallelism).md>) | Zigzag 与 Interleave 词元分配、注意力后端集成、组合限制与启动示例 |

## 3. 推测解码(Speculative Decoding)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 3.1 | [推测解码：MTP 与 EAGLE-3(Speculative Decoding)](<3.1 推测解码：MTP 与 EAGLE-3(Speculative Decoding).md>) | 草稿生成、目标校验与部署参数 |
| 3.2 | [自适应推测解码(Adaptive Speculative Decoding)](<3.2 自适应推测解码(Adaptive Speculative Decoding).md>) | 按接受长度和批大小调整草稿步数 |

## 4. 量化与剪枝(Quantization and Pruning)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 4.1 | [模型量化入门(Model Quantization)](<4.1 模型量化入门(Model Quantization).md>) | 数值精度、离线量化算法、权重格式与加载 |
| 4.2 | [量化键值缓存(Quantized KV Cache)](<4.2 量化键值缓存(Quantized KV Cache).md>) | FP8/FP4 缓存、缩放因子、显存收益与准确率 |
| 4.3 | [剪枝与推理感知压缩(Pruning and Reasoning-Aware Compression)](<4.3 剪枝与推理感知压缩(Pruning and Reasoning-Aware Compression).md>) | 剪枝原理、思维链校准与压缩后的推理效果 |

## 5. 编译与计算图(Compilation and Computation Graphs)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 5.1 | [分段 CUDA 图(Piecewise CUDA Graph)](<5.1 分段 CUDA 图(Piecewise CUDA Graph).md>) | 分段编译、捕获与回放、形状配置、内存优化、自定义算子兼容及源码入口 |

## 阅读与命名约定

- 特性文章的文件名与一级标题统一采用“章节.文章 中文标题(English Title)”，同一主题共用章节编号；核心组件系列采用“核心组件一：Engine”等命名，核心概念系列采用“核心概念一：SGLang中的并行策略”等命名。
- 各篇核心术语首次出现时给出“中文术语(English Term，缩写)”，后文可以使用已定义的中文或缩写；代码标识和配置参数保留原名。
- 翻译及基于原文撰写文章时，完整保留原文的内容、条件、示例、数据与链接；只有明确要求删减时才删减，补充内容单独标注。
- 翻译请求默认将完整译文直接保存到本目录，并更新目录索引；随后在项目中的文档上共同修改。
- 新增文章末尾附术语与生词表：缩写列英文全称、中文释义和简明英文释义；需要解释的英文单词另附音标，音标紧邻单词；不标字母缩写的音标，不附音频。
- 文档一般不超过 120 行，示意图按一行计；按已确认的完整翻译要求，1.3、1.4、1.5、1.6、1.7、2.5、5.1 可以超过此限制，内容仅按明确要求删减。插图与生成提示词统一保存在本目录的 [`assets`](assets/) 子目录中。
- 核心请求处理链路另见 [`docs/learn`](../docs/learn/)，从 HTTP 接入、输入分词到调度和模型执行。
