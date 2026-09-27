# SGLang 中文学习指南(Chinese Learning Guide)

围绕服务配置、并行策略、注意力与缓存、推测解码、量化与剪枝、编译与计算图，结合概念、示意图、配置与代码入口学习 SGLang 推理。

## 1. 服务与调优(Serving and Tuning)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 1.1 | [大模型服务功能与配置(LLM Serving Features)](<1.1 大模型服务功能与配置(LLM Serving Features).md>) | 接口选择、结构化输出、工具调用与推理内容解析 |
| 1.2 | [超参数调优(Hyperparameter Tuning)](<1.2 超参数调优(Hyperparameter Tuning).md>) | 请求并发、显存分配、调度与批大小 |
| 1.3 | [并行策略与模型网关(Parallelism and Model Gateway)](<1.3 并行策略与模型网关(Parallelism and Model Gateway).md>) | DP、DPA 与 SMG 完整译文，附从请求路由到模型内部的分层总览 |
| 1.4 | [专家并行(Expert Parallelism)](<1.4 专家并行(Expert Parallelism).md>) | EP 通信与计算后端、可扩展框架、计算通信重叠与负载均衡 |
| 1.5 | [预填充与解码分离(PD Disaggregation)](<1.5 预填充与解码分离(PD Disaggregation).md>) | PD 分离原理、路由与 Responses API 限制、Mooncake/NIXL 部署、异构 TP 与 GPU 暂存缓冲区 |
| 1.5 | [LoRA 服务(LoRA Serving)](<1.5 LoRA 服务(LoRA Serving).md>) | 共享基础模型、按请求选择适配器、动态加载、GPU 常驻、DPA 与重叠加载 |
| 1.6 | [混合专家模型入门(Mixture of Experts)](<1.6 混合专家模型入门(Mixture of Experts).md>) | 专家 MLP、上下文路由、Top-k 与加权合并、词元批量计算及参数与计算量 |
| 1.6 | [编码、预填充与解码分离(EPD Disaggregation)](<1.6 编码、预填充与解码分离(EPD Disaggregation).md>) | 视觉编码独立扩展、编码结果传输、全局多模态嵌入缓存、Qwen VL 与 gRPC 部署 |
| 1.7 | [长上下文流水线并行(Pipeline Parallelism for Long Context)](<1.7 长上下文流水线并行(Pipeline Parallelism for Long Context).md>) | 异步 P2P 通信、动态分块、平滑因子调优与 NVIDIA H20 部署示例 |
| 1.8 | [SGLang 模拟器(SGLang Simulator)](<1.8 SGLang 模拟器(SGLang Simulator).md>) | 延迟预测、逻辑时间模拟与实际时间回放、工作负载发送和指标输出 |
| 1.9 | [模型加载(Model Loading)](<1.9 模型加载(Model Loading).md>) | 加载格式、加载器额外配置、多线程与预取、远程和流式权重加载 |

## 2. 注意力与缓存(Attention and Caching)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 2.1 | [注意力机制与注意力后端(Attention Mechanisms and Backends)](<2.1 注意力机制与注意力后端(Attention Mechanisms and Backends).md>) | 模型结构、配置与计算实现的关系 |
| 2.2 | [分层稀疏注意力(Hierarchical Sparse Attention)](<2.2 分层稀疏注意力(Hierarchical Sparse Attention).md>) | 稀疏选择、锁页内存与 CPU/GPU 缓存搬运 |
| 2.3 | [会话感知基数缓存(Session-Aware Radix Cache)](<2.3 会话感知基数缓存(Session-Aware Radix Cache).md>) | 会话引用、软保护与缓存淘汰 |
| 2.4 | [解码上下文并行(Decode Context Parallelism)](<2.4 解码上下文并行(Decode Context Parallelism).md>) | MLA KV Cache 分片、局部注意力合并与组合配置 |

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

- 文件名与一级标题统一采用“章节.文章 中文标题(English Title)”，同一主题共用章节编号。
- 各篇核心术语首次出现时给出“中文术语(English Term，缩写)”，后文可以使用已定义的中文或缩写；代码标识和配置参数保留原名。
- 翻译及基于原文撰写文章时，完整保留原文的内容、条件、示例、数据与链接；只有明确要求删减时才删减，补充内容单独标注。
- 翻译请求默认将完整译文直接保存到本目录，并更新目录索引；随后在项目中的文档上共同修改。
- 新增文章末尾附术语与生词表：缩写列英文全称、中文释义和简明英文释义；需要解释的英文单词另附音标，音标紧邻单词；不标字母缩写的音标，不附音频。
- 文档一般不超过 120 行，示意图按一行计；按已确认的完整翻译要求，1.3、1.4、1.5、1.6、1.7、5.1 可以超过此限制，内容仅按明确要求删减。插图与生成提示词统一保存在本目录的 [`assets`](assets/) 子目录中。
- 核心请求处理链路另见 [`docs/learn`](../docs/learn/)，从 HTTP 接入、输入分词到调度和模型执行。
