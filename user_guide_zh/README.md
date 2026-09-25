# SGLang 中文学习指南(Chinese Learning Guide)

围绕服务配置、注意力与缓存、推测解码和量化，结合概念、示意图、配置与代码入口学习 SGLang 推理。

## 1. 服务与调优(Serving and Tuning)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 1.1 | [大模型服务功能与配置(LLM Serving Features)](<1.1 大模型服务功能与配置(LLM Serving Features).md>) | 接口选择、结构化输出、工具调用与推理内容解析 |
| 1.2 | [超参数调优(Hyperparameter Tuning)](<1.2 超参数调优(Hyperparameter Tuning).md>) | 请求并发、显存分配、调度与批大小 |

## 2. 注意力与缓存(Attention and Caching)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 2.1 | [注意力机制与注意力后端(Attention Mechanisms and Backends)](<2.1 注意力机制与注意力后端(Attention Mechanisms and Backends).md>) | 模型结构、配置与计算实现的关系 |
| 2.2 | [分层稀疏注意力(Hierarchical Sparse Attention)](<2.2 分层稀疏注意力(Hierarchical Sparse Attention).md>) | 稀疏选择、锁页内存与 CPU/GPU 缓存搬运 |
| 2.3 | [会话感知基数缓存(Session-Aware Radix Cache)](<2.3 会话感知基数缓存(Session-Aware Radix Cache).md>) | 会话引用、软保护与缓存淘汰 |

## 3. 推测解码(Speculative Decoding)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 3.1 | [推测解码：MTP 与 EAGLE-3(Speculative Decoding)](<3.1 推测解码：MTP 与 EAGLE-3(Speculative Decoding).md>) | 草稿生成、目标校验与部署参数 |
| 3.2 | [自适应推测解码(Adaptive Speculative Decoding)](<3.2 自适应推测解码(Adaptive Speculative Decoding).md>) | 按接受长度和批大小调整草稿步数 |

## 4. 量化(Quantization)

| 编号 | 文章 | 主要内容 |
|---|---|---|
| 4.1 | [模型量化入门(Model Quantization)](<4.1 模型量化入门(Model Quantization).md>) | 数值精度、离线量化算法、权重格式与加载 |
| 4.2 | [量化键值缓存(Quantized KV Cache)](<4.2 量化键值缓存(Quantized KV Cache).md>) | FP8/FP4 缓存、缩放因子、显存收益与准确率 |

## 阅读与命名约定

- 文件名与一级标题统一采用“章节.文章 中文标题(English Title)”，同一主题共用章节编号。
- 各篇核心术语首次出现时给出“中文术语(English Term，缩写)”，后文可以使用已定义的中文或缩写；代码标识和配置参数保留原名。
- 每篇正文不超过 120 行，示意图按一行计；插图与生成提示词统一保存在本目录的 [`assets`](assets/) 子目录中。
- 核心请求处理链路另见 [`docs/learn`](../docs/learn/)，从 HTTP 接入、输入分词到调度和模型执行。
