# TokenizerManager 类说明

负责输入准备、请求状态和异步回包；GPU 调度与前向计算在下游。以下只列主要方法。

![TokenizerManager 的结构及其在一次推理中的位置](tokenizer_manager_科普图.png)
## 1. 继承结构

Mixin 是提供一组能力的父类；TokenizerManager 通过多继承获得控制与评分方法。

```mermaid
classDiagram
    TokenizerControlMixin <|-- TokenizerManager
    TokenizerManagerScoreMixin <|-- TokenizerManager
    TokenizerManager "1" --> "多" ReqState : rid_to_state 管理
    TokenizerControlMixin : flush_cache()
    TokenizerControlMixin : load_lora_adapter()
    TokenizerControlMixin : open_session()
    TokenizerManagerScoreMixin : score_request()
    TokenizerManagerScoreMixin : score_prompts()
    TokenizerManager : generate_request()
    ReqState : out_list / finished / event
```

## 2. 本类主要方法分组

下图表示职责归属，不表示调用顺序。

```mermaid
flowchart LR
    T[TokenizerManager] --> A["初始化<br/>__init__<br/>init_tokenizer_and_processor<br/>init_ipc_channels"]
    T --> B["请求与输入<br/>generate_request<br/>_tokenize_one_request<br/>_validate_one_request<br/>_handle_batch_request"]
    T --> C["发送与回包<br/>_send_one_request<br/>handle_loop<br/>_handle_batch_output<br/>_wait_one_response"]
    T --> D["状态与控制<br/>_init_req_state<br/>abort_request<br/>pause_generation / continue_generation<br/>update_weights_from_disk"]
    T --> E["观测与故障处理<br/>collect_metrics<br/>dump_requests<br/>sigterm_watchdog"]
```

## 3. 核心协作：发送与收包分开运行

```mermaid
flowchart TB
    A[generate_request] --> B["_init_req_state<br/>登记 rid → ReqState"]
    B --> C["_tokenize_one_request<br/>分词、校验、构造内部消息"]
    C --> D[_send_one_request] --> S["下游：Scheduler → GPU 执行 → Detokenizer"]
    D --> W["_wait_one_response<br/>等待该请求的 event"]
    S --> H["handle_loop<br/>后台持续收包"] --> O["_handle_batch_output<br/>按 rid 更新状态并通知 event"]
    O -. 唤醒 .-> W
    W --> R["yield 结果给上游<br/>流式继续等待，完成后退出"]
```

请求登记后用 rid 对齐回包；ReqState 是状态对象，不是父类。完整生成通常包含 prefill 和多轮 decode。
