# PortArgs：服务组件之间的通信地址

## 组件通信流程图

```mermaid
flowchart LR
    E["Engine"] <-->|"rpc_ipc_name<br/>控制 RPC 与响应"| S["Scheduler"]
    T["TokenizerManager"] -->|"scheduler_input_ipc_name<br/>已分词请求"| S
    S -->|"detokenizer_ipc_name<br/>生成的 token"| D["DetokenizerManager"]
    D -->|"tokenizer_ipc_name<br/>解码结果"| T
```

主请求链路中的三条箭头分别对应 `TokenizerManager.send_to_scheduler`、`DetokenizerManager.recv_from_scheduler`、`TokenizerManager.recv_from_detokenizer` 等 socket；`Engine` 通过 `rpc_ipc_name` 发送管理 RPC 并接收结果。单次普通 HTTP 推理请求不会先经过 `Engine` 的 RPC。

## 整体链路的简短时序图

蓝色为一次请求的消息流；橙色是单独的 Engine 控制调用示例。图中字段名均与 `PortArgs` 一致。

```mermaid
sequenceDiagram
    participant C as 客户端
    participant T as TokenizerManager
    participant S as Scheduler
    participant D as DetokenizerManager
    participant E as Engine
    rect rgb(232, 242, 255)
        Note over C,D: 请求处理：各组件通过 PortArgs 中的地址交换消息
        C->>T: 提交生成请求
        T->>S: 已分词请求 [scheduler_input_ipc_name]
        S->>D: 生成的 token [detokenizer_ipc_name]
        D->>T: 解码结果 [tokenizer_ipc_name]
        T-->>C: 返回结果或流式输出
    end
    rect rgb(255, 243, 225)
        Note over S,E: 独立的控制通道：离线 Engine API 示例
        E->>S: collective_rpc(...) [rpc_ipc_name]
        S-->>E: RpcReqOutput [rpc_ipc_name]
    end
```

## PortArgs 在启动链路中的位置

```mermaid
flowchart LR
    A["Engine._launch_subprocesses()"] --> B["PortArgs.init_new(server_args)<br/>生成通信地址"]
    B --> C["各组件创建 ZeroMQ socket"]
    C --> D["请求处理与控制消息"]
    style B fill:#fff0c2,stroke:#b7791f,stroke-width:2px
```

这里的 `nccl_port` 用于分布式通信初始化；`metrics_ipc_name`、`tokenizer_worker_ipc_name` 和 `decoupled_spec_ipc_config` 服务于监控或可选模式，因此没有放进上面的普通请求时序图。
