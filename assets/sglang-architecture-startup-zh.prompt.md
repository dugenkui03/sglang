# 启动与进程拓扑图：提示词

生成方式：内置 image_gen；风格参考 `assets/sglang-architecture-srt-zh.png`（只参考风格）。成图：`assets/sglang-architecture-startup-zh.png`（16:9）。

```text
Use case: infographic-diagram (architecture). Chinese technical infographic for the SGLang learning guide, same series style as the reference image (orange-red rounded badge with number, orange-red bold title, light colored panels, rounded cards, blue arrows). Landscape 16:9, high resolution, fully opaque white background. Use the reference ONLY for style. Render ONLY the exact quoted strings below. IMPORTANT WORD CHOICE: the Chinese word for process is 进程 (jin cheng). Every place below that says 进程 must be rendered as 进程, never as 流程. Spell TokenizerManager exactly as T-o-k-e-n-i-z-e-r-M-a-n-a-g-e-r with no extra character before it.

Header: badge "00", title "启动与进程拓扑", subtitle "一条命令拉起三类进程，各自初始化后才开始接请求".

Left panel header "① 启动顺序（主进程）": six stacked cards joined by down arrows, exactly:
"sglang serve → Engine._launch_subprocesses"
"发布配置、分配管道地址"
"启动 Scheduler 进程（每张 GPU 一个）"
"启动 DetokenizerManager 进程"
"主进程创建 TokenizerManager"
"等 Scheduler 就绪，启动 HTTP 服务并预热"

Middle panel header "② Scheduler 进程初始化": six stacked cards joined by down arrows, exactly:
"建 ZMQ 管道（只有 rank 0）"
"创建 TpModelWorker 和 ModelRunner"
"初始化多卡通信、加载权重" (teal highlight, with a small second line "权重在启动时加载")
"分配 KV 缓存池"
"初始化注意力后端、捕获 CUDA Graph"
"进入主循环（默认 overlap）"

Right panel header "③ 有几个进程": a two-column table with five rows, exactly:
"单卡" | "主进程 + 1 个 Scheduler + 1 个 Detokenizer"
"TP = N" | "N 个 Scheduler"
"TP = N，PP = M" | "N × M 个 Scheduler"
"DP = K" | "多一个 DataParallelController，K 组 Scheduler"
"多机" | "只有 0 号节点运行 HTTP 和 TokenizerManager"

Bottom strip header "三条 ZMQ 管道": three chips side by side, each with a small arrow icon inside and a small caption below, exactly:
"TokenizerManager → Scheduler" caption "请求"
"Scheduler → DetokenizerManager" caption "新 token ID"
"DetokenizerManager → TokenizerManager" caption "新文本"

All text fully inside the canvas, exact spelling, no extra words.
```

## 后处理（Pillow，DejaVu Sans）

- 生成图把 `TokenizerManager` 按提示逐字母加连字符印了出来（5 处），全部按原位置重画为 `TokenizerManager`。
- 右栏标题被写成“有几个场景”：从同图“② Scheduler 进程初始化”复制“进程”两个字形，替换“场景”。

## 未采用的尝试

- 02 SRT 内部图重做了三版，箭头的起点、终点和标签总有错误（例如 TpModelWorker 直接连到 Detokenizer、回传箭头方向反了），因此保留原 02 图，在 LEARN.md 里用 Mermaid 补充数据对象与管道。
