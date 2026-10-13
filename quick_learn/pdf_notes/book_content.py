"""Functional grouping labels for the source-comment collection."""

GROUPS = [
    ("启动与运行配置", "从命令行到可处理请求的服务", "STARTUP"),
    ("模型构建与权重加载", "先创建对象与参数，再装入权重", "MODEL LOADING"),
    ("HTTP 请求与分词", "从聊天消息到调度器可接收的请求", "REQUEST INPUT"),
    ("调度与组批", "为每一轮选择请求、模式和预算", "SCHEDULING"),
    ("前向执行与跨轮次数据", "从 ScheduleBatch 到模型调用", "FORWARD EXECUTION"),
    ("Embedding、线性投影与并行", "查表、矩阵乘法与多 GPU 分工", "TENSORS & PARALLELISM"),
    ("注意力与归一化", "分清全注意力、GDN 与数值尺度", "ATTENTION & NORM"),
    ("KV 池与前缀缓存", "编号映射、实际张量与前缀复用", "CACHE & MEMORY"),
    ("采样、解码与回包", "从词表分数到用户看到的文本", "RESULT OUTPUT"),
    ("Python 机制与其他实现", "生成器、事件循环和旁支入口", "SUPPORTING CONCEPTS"),
]


def group_for(e):
    p, n = e["file"], e["name"]
    if "/multimodal_gen/" in p or "/hardware_backend/" in p or "/speculative/" in p or "qwen3_5_text.py" in p:
        return 9
    if "/cli/" in p or p.endswith("launch_server.py") or "/plugins/" in p or p.endswith(("server_args.py", "runtime_context.py", "watchdog.py")) or "/ray/" in p:
        return 0
    if p.endswith("engine.py") or n in {"launch_server", "_setup_and_run_http_server", "run_scheduler_process", "run_detokenizer_process"}:
        return 0
    if "model_loader/" in p or "load_model_utils" in p or "model_config.py" in p or "models/registry.py" in p or "layer_setup.py" in p or "distributed/utils.py" in p:
        return 1
    if n in {"ModelRunner.__init__", "ModelRunner.initialize", "ModelRunner.load_model", "ModelRunner.init_torch_distributed", "TpModelWorker.__init__", "TpModelWorker._init_model_config", "TpModelWorker._init_model_runner", "Scheduler.init_tp_model_worker", "Scheduler.init_model_worker", "Scheduler.init_all_cuda_graphs", "make_layers"}:
        return 1
    if "models/qwen3_vl.py" in p:
        return 1 if e["kind"] == "class" or n.endswith(".__init__") else 4
    if "models/qwen3_5.py" in p:
        if "AttentionDecoderLayer" in n or "GatedDeltaNet" in n:
            return 6
        return 1 if e["kind"] == "class" or ".__init__" in n else 4
    if "mem_cache/" in p or "pool_configurator.py" in p or n in {"ModelRunner.alloc_memory_pool", "TpModelWorker.alloc_memory_pool", "Scheduler.init_target_memory_pool", "Scheduler.init_memory_pools", "get_available_gpu_memory"}:
        return 7
    if any(k in p for k in ("layers/linear.py", "quantization/unquant.py", "vocab_parallel_embedding.py")):
        return 5
    if any(k in p for k in ("layers/layernorm.py", "radix_attention.py", "radix_linear_attention.py")):
        return 6
    if any(k in p for k in ("layers/logits_processor.py", "layers/sampler.py", "sampling/", "detokenizer_manager.py", "batch_result_processor.py", "output_streamer.py")):
        return 8
    if n in {"Scheduler.process_batch_result", "ReqState", "ReqState.get_text", "TokenizerManager._wait_one_response", "TokenizerManager.auto_create_handle_loop", "TokenizerManager.handle_loop", "TokenizerManager._handle_batch_output", "OpenAIServingChat._handle_non_streaming_request", "OpenAIServingBase._handle_non_streaming_request", "GenerationBatchResult", "BatchTokenIDOutput"}:
        return 8
    if "model_executor/" in p or "overlap_utils.py" in p or "tp_worker.py" in p or n == "Scheduler.run_batch":
        return 4
    if "scheduler.py" in p or "schedule_batch.py" in p or "schedule_policy.py" in p or "request_receiver.py" in p:
        return 3
    if any(k in p for k in ("http_server.py", "entrypoints/openai/", "tokenizer_manager.py", "io_struct.py", "hf_transformers/")):
        return 2
    return 9

