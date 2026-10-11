# Qwen3.5：Prefill / Decode × 两类注意力

四图均从文本主干入口到 logits，采样前停止。每页只展开指定类型的层；真实模型按配置混合执行两类层。

## Prefill × 全注意力（Full Attention）

单卡 CUDA，TP=PP=1，BF16 未量化，Eager，dense 纯文本生成，无输入 logprob。
以 FlashAttention 后端、普通 NHD 分页 KV 池为例；模型按层配置混合两类注意力。
B=请求数；N=本轮新 token 总数；H=hidden_size；V=vocab_size。每层类型由配置固定。

```text
Qwen3_5ForCausalLM.forward()：本轮 N 个 token，传入 token ID 或已有 input_embeds
├─ ① Embedding：准备 hidden_states=[N,H]
│  ├─ 已传 input_embeds → 直接使用（外层已完成向量化）
│  └─ 否则 → VocabParallelEmbedding.forward() → _embed_local_shard()
│       → UnquantizedEmbeddingMethod.embedding() → F.embedding()：按 token ID 查权重表
├─ residual_batch.start()：开始本轮残差流
├─ ② 循环 self.layers[layer_idx](...)；此页展开其中的全注意力层
│  → Qwen3_5AttentionDecoderLayer.forward()
│  ├─ attn_boundary.prepare() → input_layernorm：输入归一化
│  │    → GemmaRMSNorm → BaseFusedOp.forward() → GemmaRMSNorm.forward_cuda()
│  │    → _forward_impl() → gemma_rmsnorm() / gemma_fused_add_rmsnorm()（有残差）
│  ├─ self_attention() → _prepare_qkv_gate()
│  │  ├─ CUDA + 输出门控开启 → forward_prepare_cuda_fused()
│  │  │    → qkv_proj：QKVParallelLinear → 继承 ColumnParallelLinear.forward()
│  │  │    → UnquantizedLinearMethod.apply()：线性投影 x @ weight.T + bias
│  │  │    → F.linear() 或 BF16 GEMM 优化内核 → split()
│  │  │    → fused_qk_gemma_rmsnorm_rope_gate()：Q/K 归一化、RoPE、拆出 gate
│  │  └─ 原生对照分支 forward_prepare_native()（与上方二选一）
│  │       → 同一 qkv_proj → split/chunk → _apply_qk_norm() → q_norm/k_norm
│  │       → rotary_emb()：旋转位置编码（RoPE）
│  ├─ self_attention() 内继续 → RadixAttention.forward(q,k,v,...)
│  │    → get_attn_backend() → HybridLinearAttnBackend.forward() → forward_extend()
│  │    → FlashAttentionBackend.forward_extend() ← Prefill / EXTEND
│  │    ├─ 写入本轮 N 个 token 的 K/V；位置来自 KVWriteLoc.for_layer()
│  │    │    → HybridLinearKVPool.set_kv_buffer() → unwrap_write_loc()
│  │    │    → _transfer_full_attention_id()：模型层号 → 全注意力缓存层号
│  │    │    → MHATokenToKVPool.set_kv_buffer() → _store_kv_layer()
│  │    │    → _set_kv_buffer_impl() → store_cache() → _jit_kvcache_module()
│  │    │    → module.store_cache() → StoreKVCacheKernel.run() → LaunchKernel(...)
│  │    │    → store_kvcache_kernel()：写入 GPU 显存中的缓存槽位
│  │    ├─ prepare_paged_mha_query()：准备 Q 的 dtype/缩放；BF16/auto 通常无需转换
│  │    │    → get_paged_mha_kv_cache() → HybridLinearKVPool.get_kv_buffer()
│  │    │    → _wait_for_layer()（无 HiCache 时无需等待）→ _transfer_full_attention_id()
│  │    │    → MHATokenToKVPool.get_kv_buffer()
│  │    │    → get_key_buffer() / get_value_buffer() → _get_key_buffer() / _get_value_buffer()
│  │    │    → paged_kv_view()：取得共享 GPU 显存的 K/V 分页视图
│  │    └─ q.contiguous().view(...) → flash_attn_with_kvcache()
│  │         → Q=[N,Q头数,head_dim]；内核实际按 page_table 读取 K/V
│  │         → cu_seqlens_q 区分请求；每个新 token 可看缓存前缀及本轮较早 token/自身
│  │         → Q @ K.T → 缩放 + 因果掩码 → softmax → V 加权和
│  │         → o.view(...)：展平各头输出，得到 [N,Q头数 × head_dim]
│  ├─ self_attention() 内继续 → fused_sigmoid_mul()：注意力输出 × sigmoid(gate)
│  │    → o_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │    → F.linear() / BF16 GEMM：投影回 [N,H]；TP=1，无跨卡求和
│  ├─ attn_boundary.finish() → ffn_boundary.prepare()
│  │    → 残差相加 + post_attention_layernorm（GemmaRMSNorm）
│  ├─ ffn_boundary.exit() → Qwen2MoeMLP.forward()：稠密前馈网络（FFN）
│  │    ├─ gate_up_proj：MergedColumnParallelLinear → ColumnParallelLinear.forward()
│  │    │    → UnquantizedLinearMethod.apply()：投影得到 gate/up
│  │    ├─ SiluAndMul → BaseFusedOp.forward() → SiluAndMul.forward_cuda()
│  │    │    → silu_and_mul()：SiLU(gate) × up
│  │    └─ down_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  └─ ffn_exit.finish()：交给下一层；残差相加可延后到下一次归一化
├─ ③ 所有层完成 → residual_batch.final_norm() → access.final_norm_pair()
│    → self.norm（GemmaRMSNorm）：最终归一化 → 返回 hidden_states=[N,H]
└─ ④ 返回外层调用方 Qwen3VLForConditionalGeneration.forward()
     │ 实际外层对象为 Qwen3_5ForConditionalGeneration，继承此 forward()
     → LogitsProcessor.forward(..., self.lm_head, ...)
     ├─ _get_pruned_states()：每请求取本轮最后一行，[N,H] → [B,H]
     │    last_index = cumsum(extend_seq_lens) - 1
     ├─ _get_logits() → _compute_lm_head() → torch.matmul(hidden_states, lm_head.weight.T)
     │    ParallelLMHead 提供 weight=[V,H]；这里直接用权重，不调用其 forward()
     └─ _copy_logits_to_buffer() → LogitsProcessorOutput(next_token_logits=[B,V]) ← 采样前
```

## Decode × 全注意力（Full Attention）

单卡 CUDA，TP=PP=1，BF16 未量化，Eager，dense 纯文本生成，无输入 logprob。
以 FlashAttention 后端、普通 NHD 分页 KV 池为例；模型按层配置混合两类注意力。
B=请求数；N=B，每请求本轮输入一个 token；H=hidden_size；V=vocab_size。层类型由配置固定。

```text
Qwen3_5ForCausalLM.forward()：本轮 B 个 token，传入 token ID 或已有 input_embeds
├─ ① Embedding：准备 hidden_states=[B,H]
│  ├─ 已传 input_embeds → 直接使用（外层已完成向量化）
│  └─ 否则 → VocabParallelEmbedding.forward() → _embed_local_shard()
│       → UnquantizedEmbeddingMethod.embedding() → F.embedding()：按 token ID 查权重表
├─ residual_batch.start()：开始本轮残差流
├─ ② 循环 self.layers[layer_idx](...)；此页展开其中的全注意力层
│  → Qwen3_5AttentionDecoderLayer.forward()
│  ├─ attn_boundary.prepare() → input_layernorm：输入归一化
│  │    → GemmaRMSNorm → BaseFusedOp.forward() → GemmaRMSNorm.forward_cuda()
│  │    → _forward_impl() → gemma_rmsnorm() / gemma_fused_add_rmsnorm()（有残差）
│  ├─ self_attention() → _prepare_qkv_gate()
│  │  ├─ CUDA + 输出门控开启 → forward_prepare_cuda_fused()
│  │  │    → qkv_proj：QKVParallelLinear → 继承 ColumnParallelLinear.forward()
│  │  │    → UnquantizedLinearMethod.apply()：线性投影 x @ weight.T + bias
│  │  │    → F.linear() 或 BF16 GEMM 优化内核 → split()
│  │  │    → fused_qk_gemma_rmsnorm_rope_gate()：Q/K 归一化、RoPE、拆出 gate
│  │  └─ 原生对照分支 forward_prepare_native()（与上方二选一）
│  │       → 同一 qkv_proj → split/chunk → _apply_qk_norm() → q_norm/k_norm
│  │       → rotary_emb()：使用本轮各 token 在各自序列中的位置
│  ├─ self_attention() 内继续 → RadixAttention.forward(q,k,v,...)
│  │    → get_attn_backend() → HybridLinearAttnBackend.forward() → forward_decode()
│  │    → FlashAttentionBackend.forward_decode() ← 每请求一个新 token
│  │    ├─ 写入本轮 B 个 token 的 K/V；位置来自 KVWriteLoc.for_layer()
│  │    │    → HybridLinearKVPool.set_kv_buffer() → unwrap_write_loc()
│  │    │    → _transfer_full_attention_id()：模型层号 → 全注意力缓存层号
│  │    │    → MHATokenToKVPool.set_kv_buffer() → _store_kv_layer()
│  │    │    → _set_kv_buffer_impl() → store_cache() → _jit_kvcache_module()
│  │    │    → module.store_cache() → StoreKVCacheKernel.run() → LaunchKernel(...)
│  │    │    → store_kvcache_kernel()：写入 GPU 显存中的缓存槽位
│  │    ├─ prepare_paged_mha_query()：准备 Q 的 dtype/缩放；BF16/auto 通常无需转换
│  │    │    → get_paged_mha_kv_cache() → HybridLinearKVPool.get_kv_buffer()
│  │    │    → _wait_for_layer()（无 HiCache 时无需等待）→ _transfer_full_attention_id()
│  │    │    → MHATokenToKVPool.get_kv_buffer()
│  │    │    → get_key_buffer() / get_value_buffer() → _get_key_buffer() / _get_value_buffer()
│  │    │    → paged_kv_view()：取得共享 GPU 显存的 K/V 分页视图
│  │    └─ q.contiguous().view(...) → flash_attn_with_kvcache()
│  │         → Q=[B,Q头数,head_dim]；内核实际按 page_table 读取 K/V
│  │         → cache_seqlens 包含本轮 token；本轮 Q 关注历史及本轮 K/V
│  │         → Q @ K.T → 缩放 + 因果约束 → softmax → V 加权和
│  │         → o.view(...)：展平各头输出，得到 [B,Q头数 × head_dim]
│  ├─ self_attention() 内继续 → fused_sigmoid_mul()：注意力输出 × sigmoid(gate)
│  │    → o_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │    → F.linear() / BF16 GEMM：投影回 [B,H]；TP=1，无跨卡求和
│  ├─ attn_boundary.finish() → ffn_boundary.prepare()
│  │    → 残差相加 + post_attention_layernorm（GemmaRMSNorm）
│  ├─ ffn_boundary.exit() → Qwen2MoeMLP.forward()：稠密前馈网络（FFN）
│  │    ├─ gate_up_proj：MergedColumnParallelLinear → ColumnParallelLinear.forward()
│  │    │    → UnquantizedLinearMethod.apply()：投影得到 gate/up
│  │    ├─ SiluAndMul → BaseFusedOp.forward() → SiluAndMul.forward_cuda()
│  │    │    → silu_and_mul()：SiLU(gate) × up
│  │    └─ down_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  └─ ffn_exit.finish()：交给下一层；残差相加可延后到下一次归一化
├─ ③ 所有层完成 → residual_batch.final_norm() → access.final_norm_pair()
│    → self.norm（GemmaRMSNorm）：最终归一化 → 返回 hidden_states=[B,H]
└─ ④ 返回外层调用方 Qwen3VLForConditionalGeneration.forward()
     │ 实际外层对象为 Qwen3_5ForConditionalGeneration，继承此 forward()
     → LogitsProcessor.forward(..., self.lm_head, ...)
     ├─ _get_pruned_states()：保留本轮每请求的一行，仍为 [B,H]
     │    pruned_states = hidden_states（每请求已只有一个本轮 token）
     ├─ _get_logits() → _compute_lm_head() → torch.matmul(hidden_states, lm_head.weight.T)
     │    ParallelLMHead 提供 weight=[V,H]；这里直接用权重，不调用其 forward()
     └─ _copy_logits_to_buffer() → LogitsProcessorOutput(next_token_logits=[B,V]) ← 采样前
```

## Prefill × 线性注意力（Gated DeltaNet，GDN）

单卡 CUDA，TP=PP=1，BF16 未量化，Eager，dense 纯文本生成，无输入 logprob。
以 Triton GDN、普通连续状态池为例；图内缓存均在 GPU 显存。实际模型按配置混合两类层。
B=请求数；N=本轮新 token 总数；H=hidden_size；V=vocab_size。GDN 保存卷积与递推状态。

```text
Qwen3_5ForCausalLM.forward()：本轮 N 个 token，传入 token ID 或已有 input_embeds
├─ ① Embedding：准备 hidden_states=[N,H]
│  ├─ 已传 input_embeds → 直接使用（外层已完成向量化）
│  └─ 否则 → VocabParallelEmbedding.forward() → _embed_local_shard()
│       → UnquantizedEmbeddingMethod.embedding() → F.embedding()：按 token ID 查权重表
├─ residual_batch.start()：开始本轮残差流
├─ ② 循环 self.layers[layer_idx](...)；此页展开其中的线性注意力层
│  → Qwen3_5LinearDecoderLayer.forward()
│  ├─ attn_boundary.prepare() → input_layernorm：输入归一化
│  │    → GemmaRMSNorm → BaseFusedOp.forward() → GemmaRMSNorm.forward_cuda()
│  │    → _forward_impl() → gemma_rmsnorm() / gemma_fused_add_rmsnorm()（有残差）
│  ├─ self.linear_attn(...) → Qwen3_5GatedDeltaNet.forward()
│  │  ├─ _forward_input_proj()：生成 projected_states_qkvz / projected_states_ba
│  │  │    ├─ BF16、已准备打包权重且 N≤1024 → bf16_gemm_dispatch()：融合两次投影
│  │  │    └─ 其他情况分别调用 in_proj_qkvz / in_proj_ba（MergedColumnParallelLinear）
│  │  │         → ColumnParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │  │         → F.linear() / BF16 GEMM：本质为 x @ weight.T + bias
│  │  ├─ qwen3_5_gdn_prefill_projection_views()：支持头数比例的 CUDA 路径
│  │  │    → 拆出 mixed_qkv、z、b、a 的视图；z 用于输出门控
│  │  ├─ RadixLinearAttention.forward() → get_attn_backend()
│  │  │    → HybridLinearAttnBackend.forward() → forward_extend()
│  │  │    → GDNAttnBackend.forward_extend() ← Prefill / EXTEND
│  │  │    ├─ HybridReqToTokenPool.mamba2_layer_cache(layer_id)
│  │  │    │    → MambaPool.mamba2_layer_cache()：取得本层状态池
│  │  │    │    conv[0]=短卷积历史；temporal=递推状态；cache_indices=各请求状态槽位
│  │  │    │    新请求的零状态 / 命中前缀的状态恢复，已在进入文本主干前准备
│  │  │    ├─ _convolve_prefill() → causal_conv1d_fn()：因果短卷积 + 激活
│  │  │    │    → has_initial_state = extend_prefix_lens > 0
│  │  │    │    → 有前缀则使用卷积历史，处理本轮 token，并更新 conv[0]
│  │  │    ├─ fused_qkv_split_gdn_prefill()：拆分、整理卷积后的 Q/K/V
│  │  │    ├─ _prefill_gates() → fused_gdn_gating()：由 a/b、A_log、dt_bias 计算 g/beta
│  │  │    └─ GDNKernelDispatcher.extend() → TritonGDNKernel.extend()
│  │  │         → chunk_gated_delta_rule() ← 多 token 分块计算
│  │  │         → 按 cache_indices 读取已有 temporal；cu_seqlens 区分各请求
│  │  │         → Q/K L2 归一化 + 门控 → 分块状态更新与注意力输出
│  │  │         → inplace_update=True：原位写回本轮最终 temporal
│  │  │         → 返回每个新 token 的输出；不构造全历史 QK 分数矩阵或做 softmax
│  │  │    按配置 _track_mamba_state_extend()：为前缀复用记录状态快照
│  │  ├─ self.norm(core_attn_out,z) → RMSNormGated.forward()：输出归一化 + z 门控
│  │  │    → reshape：合并各 value head 的结果
│  │  └─ out_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │       → F.linear() / BF16 GEMM：投影回 [N,H]；TP=1，无跨卡求和
│  ├─ attn_boundary.finish() → ffn_boundary.prepare()
│  │    → 残差相加 + post_attention_layernorm（GemmaRMSNorm）
│  ├─ ffn_boundary.exit() → Qwen2MoeMLP.forward()：稠密前馈网络（FFN）
│  │    ├─ gate_up_proj：MergedColumnParallelLinear → ColumnParallelLinear.forward()
│  │    │    → UnquantizedLinearMethod.apply()：投影得到 gate/up
│  │    ├─ SiluAndMul → BaseFusedOp.forward() → SiluAndMul.forward_cuda()
│  │    │    → silu_and_mul()：SiLU(gate) × up
│  │    └─ down_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  └─ ffn_exit.finish()：交给下一层；残差相加可延后到下一次归一化
├─ ③ 所有层完成 → residual_batch.final_norm() → access.final_norm_pair()
│    → self.norm（GemmaRMSNorm）：最终归一化 → 返回 hidden_states=[N,H]
└─ ④ 返回外层调用方 Qwen3VLForConditionalGeneration.forward()
     │ 实际外层对象为 Qwen3_5ForConditionalGeneration，继承此 forward()
     → LogitsProcessor.forward(..., self.lm_head, ...)
     ├─ _get_pruned_states()：每请求取本轮最后一行，[N,H] → [B,H]
     │    last_index = cumsum(extend_seq_lens) - 1
     ├─ _get_logits() → _compute_lm_head() → torch.matmul(hidden_states, lm_head.weight.T)
     │    ParallelLMHead 提供 weight=[V,H]；这里直接用权重，不调用其 forward()
     └─ _copy_logits_to_buffer() → LogitsProcessorOutput(next_token_logits=[B,V]) ← 采样前
```

## Decode × 线性注意力（Gated DeltaNet，GDN）

单卡 CUDA，TP=PP=1，BF16 未量化，Eager，dense 纯文本生成，无输入 logprob。
以 Triton GDN、普通连续状态池为例；图内缓存均在 GPU 显存。实际模型按配置混合两类层。
B=请求数；N=B，每请求本轮输入一个 token；H=hidden_size；V=vocab_size。默认拆分/卷积融合。

```text
Qwen3_5ForCausalLM.forward()：本轮 B 个 token，传入 token ID 或已有 input_embeds
├─ ① Embedding：准备 hidden_states=[B,H]
│  ├─ 已传 input_embeds → 直接使用（外层已完成向量化）
│  └─ 否则 → VocabParallelEmbedding.forward() → _embed_local_shard()
│       → UnquantizedEmbeddingMethod.embedding() → F.embedding()：按 token ID 查权重表
├─ residual_batch.start()：开始本轮残差流
├─ ② 循环 self.layers[layer_idx](...)；此页展开其中的线性注意力层
│  → Qwen3_5LinearDecoderLayer.forward()
│  ├─ attn_boundary.prepare() → input_layernorm：输入归一化
│  │    → GemmaRMSNorm → BaseFusedOp.forward() → GemmaRMSNorm.forward_cuda()
│  │    → _forward_impl() → gemma_rmsnorm() / gemma_fused_add_rmsnorm()（有残差）
│  ├─ self.linear_attn(...) → Qwen3_5GatedDeltaNet.forward()
│  │  ├─ _forward_input_proj()：生成 projected_states_qkvz / projected_states_ba
│  │  │    ├─ BF16、已准备打包权重且 B≤1024 → bf16_gemm_dispatch()：融合两次投影
│  │  │    └─ 其他情况分别调用 in_proj_qkvz / in_proj_ba（MergedColumnParallelLinear）
│  │  │         → ColumnParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │  │         → F.linear() / BF16 GEMM：本质为 x @ weight.T + bias
│  │  ├─ 将 (projected_states_qkvz, projected_states_ba) 元组传给注意力后端
│  │  │    默认 SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=True
│  │  ├─ RadixLinearAttention.forward() → get_attn_backend()
│  │  │    → HybridLinearAttnBackend.forward() → forward_decode()
│  │  │    → GDNAttnBackend.forward_decode() ← 每请求一个新 token
│  │  │    ├─ HybridReqToTokenPool.mamba2_layer_cache(layer_id)
│  │  │    │    → MambaPool.mamba2_layer_cache()：取得本层状态池
│  │  │    │    conv_states=conv[0]；ssm_states=temporal；cache_indices=请求状态槽位
│  │  │    ├─ 满足融合条件 → fused_qkvzba_causal_conv1d_update_contiguous()
│  │  │    │    → 拆出 QKV/z/b/a，读取 conv_states 历史窗口
│  │  │    │    → 因果短卷积 + 激活 → 更新 conv_states，输出 mixed_qkv、z、b、a
│  │  │    │    回退对照：fused_qkvzba_split_reshape_cat_contiguous()
│  │  │    │              → causal_conv1d_update()（与融合操作二选一）
│  │  │    └─ GDNKernelDispatcher.packed_decode() → TritonGDNKernel.packed_decode()
│  │  │         → fused_recurrent_gated_delta_rule_packed_decode() ← 单步递推内核
│  │  │         → 按 cache_indices 读取该请求已有的 temporal
│  │  │         → 内核内拆 Q/K/V、Q/K L2 归一化、计算门控 g/beta
│  │  │         → 用本轮 K/V 更新递推状态，用 Q 计算输出，原位写回 temporal
│  │  │         → packed_decode() 转置后为 [1,B,value头数,value_head_dim]
│  │  │    按配置 _track_mamba_state_decode()：为前缀复用记录状态快照
│  │  │    → 返回 (core_attn_out,z) 给 Qwen3_5GatedDeltaNet.forward()
│  │  ├─ self.norm(core_attn_out,z) → RMSNormGated.forward()：输出归一化 + z 门控
│  │  │    → reshape：合并各 value head 的结果
│  │  └─ out_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  │       → F.linear() / BF16 GEMM：投影回 [B,H]；TP=1，无跨卡求和
│  ├─ attn_boundary.finish() → ffn_boundary.prepare()
│  │    → 残差相加 + post_attention_layernorm（GemmaRMSNorm）
│  ├─ ffn_boundary.exit() → Qwen2MoeMLP.forward()：稠密前馈网络（FFN）
│  │    ├─ gate_up_proj：MergedColumnParallelLinear → ColumnParallelLinear.forward()
│  │    │    → UnquantizedLinearMethod.apply()：投影得到 gate/up
│  │    ├─ SiluAndMul → BaseFusedOp.forward() → SiluAndMul.forward_cuda()
│  │    │    → silu_and_mul()：SiLU(gate) × up
│  │    └─ down_proj：RowParallelLinear.forward() → UnquantizedLinearMethod.apply()
│  └─ ffn_exit.finish()：交给下一层；残差相加可延后到下一次归一化
├─ ③ 所有层完成 → residual_batch.final_norm() → access.final_norm_pair()
│    → self.norm（GemmaRMSNorm）：最终归一化 → 返回 hidden_states=[B,H]
└─ ④ 返回外层调用方 Qwen3VLForConditionalGeneration.forward()
     │ 实际外层对象为 Qwen3_5ForConditionalGeneration，继承此 forward()
     → LogitsProcessor.forward(..., self.lm_head, ...)
     ├─ _get_pruned_states()：保留本轮每请求的一行，仍为 [B,H]
     │    pruned_states = hidden_states（每请求已只有一个本轮 token）
     ├─ _get_logits() → _compute_lm_head() → torch.matmul(hidden_states, lm_head.weight.T)
     │    ParallelLMHead 提供 weight=[V,H]；这里直接用权重，不调用其 forward()
     └─ _copy_logits_to_buffer() → LogitsProcessorOutput(next_token_logits=[B,V]) ← 采样前
```
