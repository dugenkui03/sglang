# todo：普通 Decode 中的请求槽位与 KV 页表

用两个请求串起请求槽位(request slot)、KV 槽位(KV slot)和页表(page table)。只讨论普通解码(decode)的即时执行(Eager)路径、全注意力(full attention)层与普通分页缓存池。

## 1. 示例输入：每个 Tensor 具体保存什么

本轮请求 A、B 各处理一个输入 token，ID 为 101、202（上一轮采样结果）；已有 KV 长度分别为 2、1。
请求槽位此前已在预填充(prefill)时分配，decode 继续使用。`Req.req_pool_idx` 各自为 5、9；`ScheduleBatch.req_pool_indices` 汇总为一维张量(Tensor)，随后传给 `ForwardBatch`。

| 变量 | shape / dtype | 本轮准备前的值与含义 |
|---|---|---|
| `input_ids` | `[2]` / int64 | `tensor([101, 202])`：本轮输入 token ID |
| `req_pool_indices` | `[2]` / int64 | `tensor([5, 9])`：本批两个请求的 req slot |
| `seq_lens` | `[2]` / int64 | `tensor([2, 1])`：两个请求已有 KV 的 token 数 |
| `req_to_token` | `[10, 4]` / int32 | 下面完整展开；行是 req slot，列是请求内 token 位置 |
| `page_size` | 整数，无 shape | `2`：每页包含两个 KV 槽位，不是 Tensor |

以下完整构造输入；演示池包含 10 行（含保留行 0），每行最多记录 4 个 token。其他请求行暂未使用，均为 0。

```python
import torch
device = "cuda"  # 仅验证索引示例时可改为 "cpu"
page_size = 2
input_ids = torch.tensor([101, 202], dtype=torch.int64, device=device)
req_pool_indices = torch.tensor([5, 9], dtype=torch.int64, device=device)
seq_lens = torch.tensor([2, 1], dtype=torch.int64, device=device)
req_to_token = torch.tensor([
    [0, 0, 0, 0],  # 行 0：保留行
    [0, 0, 0, 0],  # 行 1
    [0, 0, 0, 0],  # 行 2
    [0, 0, 0, 0],  # 行 3
    [0, 0, 0, 0],  # 行 4
    [6, 7, 0, 0],  # 行 5：请求 A，已有 KV 槽位 [6, 7]
    [0, 0, 0, 0],  # 行 6
    [0, 0, 0, 0],  # 行 7
    [0, 0, 0, 0],  # 行 8
    [8, 0, 0, 0],  # 行 9：请求 B，已有 KV 槽位 [8]
], dtype=torch.int32, device=device)  # [请求行容量=10, token位置容量=4]
```

例如 `req_to_token[5, 1] = 7`：请求 A 的位置 1（第二个 token）使用 KV 槽位 7。二维表里存的是编号，不是 K/V 向量。

## 2. 本轮 Decode：增加 KV 槽位、更新长度

```text
ScheduleBatch.prepare_for_decode()
    │
    ├─ ① alloc_for_decode()：假设本轮 out_cache_loc = tensor([2, 9])
    │    → req_to_token_pool.write()：按旧 seq_lens=[2,1] 确定写入列
    │         ├─ req_to_token[5, 2] = 2：A 的有效部分变成 [6, 7, 2]
    │         └─ req_to_token[9, 1] = 9：B 的有效部分变成 [8, 9]
    │
    └─ ② self.seq_lens = self.seq_lens + 1：tensor([2, 1]) → tensor([3, 2])
```

A 的页 3 已满，假设新分配页 1，从槽位 2 开始；B 的页 4 尚有空间，续写槽位 9。
下面用实际 Tensor 操作复现结果，`out_cache_loc` 是 KV 槽位编号，与请求编号 `[5,9]` 属于不同编号体系。

```python
out_cache_loc = torch.tensor([2, 9], dtype=torch.int64, device=device)  # [2]
req_to_token[req_pool_indices, seq_lens] = out_cache_loc.to(torch.int32)
seq_lens = seq_lens + 1  # tensor([3, 2])，shape=[2]，dtype=int64
# 更新后 req_to_token 的第 5、9 行如下，其他八行仍为 [0, 0, 0, 0]：
# tensor([[6, 7, 2, 0],
#         [8, 9, 0, 0]], dtype=torch.int32)  # shape=[2, 4]
```

这一步只登记索引并更新长度，本轮 K/V 数值稍后在各层前向中产生并写入。无效位置用 0 示意，实际可能留有旧值；有效长度由 `seq_lens` 决定。
随后 `TpModelWorker.forward_batch_generation() → ForwardBatch.init_new()` 传递请求编号与更新后的长度。

## 3. 前向准备：逐步生成页表

```text
EagerRunner._execute_decode()
→ HybridLinearAttnBackend.init_forward_metadata()：Qwen3.5 等混合模型的包装层
→ FlashAttentionBackend.init_forward_metadata()
    ├─ ① 选取请求行、抽取每页首槽位、整除页大小 → metadata.page_table
    ├─ ② seq_lens.to(torch.int32) → metadata.cache_seqlens_int32
    └─ ③ self.forward_metadata = metadata：保存本轮元数据
```

下面对应同一构造过程；本例最大有效长度为 3，`token_slots`、`page_starts` 是为展示中间结果起的示例变量名。

```python
token_slots = req_to_token[req_pool_indices, :3]  # shape=[2, 3]
# tensor([[6, 7, 2],
#         [8, 9, 0]], dtype=torch.int32)
page_starts = token_slots[:, ::page_size]         # shape=[2, 2]
# tensor([[6, 2],
#         [8, 0]], dtype=torch.int32)
page_table = page_starts // page_size            # shape=[2, 2]
# tensor([[3, 1],
#         [4, 0]], dtype=torch.int32)
cache_seqlens_int32 = seq_lens.to(torch.int32)    # shape=[2]
# tensor([3, 2], dtype=torch.int32)
```

页表每行按本批请求顺序排列：行 0 对应 req slot 5，行 1 对应 req slot 9。实际表可能更宽，无效部分不参与读取。上方 Tensor 输出省略 `device` 标注，实际仍在 GPU 上。
为什么每隔两个槽位取一个就够了？分页分配器保证按请求内页边界分组后，同组 token 使用同一缓存页。以请求 A 为例：

| 请求内逻辑页 | token 位置（从 0 开始） | 有效 KV 槽位 | 取出的首槽位 | 缓存页号 |
|---|---|---|---:|---|
| 第 0 页 | 0、1 | `[6, 7]` | 6 | `6 // 2 = 3` |
| 第 1 页 | 2（目前仅一个有效 token） | `[2]` | 2 | `2 // 2 = 1` |

任意相邻两个 token 不一定同页，例如槽位 7、2 就跨页；必须从 token 位置 0 开始按页分组。

## 4. 注意力内核实际读取哪些槽位

`FlashAttentionBackend.forward_decode()` 先写入本轮 K/V，再将页表与有效长度传给注意力内核(kernel)。
定位公式：`kv_index = page_table[b, t // page_size] * page_size + t % page_size`，`b` 是本批请求下标，`t` 是请求内 token 位置，均从 0 开始。

| 本批下标 | req slot | page_table 对应行 | cache_seqlens | 实际读取的 KV 槽位 |
|---|---:|---|---:|---|
| 0（A） | 5 | `[3, 1]` | 3 | `[6, 7, 2]` |
| 1（B） | 9 | `[4, 0]` | 2 | `[8, 9]`，页号 0 被忽略 |

例如 A 的位置 2：`page_table[0, 1] * 2 + 0 = 1 * 2 = 2`；B 只需页 4 的两个槽位。选定全注意力层后，用这些编号定位该层 K/V；本轮新计算两个 token，注意力可使用的长度分别为 3、2。

源码：[schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py)、[allocation.py](../../python/sglang/srt/mem_cache/allocation.py)、[flashattention_backend.py](../../python/sglang/srt/layers/attention/flashattention_backend.py)。
