# 增量解码：surr_ids 与 read_ids

[`_decode_batch_token_id_output`](detokenizer_manager.py#L360) 每轮为每个请求准备两组 token ID，解码后相减得到新文本。

- `surr_ids`（surr 即 surrounding）：上一轮已输出的那段 token，只当上下文；首次是输入末尾最多 5 个 token。
- `read_ids`：`surr_ids` 加本次新 token；请求因结束符停止时去掉末尾的 `<eos>`。
- 新文本 = `decode(read_ids)` 去掉开头的 `decode(surr_ids)`；输出后窗口后移：`surr_offset ← read_offset`，`read_offset ← len(decode_ids)`。
- 为什么要相减：单独解码新 token 可能出错。例（Llama 等 SentencePiece 分词器）：上一轮已输出 `"Hello"`，新 token 是 `▁world`（`▁` 表示前面有空格）
  - 只解码新 token：`decode([▁world])` 得到 `"world"`，开头空格被去掉，拼起来成了 `"Helloworld"` ✗
  - 带上下文再相减：`decode([▁Hello, ▁world])` 得到 `"Hello world"`，减去 `decode([▁Hello])` 的 `"Hello"`，得到 `" world"` ✓；两次解码开头的副作用相同，相减正好抵消
- 一个字可能拆成两个 token，只解码前一个会得到 `"�"`：此时偏移不推进，等下一个 token 到了一起解码。

示例（窗口怎么移动）：输入切成 `[数, 三个, 数字]`，依次生成 `1`、`2`、`3`、`<eos>`：

| 轮次 | surr_ids → 文本 | read_ids → 文本 | 新文本 |
|---|---|---|---|
| 1 | `[数, 三个, 数字]` → `数三个数字` | `[数, 三个, 数字, 1]` → `数三个数字1` | `1` |
| 2 | `[1]` → `1` | `[1, 2]` → `12` | `2` |
| 3 | `[2]` → `2` | `[2, 3]` → `23` | `3` |
| 4 | `[3]` → `3` | `[3]`（去掉 `<eos>`）→ `3` | 空，请求结束 |
