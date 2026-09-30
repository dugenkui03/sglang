# 模型精度科普：float32、float16 与 bfloat16

> **模型权重用哪种浮点格式存，决定了占多少显存、能表示多大的数、有多准。大模型最常用的是 bfloat16。**

## 1. 浮点数怎么存：符号、指数、尾数

用科学计数法类比：`1.2345 × 10³` 里，`1.2345` 是尾数(Mantissa)，决定**精度**（能记几位有效数字）；`3` 是指数(Exponent)，决定**数量级**（数有多大或多小）。

计算机里的浮点数(Floating Point)是同一个思路，只是底数换成 2：`值 = 符号 × 1.尾数 × 2^指数`。

- **符号位**：1 位，表示正负。
- **指数位**：位数越多，能表示的范围越大。
- **尾数位**：位数越多，精度越高。

```text
float32   [符号 1][指数 8      ][尾数 23                        ]   4 字节
float16   [符号 1][指数 5 ][尾数 10         ]                        2 字节
bfloat16  [符号 1][指数 8      ][尾数 7  ]                           2 字节
```

## 2. 三种格式对比

| 格式 | 字节数 | 符号 / 指数 / 尾数（位） | 最大值 | 十进制有效数字 |
|---|---|---|---|---|
| float32 | 4 | 1 / 8 / 23 | 约 3.4 × 10³⁸ | 约 7 位 |
| float16 | 2 | 1 / 5 / 10 | 65504 | 约 3 位 |
| bfloat16 | 2 | 1 / 8 / 7 | 约 3.4 × 10³⁸ | 约 2–3 位 |

- **bfloat16 和 float32 比**：指数位一样多，范围一样大，不容易溢出；尾数少很多，精度低。相当于把 float32 的尾数截短。
- **bfloat16 和 float16 比**：字节数相同，bfloat16 用精度换范围。
- **精度差别举例**：`1.01` 存成 bfloat16 变成 `1.0078125`，存成 float16 是 `1.0097656`，float32 约为 `1.0099999`。

## 3. 什么时候用 float16，什么时候用 bfloat16

- **基本原则：跟模型训练时用的格式走**，也就是 `config.json` 里的 `dtype`。
- **用 bfloat16**：
  - 模型本身是 bfloat16 训练、发布的，现在主流大模型基本都是（Qwen、Llama 3、DeepSeek 等）。
  - 硬件支持：NVIDIA Ampere 及以后（A100、A10、H100、RTX 30/40 系列等）、AMD MI200/MI300、TPU。
- **用 float16**：
  - 硬件不支持 bfloat16，比如 V100、T4 这类老卡。
  - 模型本来就以 float16 发布，多见于较早的模型。
  - 数值范围确定不大、又想多一点精度（float16 尾数 10 位，bfloat16 只有 7 位）。
- **跨格式转换的风险**：bfloat16 模型硬转成 float16，中间结果一旦超过 65504 就变成 inf / NaN，输出直接乱掉；float16 模型转成 bfloat16 只是精度略降，相对安全。

## 4. SGLang 怎么选：`dtype="auto"`

启动参数 `--dtype` 传到 `ModelConfig`，由 [`_get_and_verify_dtype`](model_config.py#L1823) 决定最终精度，结果存在 [`self.dtype`](model_config.py#L602)。

- **读哪个配置**：读语言模型那一层（`hf_text_config`，多模态模型是 `text_config`）的 `dtype`，旧字段名是 `torch_dtype`；没写就当作 float32。
- **`auto`**：配置写 bfloat16 就用 bfloat16，写 float16 就用 float16；只有配置是 float32 时才降精度，默认降为 float16，Gemma 系列降为 bfloat16。**不看硬件是否支持。**
- **显式指定**（如 `--dtype bfloat16`）：直接使用；float16 和 bfloat16 之间可以互转，float32 可以降精度，也可以升到 float32。

## 5. 例子：Qwen3.8-27B

- 它的 `config.json` 里 `text_config` 写的是 `"dtype": "bfloat16"`，所以 `auto` 选 bfloat16。
- 27B 个参数 × 2 字节 ≈ 54 GB，和仓库 55.6 GB 的总大小基本一致；差出来的部分是视觉编码器等其他权重，以及单位换算。

## 术语与生词

| 术语或单词 | 中文释义 | 简明英文释义 |
|---|---|---|
| Floating Point /ˈfloʊtɪŋ pɔɪnt/ | 浮点数：用符号、指数、尾数表示实数 | A number format using sign, exponent and mantissa. |
| Exponent /ɪkˈspoʊnənt/ | 指数：决定数的数量级和表示范围 | The part that sets a number's magnitude. |
| Mantissa /mænˈtɪsə/ | 尾数：决定有效数字和精度，也叫 significand | The part that holds the significant digits. |
| Precision /prɪˈsɪʒn/ | 精度：能表示的数有多准 | How exactly a value can be represented. |
| Overflow /ˌoʊvərˈfloʊ/ | 溢出：数超过格式能表示的最大值 | A value exceeding the largest representable number. |
| bfloat16（Brain Floating Point 16） | Google Brain 提出的 16 位浮点格式，范围同 float32 | A 16-bit float with float32's range and less precision. |
