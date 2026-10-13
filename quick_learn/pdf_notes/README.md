# 源码学习笔记 PDF

生成文件：[SGLang源码学习笔记.pdf](../../user_guide_zh/SGLang源码学习笔记.pdf)。

黑白、5 个功能组、类与方法连续罗列；调用关系用文本箭头表示。学习注释按完整注释块或说明字符串收录，不按句数或字数截断；公式、调用链、英文续行和模块级注释一并保留。参数与赋值旁的行内注释保留原代码行。

```bash
python quick_learn/pdf_notes/extract_annotations.py
python quick_learn/pdf_notes/build_pdf.py
```

构建依赖 `weasyprint` 与系统中文字体 `Noto Sans CJK SC`。`annotations.json` 保存注释证据，`excerpts.json` 保存完整原文块，`notes.html` 是打印源文件。修改源码后先重新提取；构建时逐条核验注释覆盖、原文来源和分页边界，结果保存在 `layout_check.json`。

## 已读方法代码打印版

[SGLang已读核心方法_打印版.pdf](../../user_guide_zh/SGLang已读核心方法_打印版.pdf) 使用本轮约定的 24 个核心文件范围，收录源码中有学习注释的方法，并合并全核心清单按链路／分支推定已读的方法。保留方法体、装饰器和注释，省略文件与方法内部的导入；局部方法若已经包含在选中的外层方法里，不重复打印。类级学习说明单独保留，不因此展开未读方法。

```bash
python quick_learn/pdf_notes/build_read_methods_pdf.py
```

采用 A4 黑白排版、原始行号、五个功能分组和 PDF 书签。`read_methods_selection.json` 保存已读依据与源码指纹；`read_methods_catalog.json` 保存逐行打印内容；`read_methods_check.json` 保存分页检查结果。`print_comment_corrections.json` 记录仅用于打印版的注释修正，不修改源码；若源码已变，修正清单会先校验再使用。仅调整排版且源码指纹相同时，可加 `--reuse-selection`。

## 已读核心链路精简版

[SGLang已读核心链路_精简打印版.pdf](../../user_guide_zh/SGLang已读核心链路_精简打印版.pdf) 在上述已读打印版的快照上，按 Qwen3.5 dense 文本生成主链路进一步选择方法内部的核心分支。每个方法保留必要输入、关键变量、核心调用和结果，优先保留相关学习注释；特殊模型、硬件、监控及附加输出等分支不展开。所有删除区间明确标记，不能把阅读节选当作独立可运行函数。

开头目录保留五个功能分组，每个类用“类名：方法列表”展示，不展开方法子目录；附正文页码和跳转链接。目录内容由正文生成并保存到 `core_toc.json`。

```bash
python quick_learn/pdf_notes/build_core_excerpts_pdf.py
```

`core_excerpt_plan.json` 保存逐方法保留范围、删除理由及补充注释修正；`core_excerpts_catalog.json` 保存实际节选；`core_excerpts_check.json` 保存分页与覆盖检查。范围计划绑定上一版 catalog 的 SHA-256；更新源码/快照后需重新审阅范围，避免旧行号误用。精简不能删除维持理解所必需的连接，例如 Qwen3.5 文本请求可经过通用多模态入口、CUDA 融合准备的实际分派，以及 CUDA Graph 提前返回与 eager 的边界。
