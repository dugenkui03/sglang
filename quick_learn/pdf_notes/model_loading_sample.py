#!/usr/bin/env python3
"""Build a small hierarchy sample; leave the complete notes PDF untouched."""

import ast
import html
import json
from pathlib import Path

from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "user_guide_zh/SGLang源码学习笔记_模型加载格式样例.pdf"

CSS = """
@page { size:A4; margin:16mm 16mm 17mm;
  @bottom-left { content:'模型加载 · 格式样例'; font-family:'Noto Sans CJK SC'; font-size:7pt; color:#666; }
  @bottom-right { content:counter(page); font-family:'Noto Sans CJK SC'; font-size:8pt; color:#555; }
}
body { margin:0; color:#111; font-family:'Noto Sans CJK SC',sans-serif; font-size:9.4pt; line-height:1.6; }
h1 { font-size:16pt; margin:0 0 4mm; padding-bottom:2mm; border-bottom:1px solid #222; bookmark-level:1; }
h2 { font-size:13pt; margin:4mm 0 3mm; bookmark-level:2; }
h3 { font-size:11.5pt; margin:5mm 0 2.5mm; bookmark-level:3; }
h4 { font-size:10.2pt; font-weight:700; margin:4mm 0 2mm; bookmark-level:4; }
h1,h2,h3,h4 { break-after:avoid; }
p { margin:0 0 2.5mm; orphans:2; widows:2; }
.meta { font-size:8pt; color:#555; }
.class-info { break-inside:avoid; break-after:avoid; }
.method { break-inside:avoid; margin-bottom:4mm; }
.method p { margin-left:3mm; }
.label { font-weight:700; }
pre { font-family:'DejaVu Sans Mono','Noto Sans CJK SC'; font-size:8.1pt; line-height:1.55; white-space:pre-wrap; overflow-wrap:anywhere; margin:2mm 0 3mm 3mm; }
.chain { break-inside:avoid; }
.inheritance { font-size:8.2pt; margin:1mm 0 2mm 3mm; }
.methods { margin:2mm 0; }
.code { break-inside:avoid; padding-left:3mm; border-left:1px solid #aaa; }
.source { font-family:'DejaVu Sans Mono','Noto Sans CJK SC'; font-size:7pt; color:#666; overflow-wrap:anywhere; }
"""

CHAIN = """ModelRunner.__init__()
  → ModelRunner.initialize()
  → ModelRunner.load_model()
  → load_model_with_memory_saver()
  → DefaultModelLoader.load_model()
      │
      ├─ ① _initialize_model()：创建模型和参数
      │   → Qwen3_5ForConditionalGeneration.__init__()
      │   → Qwen3VLForConditionalGeneration.__init__()
      │       → Qwen3_5ForCausalLM.__init__()
      │           → VocabParallelEmbedding.__init__()
      │               → UnquantizedEmbeddingMethod.create_weights()
      │                   ├─ 创建 weight，并注册到 embedding 层
      │                   └─ 将 weight_loader 保存到 weight 上
      │
      └─ ② load_weights_and_postprocess()：加载实际权重
          → Qwen3_5ForConditionalGeneration.load_weights()
              ├─ 根据参数名找到之前创建的 weight
              ├─ 取出 weight.weight_loader 并执行
              └─ VocabParallelEmbedding.weight_loader()
                  → 把本卡负责的词表权重拷贝进 weight"""

LOAD_MODEL = '''def load_model(self, *, model_config, device_config):
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)

    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            # NOTE 创建模型相关对象与参数空间，尚未加载实际权重。
            model = _initialize_model(
                model_config, self.load_config, quant_config
            )

        # NOTE 加载实际权重，并执行加载后的处理。
        self.load_weights_and_postprocess(
            model,
            # 返回权重生成器；遍历时读取权重，按需准备或下载文件。
            self._get_all_weights(model_config, model),
            target_device,
        )

    return model.eval()'''

LOAD_WEIGHTS = '''@staticmethod
def load_weights_and_postprocess(model, weights, target_device):
    # NOTE 调用模型自身的加载方法，填充先前创建的参数。
    model.load_weights(weights)

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            # 执行加载后处理，例如重排或量化；具体行为由实现决定。
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)'''

CREATE_WEIGHTS = '''def create_weights(
    self, layer, input_size_per_partition, output_partition_sizes,
    input_size, output_size, params_dtype, **extra_weight_attrs,
):
    # NOTE 为本卡 embedding 权重分配未初始化的参数空间。
    weight = Parameter(
        torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

    # NOTE 创建 weight 参数，并注册到 embedding 层。
    layer.register_parameter("weight", weight)

    # NOTE 把 embedding 层的 weight_loader 保存到 weight 参数对象上。
    set_weight_attrs(weight, extra_weight_attrs)'''


def code(text, css_class="code"):
    return f'<pre class="{css_class}">{html.escape(text)}</pre>'


def method(number, name, description, snippet, source):
    return (f'<div class="method"><h4>{number}　{name}</h4>'
            f'<p><span class="label">方法说明：</span>{description}</p>'
            '<p class="label">带注释的核心代码：</p>' + code(snippet)
            + f'<p class="source">{source}</p></div>')


def main():
    for snippet in (LOAD_MODEL, LOAD_WEIGHTS, CREATE_WEIGHTS):
        ast.parse(snippet)
    body = '<h1>1. 启动与模型加载</h1><h2>1.1 模型加载链路</h2>'
    body += '<p class="meta">格式样例：先展示完整主链路，再选两个类、三个方法展开。采用默认加载器、普通加载路径；代码为核心节选，说明与注释按当前实现校正。</p>'
    body += code(CHAIN, 'chain')
    body += '<div class="class-info"><h3>1.1.1 DefaultModelLoader</h3>'
    body += '<p><b>作用：</b>默认模型加载器，支持从不同类型的权重文件加载模型；先创建模型和参数，再加载实际权重。</p>'
    body += '<p><b>继承结构：</b></p>'
    body += code('DefaultModelLoader\n  → BaseModelLoader\n      → ABC\n          → object', 'inheritance')
    body += '<p class="methods"><b>本节方法：</b><b>load_model()</b>、<b>load_weights_and_postprocess()</b>。</p></div>'
    body += method('1.1.1.1', 'load_model()',
                   '加载模型权重的核心入口。先调用 _initialize_model() 创建模型，再调用 load_weights_and_postprocess() 加载实际权重。',
                   LOAD_MODEL, 'srt/model_loader/loader.py:962')
    body += method('1.1.1.2', 'load_weights_and_postprocess()',
                   '调用模型的 load_weights() 加载权重，然后遍历模块，执行各计算实现需要的加载后处理。',
                   LOAD_WEIGHTS, 'srt/model_loader/loader.py:1014')
    body += '<div class="class-info"><h3>1.1.2 UnquantizedEmbeddingMethod</h3>'
    body += '<p><b>作用：</b>Embedding 的未量化计算实现，负责创建权重参数，并通过 embedding() 按 token ID 查表。</p>'
    body += '<p><b>继承结构：</b></p>'
    body += code('UnquantizedEmbeddingMethod\n  → QuantizeMethodBase\n      → ABC\n          → object', 'inheritance')
    body += '<p class="methods"><b>本节方法：</b><b>create_weights()</b>。</p></div>'
    body += method('1.1.2.1', 'create_weights()',
                   '给 VocabParallelEmbedding 创建 weight 参数并注册到层上；把 weight_loader 等加载属性保存到该参数上，供后续填入实际权重。',
                   CREATE_WEIGHTS, 'srt/layers/quantization/unquant.py:184')
    body += '<p class="meta">这里 extra_weight_attrs 中的 weight_loader 来自 VocabParallelEmbedding.__init__() 传入的 weight_loader=self.weight_loader。torch.empty() 分配空间，但不保证初始数值为零。</p>'
    document = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>模型加载 · 格式样例</title><style>{CSS}</style></head><body>{body}</body></html>'
    (HERE / 'model_loading_sample.html').write_text(document)
    rendered = HTML(string=document, base_url=str(HERE)).render()
    overflow = []
    for i, page in enumerate(rendered.pages):
        for b in page._page_box.descendants():
            if type(b).__name__ == 'TextBox' and (
                b.position_x < -1 or b.position_x+b.width > page.width+1
                or b.position_y < -1 or b.position_y+b.height > page.height+1
            ):
                overflow.append({'page':i+1, 'text':b.text})
    if overflow:
        raise ValueError(overflow)
    rendered.write_pdf(OUTPUT, pdf_identifier=True, srgb=True)
    print(json.dumps({'pdf':str(OUTPUT),'pages':len(rendered.pages),'overflow':overflow},ensure_ascii=False))


if __name__ == '__main__':
    main()
