#!/usr/bin/env python3
"""Render the reviewed Qwen3.5 class relationships as printable UML diagrams."""

import html
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jsonschema
import pymupdf
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
ART = HERE / "qwen35_uml"
ART.mkdir(exist_ok=True)
RUNTIME = Path("/tmp/sglang-reflections-print-sPY6xDvc")
SKILL = Path.home() / ".codex/plugins/cache/openai-curated-remote/mermaid-diagrams/2.1.0/skills/mermaid-diagrams"
CLI = RUNTIME / "mermaid/node_modules/.bin/mmdc"
ENV = dict(os.environ, PATH=str(RUNTIME / "venv/bin") + ":" + os.environ["PATH"])
ENV["LD_LIBRARY_PATH"] = str(RUNTIME / "libs/runtime/usr/lib/x86_64-linux-gnu")
OUTPUT = HERE.parents[1] / "user_guide_zh/Qwen3.5核心类关系_UML.pdf"
ALIASES = {"nn.Module": "Module", "abc.ABC": "ABC", "RMSNorm": "RMSNormGated"}
LABELS = {"Module": "nn.Module", "ABC": "abc.ABC", "RMSNormGated": "RMSNorm (RMSNormGated)"}
roles, figures = {}, []


def figure(key, title, edges, nodes=(), note="", direction="LR", category="组合与引用"):
    names = list(dict.fromkeys([*nodes, *(n for e in edges for n in e[:2])]))
    source = ["classDiagram", "direction " + direction]
    for name in names:
        source.append(f'class {name}["{LABELS.get(name, name)}"]')
    for start, end, kind, label in edges:
        source.append(f"{start} {kind} {end}" + (" : " + label if label else ""))
    figures.append(dict(id=key, title=title, kind="raw", source="\n".join(source) + "\n",
                        description=note or None, accessibility=category))


def read_inheritance():
    text = (HERE / "qwen35_core_inheritance.txt").read_text().replace("===PAGE===", "")
    for index, block in enumerate(re.split(r"(?m)^## ", text)[1:], 1):
        heading, body = block.split("\n", 1)
        nodes, edges, parents = [], [], {}
        for line in body.splitlines():
            stripped = line.lstrip(" │├└─")
            if stripped.startswith("nn.Module + abc.ABC"):
                parents = {-1: ["Module", "ABC"]}
                nodes += ["Module", "ABC"]
                continue
            match = re.match(r"([A-Za-z_][\w.]*)(?:[：:]|（|$)", stripped)
            if not match or (not any(c in line for c in "├└") and match[1] not in {"nn.Module", "abc.ABC", "object"}):
                continue
            name = ALIASES.get(match[1], match[1])
            nodes.append(name)
            if "：" in stripped:
                roles[name] = stripped.split("：", 1)[1]
            branch = re.search("[├└]", line)
            if branch is None:
                parents = {-1: [name]}
            else:
                depth = branch.start() // 4
                for parent in parents[depth - 1]:
                    edges.append((parent, name, "<|--", ""))
                parents[depth] = [name]
        if index == 1:
            subsets = [
                ("1.1 模型外层的继承", ["Module", "Qwen3VLForConditionalGeneration", "Qwen3_5ForConditionalGeneration"]),
                ("1.2 文本模型与 Decoder 层的继承", ["Module", "Qwen3_5ForCausalLM", "Qwen3_5AttentionDecoderLayer", "Qwen3_5LinearDecoderLayer", "Qwen3_5GatedDeltaNet", "Qwen2MoeMLP", "ModuleList"]),
                ("1.3 注意力入口与输出模块的继承", ["Module", "RadixAttention", "RadixLinearAttention", "LogitsProcessor", "Sampler"]),
            ]
            for part, (title, selected) in enumerate(subsets, 1):
                figure(f"inherit_1_{part}", title, [e for e in edges if e[0] in selected and e[1] in selected],
                       selected, direction="TB" if part == 1 else "LR", category="继承")
        else:
            figure(f"inherit_{index}", heading, edges, nodes,
                   direction="TB" if index in {2, 3, 4} else "LR", category="继承")
    roles["object"] = "Python 类的基础类。"
    roles["RMSNormGated"] = "来自 fla/layernorm_gated.py；归一化并应用 z 门控。"


def composition(key, title, text, note="", direction="LR"):
    edges = []
    for line in text.strip().splitlines():
        match = re.fullmatch(r"(\w+)\s+(\*--|o--|-->|\.\.>)\s+(\w+)(?:\s*:\s*(.*))?", line.strip())
        assert match, line
        edges.append((match[1], match[3], match[2], match[4] or ""))
    figure(key, title, edges, note=note, direction=direction)


def build_sources():
    read_inheritance()
    composition("worker", "9.1 模型工作进程与 ModelRunner", """
TpModelWorker *-- ModelRunner : _model_runner
""", "TpModelWorker 通过 model_runner 属性访问 _model_runner。")
    composition("runner", "9.2 ModelRunner 持有的对象", """
ModelRunner *-- EagerRunner : eager_runner
ModelRunner *-- Sampler : sampler
ModelRunner *-- Qwen3_5ForConditionalGeneration : model
ModelRunner o-- HybridLinearAttnBackend : attn_backend
ModelRunner o-- HybridReqToTokenPool : req_to_token_pool
ModelRunner o-- HybridLinearKVPool : token_to_kv_pool
""", "EagerRunner.model_runner 引回同一个 ModelRunner。空心菱形表示它持有的共享对象。")
    composition("model", "10. Qwen3.5 模型内部的组合", """
Qwen3_5ForConditionalGeneration *-- Qwen3_5ForCausalLM : model
Qwen3_5ForConditionalGeneration *-- ParallelLMHead : lm_head
Qwen3_5ForConditionalGeneration *-- LogitsProcessor : logits_processor
Qwen3_5ForCausalLM *-- VocabParallelEmbedding : embed_tokens
Qwen3_5ForCausalLM *-- ModuleList : layers
Qwen3_5ForCausalLM *-- GemmaRMSNorm : norm
""", "ModuleList 按层配置保存全注意力层或线性注意力层；这里列出纯文本生成使用的成员。lm_head 使用独立的词表输出权重。", direction="TB")
    composition("attention", "11. 全注意力层的组合", """
Qwen3_5AttentionDecoderLayer *-- QKVParallelLinear : qkv_proj
Qwen3_5AttentionDecoderLayer *-- GemmaRMSNorm : 归一化模块
Qwen3_5AttentionDecoderLayer o-- RotaryEmbedding : rotary_emb
Qwen3_5AttentionDecoderLayer *-- RadixAttention : attn
Qwen3_5AttentionDecoderLayer *-- RowParallelLinear : o_proj
Qwen3_5AttentionDecoderLayer *-- StageBoundary : 阶段边界
Qwen3_5AttentionDecoderLayer *-- Qwen2MoeMLP : mlp
""", "归一化成员：q_norm、k_norm、input_layernorm、post_attention_layernorm。阶段边界：attn_boundary、ffn_boundary。rotary_emb 由 get_rope 按配置选择，也可以是 MRotaryEmbedding。")
    composition("linear_layer", "12. 线性注意力层的组合", """
Qwen3_5LinearDecoderLayer *-- Qwen3_5GatedDeltaNet : linear_attn
Qwen3_5LinearDecoderLayer *-- GemmaRMSNorm : 归一化模块
Qwen3_5LinearDecoderLayer *-- StageBoundary : 阶段边界
Qwen3_5LinearDecoderLayer *-- Qwen2MoeMLP : mlp
""", "归一化成员：input_layernorm、post_attention_layernorm。阶段边界：attn_boundary、ffn_boundary。")
    composition("gdn", "13. Gated DeltaNet 内部的组合", """
Qwen3_5GatedDeltaNet *-- MergedColumnParallelLinear : 输入投影
Qwen3_5GatedDeltaNet *-- ColumnParallelLinear : conv1d
Qwen3_5GatedDeltaNet *-- RadixLinearAttention : attn
Qwen3_5GatedDeltaNet *-- RMSNormGated : norm
Qwen3_5GatedDeltaNet *-- RowParallelLinear : out_proj
""", "输入投影成员：in_proj_qkvz、in_proj_ba。conv1d 提供短卷积权重；RadixLinearAttention 引用该权重及 A_log、dt_bias 门控参数。")
    composition("mlp", "14. FFN 与普通线性计算策略", """
Qwen2MoeMLP *-- MergedColumnParallelLinear : gate_up_proj
Qwen2MoeMLP *-- SiluAndMul : act_fn
Qwen2MoeMLP *-- RowParallelLinear : down_proj
LinearBase *-- UnquantizedLinearMethod : quant_method
VocabParallelEmbedding *-- UnquantizedEmbeddingMethod : quant_method
""", "未量化配置下，线性层通过 quant_method.apply(layer, x, bias) 使用层上的 weight；Embedding 通过 embedding() 查权重表。")
    composition("backend", "15. 混合注意力后端与内核对象", """
HybridLinearAttnBackend o-- FlashAttentionBackend : full_attn_backend
HybridLinearAttnBackend o-- GDNAttnBackend : linear_attn_backend
GDNAttnBackend *-- GDNKernelDispatcher : kernel_dispatcher
GDNKernelDispatcher *-- TritonGDNKernel : extend_kernel / decode_kernel
""", "本例的 Prefill 和 Decode 都使用 Triton，两个成员引用同一个 TritonGDNKernel。", direction="TB")
    composition("context", "16. 当前注意力后端的引用", """
ForwardContext --> HybridLinearAttnBackend : attn_backend
RadixAttention ..> ForwardContext : get_attn_backend()
RadixLinearAttention ..> ForwardContext : get_attn_backend()
ModelRunner ..> ForwardContext : 创建并发布
""", "ModelRunner._forward_raw() 为本轮设置 ForwardContext。Radix 层读取上下文中的后端，再调用其 forward；后端是共享对象。", direction="TB")
    composition("pool", "17.1 缓存池与共享状态池", """
ModelRunner o-- HybridReqToTokenPool : req_to_token_pool
ModelRunner o-- HybridLinearKVPool : token_to_kv_pool
HybridReqToTokenPool *-- MambaPool : mamba_pool
HybridLinearKVPool o-- MambaPool : mamba_pool
HybridLinearKVPool *-- MHATokenToKVPool : full_kv_pool
""", "两个混合缓存池引用同一 MambaPool。MHATokenToKVPool.k_buffer/v_buffer 保存全注意力 K/V；MambaPool.mamba_cache.conv/temporal 保存卷积和递推状态，均位于 GPU 显存(GPU memory)。", direction="TB")
    composition("pool_refs", "17.2 注意力后端共享缓存池", """
FlashAttentionBackend --> HybridReqToTokenPool : 共享引用
FlashAttentionBackend --> HybridLinearKVPool : 共享引用
GDNAttnBackend --> HybridReqToTokenPool : 共享引用
GDNAttnBackend --> HybridLinearKVPool : 共享引用
""", "对应成员是 req_to_token_pool 和 token_to_kv_pool；两个后端引用 ModelRunner 持有的同一组缓存池。")
    composition("batch", "18.1 批次与残差流", """
ForwardBatch *-- ResidualStream : residual_stream
StageBoundary ..> ResidualStream : prepare / finish
StageBoundary ..> FfnExit : exit() 创建
FfnExit --> ResidualStream : _stream
""", "ForwardBatch 保存 input_ids、positions、req_pool_indices、seq_lens、out_cache_loc。ResidualStream 保存 residual 和 pending。", direction="TB")
    composition("output", "18.2 结果对象与采样器", """
ModelRunnerOutput *-- LogitsProcessorOutput : logits_output
ModelRunner ..> Sampler : sample() 调用
Sampler ..> LogitsProcessorOutput : 读取 logits
""", "LogitsProcessorOutput.next_token_logits 是词表分数；Sampler 从分数或概率中选择下一个 token。", direction="TB")
    spec = dict(version=2, title="Qwen3.5 核心类 UML 关系", diagrams=figures)
    schema = json.loads((SKILL / "references/canonical-v2.schema.json").read_text())
    jsonschema.validate(spec, schema)
    (ART / "canonical.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    subprocess.run([sys.executable, str(SKILL / "scripts/mermaid_tool.py"), "create",
                    str(ART / "canonical.json"), str(ART / "diagrams.md"), "--overwrite"], check=True)
    config = dict(theme="base", look="classic", fontFamily="Noto Sans CJK SC", htmlLabels=False,
                  themeVariables=dict(fontFamily="Noto Sans CJK SC", fontSize="16px",
                    primaryColor="#ffffff", primaryTextColor="#111111", primaryBorderColor="#111111",
                    secondaryColor="#ffffff", tertiaryColor="#ffffff", lineColor="#111111"))
    config["class"] = dict(htmlLabels=False, hideEmptyMembersBox=True, padding=8,
                           nodeSpacing=20, rankSpacing=28, diagramPadding=8)
    (ART / "mermaid.json").write_text(json.dumps(config, ensure_ascii=False))


def main():
    build_sources()
    command = [str(CLI), "-i", str(ART / "diagrams.md"), "-o", str(ART / "rendered.svg"),
               "-c", str(ART / "mermaid.json"), "-p", str(RUNTIME / "puppeteer.json"),
               "--no-font-embed", "-j", "3"]
    subprocess.run(command, env=ENV, check=True)
    dimensions, sections, all_names, described = [], [], set(), set()
    for index, figure in enumerate(figures, 1):
        svg = ART / f"rendered-{index}.svg"
        root = ET.parse(svg).getroot()
        _, _, width, height = map(float, root.attrib["viewBox"].split())
        scale = min(0.75, 544 / width)
        font_pt = 16 * scale
        assert font_pt >= 11, f'{figure["id"]}: split this diagram to preserve readable type ({font_pt:.1f} pt)'
        dimensions.append(dict(id=figure["id"], width=width, height=height, font_pt=font_pt))
        names = re.findall(r'^class (\w+)\[', figure['source'], re.M)
        all_names.update(names)
        cards = ''
        if figure['accessibility'] == '继承':
            for name in names:
                if name in described:
                    continue
                cards += ('<div class="card"><b>' + html.escape(LABELS.get(name, name))
                          + '</b><div>' + html.escape(roles.get(name, '')) + '</div></div>')
                described.add(name)
        note = '<p class="note">' + html.escape(figure['description']) + '</p>' if figure['description'] else ''
        sections.append('<section><h2>' + html.escape(figure['title']) + '</h2>'
                        + f'<img src="{svg.as_uri()}" style="width:{width*scale:.3f}pt;height:{height*scale:.3f}pt">'
                        + note + ('<div class="roles">' + cards + '</div>' if cards else '') + '</section>')
    (ART / "layout.json").write_text(json.dumps(dimensions, indent=2))
    css = '''@page { size:A4 portrait; margin:9mm; }
body { margin:0; color:#111; font-family:"Noto Sans CJK SC",sans-serif; font-size:11pt; }
section { break-inside:avoid; margin:0 0 6mm; }
h2 { font-size:14pt; line-height:1.4; margin:0 0 3mm; font-weight:700; }
img { display:block; margin:0 auto 3mm; }
.legend { font-size:10.5pt; line-height:1.5; margin:0 0 5mm; }
.roles { display:grid; grid-template-columns:1fr 1fr; gap:2mm 5mm; }
.card { break-inside:avoid; font-size:11pt; line-height:1.45; }
.card b { display:block; font-weight:500; overflow-wrap:anywhere; }
.note { font-size:11pt; line-height:1.6; margin:2mm 0 0; }
'''
    legend = 'UML 图例：空心三角指向父类；实心菱形表示组合；空心菱形表示聚合或共享；实线箭头表示引用，虚线箭头表示使用。'
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Qwen3.5 核心类 UML</title>'
                '<style>' + css + '</style><body><p class="legend">' + legend + '</p>'
                + ''.join(sections) + '</body></html>')
    (ART / 'print.html').write_text(document)
    HTML(string=document, base_url=str(ART)).write_pdf(OUTPUT)
    pdf = pymupdf.open(OUTPUT)
    normalize = lambda s: re.sub(r'\s+', '', s)
    printed = normalize(''.join(page.get_text() for page in pdf))
    missing = [name for name in sorted(all_names) if normalize(LABELS.get(name, name)) not in printed]
    overflow = []
    for number, page in enumerate(pdf, 1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if not page.rect.contains(pymupdf.Rect(span['bbox'])):
                        overflow.append({'page':number, 'text':span['text']})
    report = dict(pdf=str(OUTPUT), pages=len(pdf), diagrams=len(figures),
                  min_diagram_font_pt=min(d['font_pt'] for d in dimensions),
                  font='Noto Sans CJK SC', call_tree_included=False, missing_classes=missing,
                  overflow=overflow, render_command=command,
                  renderer_version=subprocess.check_output([str(CLI), '--version'], env=ENV, text=True).strip())
    (ART / 'check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    assert not missing and not overflow, 'PDF content/layout check failed'


if __name__ == "__main__":
    main()
