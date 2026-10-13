#!/usr/bin/env python3
"""Render a PDF using literal source comments as all explanatory text."""

import html
import json
import sys
from collections import Counter
from pathlib import Path

from book_content import group_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from source_blocks import collect_source, excerpts_for, validate_excerpts

OUTPUT = ROOT / "user_guide_zh" / "SGLang源码学习笔记.pdf"

def entry_id(i):
    return f"entry-{i:03d}"


def render_note(entry, index, blocks):
    name = html.escape(entry['name']).replace('.', '.<wbr>').replace('_', '_<wbr>')
    kind = 'class ' if entry['kind'] == 'class' else ''
    short = sum(len(b['text']) for b in blocks) < 450 and sum(b['text'].count('\n') for b in blocks) < 8
    ident = entry_id(index) if isinstance(index, int) else index
    parts = [f'<div class="note{" short" if short else ""}" id="{ident}" data-entry="{index}"><div class="note-name">{kind}{name}<span class="line">L{entry["line"]}</span></div>']
    for block in blocks:
        label = ''
        if block.get('from_method'):
            label = f'<span class="block-line">{html.escape(block["from_method"])}</span>'
        parts.append(f'<div class="source-block">{label}<div class="note-text">{html.escape(block["text"])}</div></div>')
    parts.append('</div>')
    return ''.join(parts)


def render(snapshot,catalog,module_catalog):
    entries = snapshot["entries"]
    counts = Counter(e["kind"] for e in entries)
    methods = counts["method"] + counts["function"]
    css = (HERE / "style.css").read_text()
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>SGLang 源码学习笔记</title><meta name="author" content="SGLang learning notes">',f'<style>{css}</style></head><body>']
    out.append('<h1>SGLang 源码学习笔记</h1>')
    out.append(f'<p class="intro">{counts["class"]} 个类（含方法归属类） · {methods} 个方法 / 函数 · {snapshot["created_utc"][:10]}<br>保留已有学习注释、公式和示例。源码路径省略 python/sglang/。</p>')
    titles=['启动与模型加载','请求处理与调度','模型前向与并行算子','KV 池与前缀缓存','采样、解码与回包']
    for g,title in enumerate(titles):
        group_entries = [(i,e) for i,e in enumerate(entries) if note_group(e)==g]
        out.append(f'<h2 id="group-{g}">{g+1}. {title}</h2>')
        if NOTE_CHAINS.get(g):
            for title_,text in NOTE_CHAINS[g]:
                out.append(f'<div class="call-chain"><div class="chain-label">{title_}</div><pre>{html.escape(text)}</pre></div>')
        previous_file=None
        for i,e in group_entries:
            if e['file']!=previous_file:
                path=e['file'].removeprefix('python/sglang/')
                out.append(f'<p class="file-label">{html.escape(path)}</p>')
                previous_file=e['file']
            out.append(render_note(e, i, catalog[i]['excerpts']))
        for j, record in enumerate(module_catalog):
            e = record['entry']
            if note_group(e) == g:
                path = e['file'].removeprefix('python/sglang/')
                out.append(f'<p class="file-label">{html.escape(path)}</p>')
                out.append(render_note(e, f'module-{j}', record['excerpts']))
    out.append('</body></html>')
    return "".join(out)


def note_group(e):
    old=group_for(e)
    if old==9:
        if 'layers/linear' in e['file'] or 'qwen3_5_text' in e['file']:
            return 2
        if 'speculative/' in e['file'] or e['name']=='get_or_create_event_loop':
            return 1
        return 0
    return {0:0,1:0,2:1,3:1,4:2,5:2,6:2,7:3,8:4}[old]


NOTE_CHAINS={
    0:[('调用关系', '''ModelRunner.__init__()
  → initialize() → load_model()
  → load_model_with_memory_saver()
  → DefaultModelLoader.load_model()
      ├─ _initialize_model()
      │   → Qwen3_5ForConditionalGeneration.__init__()
      │   → Qwen3VLForConditionalGeneration.__init__()
      │       → Qwen3_5ForCausalLM.__init__()
      └─ load_weights_and_postprocess()
          → model.load_weights(...)''')],
    1:[('调用关系', '''openai_v1_chat_completions()
  → OpenAIServingBase.handle_request()
      ├─ OpenAIServingChat._convert_to_internal_request()
      └─ OpenAIServingChat._handle_non_streaming_request()
          → TokenizerManager.generate_request()
              → _send_one_request() → _dispatch_to_scheduler()

Scheduler.event_loop_overlap()
  ├─ process_input_requests()
  ├─ get_next_batch_to_run()
  ├─ run_batch()
  └─ process_batch_result()''')],
    2:[('前向调用关系', '''Scheduler.run_batch()
  → TpModelWorker.forward_batch_generation()
  → ModelRunner.forward() → _forward_raw()
  → EagerRunner.execute()
      ├─ _execute_extend() → model.forward(...)
      └─ _execute_decode() → model.forward(...)

Qwen3VLForConditionalGeneration.forward()
  ├─ Qwen3_5ForCausalLM.forward()
  └─ LogitsProcessor.forward()'''),
       ('QKV 调用关系', '''self.qkv_proj(hidden_states)
  │ qkv_proj 是 QKVParallelLinear 对象
  │ 本例 bias = None
  → torch.nn.Module.__call__()
      → ColumnParallelLinear.forward(input_=hidden_states)
          │ QKVParallelLinear 没有重写 forward，使用父类实现
          ├─ self.quant_method.apply(self, input_, bias)
          │   → UnquantizedLinearMethod.apply()
          │       → F.linear(x, layer.weight, bias)
          │           └─ output = x @ weight.T
          └─ 返回 (output, output_bias)''')],
    3:[('调用关系', '''Scheduler.init_memory_pools()
  → init_target_memory_pool()
  → TpModelWorker.alloc_memory_pool()
  → ModelRunner.alloc_memory_pool()
  → KVCacheConfigurator.configure()
      ├─ _derive_pool_sizes()
      └─ _init_pools()''')],
    4:[],
}


def main():
    from weasyprint import HTML
    snapshot = json.loads((HERE / "annotations.json").read_text())
    if snapshot["skipped"]:
        raise ValueError(snapshot["skipped"])
    sources = collect_source(snapshot)
    catalog=[]
    for e in snapshot["entries"]:
        excerpts=excerpts_for(e,sources,snapshot["entries"])
        catalog.append({"file":e["file"],"name":e["name"],"excerpts":excerpts})
    module_catalog=[]
    for module in snapshot['module_notes']:
        entry={**module, 'kind':'module', 'name':'模块级注释', 'line':module['annotations'][0]['line']}
        module_catalog.append({'entry':entry,'excerpts':excerpts_for(entry,sources,[])})
    coverage=validate_excerpts(snapshot,sources,catalog,module_catalog)
    (HERE/"excerpts.json").write_text(json.dumps({'entries':catalog,'module_notes':module_catalog},ensure_ascii=False,indent=2)+"\n")
    document=render(snapshot,catalog,module_catalog)
    (HERE/"notes.html").write_text(document)
    rendered=HTML(string=document,base_url=str(HERE)).render()
    ids=set()
    overflow=[]
    for i,page in enumerate(rendered.pages):
        for b in page._page_box.descendants():
            if getattr(b,"element",None) is not None and b.element.get("data-entry") is not None:
                ids.add(b.element.get("data-entry"))
            if type(b).__name__=="TextBox" and (b.position_x < -1 or b.position_x+b.width > page.width+1 or b.position_y < -1 or b.position_y+b.height > page.height+1):
                overflow.append({"page":i+1,"text":b.text})
    expected_ids={str(i) for i in range(len(snapshot['entries']))} | {f'module-{i}' for i in range(len(module_catalog))}
    assert ids==expected_ids
    if overflow:
        raise ValueError(f'PDF page overflow: {overflow}')
    temporary=OUTPUT.with_suffix('.tmp.pdf')
    rendered.write_pdf(temporary,pdf_identifier=True,srgb=True)
    temporary.replace(OUTPUT)
    report={"pdf":str(OUTPUT.relative_to(ROOT)),"pages":len(rendered.pages),"entries":len(snapshot['entries']),"module_notes":len(module_catalog),"page_overflow":overflow,"verbatim_source_check":"passed","authored_explanations":0,"coverage":coverage}
    (HERE/"layout_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
