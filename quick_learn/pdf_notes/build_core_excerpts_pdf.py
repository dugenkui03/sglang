#!/usr/bin/env python3
"""Build readable core-path excerpts from the frozen learned-method print edition."""

import hashlib
import html
import json
import re
from collections import OrderedDict
from pathlib import Path

from build_read_methods_pdf import CSS, GROUPS, name_html

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / 'user_guide_zh/SGLang已读核心链路_精简打印版.pdf'

CHAINS = [
    ('请求与加载', 'HTTP 接口 → OpenAI handler → TokenizerManager → Scheduler\n'
     'ModelRunner.load_model() → DefaultModelLoader.load_model()\n'
     '    ├─ _initialize_model()：创建模型与参数空间\n'
     '    └─ load_weights_and_postprocess()：加载权重与后处理'),
    ('每轮调度', 'event_loop_overlap()\n'
     '    ├─ 接收请求 → 入队 → 选择 prefill / decode 批次\n'
     '    ├─ run_batch() → TpModelWorker → ModelRunner\n'
     '    └─ 按重叠顺序处理前一批结果，交接下一轮输入'),
    ('前向数据顺序', '文本 token / embedding → Qwen3_5ForCausalLM.forward()\n'
     '    → 各层 Attention 或 GDN + MLP → 最终 norm\n'
     '    → 返回外层 → LogitsProcessor / LM head → logits'),
    ('缓存关系', 'req slot → req_to_token[slot, token 位置] → KV index\n'
     '    → 本层 K/V 缓冲区\n'
     'RadixCache：前缀复用、引用锁与可淘汰节点'),
    ('结果路径', 'logits → Sampler → token ID\n'
     '    → Scheduler 处理结果 → DetokenizerManager\n'
     '    → TokenizerManager 累积文本 → HTTP 响应'),
]


def load_inputs():
    source_path = HERE / 'read_methods_catalog.json'
    raw = source_path.read_bytes()
    original = json.loads(raw)
    selection = json.loads((HERE / 'read_methods_selection.json').read_text())
    plan = json.loads((HERE / 'core_excerpt_plan.json').read_text())
    if hashlib.sha256(raw).hexdigest() != plan['base_catalog_sha256']:
        raise ValueError('Print-edition source changed: review the excerpt ranges again.')
    by_key = {(r['file'], r['name']): r for r in original['records']}
    decisions = {(r['file'], r['name']): r for r in plan['items']}
    assert len(decisions) == len(plan['items'])
    assert decisions.keys() == by_key.keys(), 'Every previously printed method needs a decision.'
    for key, item in decisions.items():
        source = by_key[key]
        assert bool(item['keep']) != bool(item['omit']), key
        previous_end = source['start'] - 1
        for start, end in item['keep']:
            assert source['start'] <= start <= end <= source['end'], (key, start, end)
            assert start > previous_end, (key, 'overlapping or unsorted ranges')
            previous_end = end
    corrections = {(c['file'], c['line']): c for c in plan['corrections']}
    for key, correction in corrections.items():
        matches = [row for r in original['records'] + original['class_notes']
                   if r['file'] == key[0] for row in r['rows'] if row['line'] == key[1]]
        assert matches and all(r['text'].lstrip() == correction['old'].lstrip() for r in matches), key
    return original, selection, plan, decisions, corrections


def corrected_row(path, row, corrections):
    result = dict(row)
    change = corrections.get((path, row['line']))
    if change:
        indent = len(row['text']) - len(row['text'].lstrip())
        result['text'] = ' ' * indent + change['new'].lstrip()
    return result


def excerpt(record, decision, corrections):
    included = {n for a, b in decision['keep'] for n in range(a, b + 1)}
    output, pending = [], []

    def gap():
        if not any(r['text'].strip() for r in pending):
            pending.clear()
            return
        first, last = pending[0]['line'], pending[-1]['line']
        indent = min((len(r['text']) - len(r['text'].lstrip()) for r in pending
                      if r['text'].strip()), default=0)
        output.append({'line': None, 'text': ' ' * indent + f'… 〔省略原 L{first}–{last}〕',
                       'omitted_range': [first, last]})
        pending.clear()

    for row in record['rows']:
        if row['line'] in included:
            gap()
            output.append(corrected_row(record['file'], row, corrections))
        else:
            pending.append(row)
    gap()
    return output


def code_rows(rows):
    out = ['<div class="code">']
    for row in rows:
        is_gap = row['line'] is None
        num = '⋯' if is_gap else str(row['line'])
        out.append(f'<div class="row{" gap" if is_gap else ""}"><span class="num">{num}</span>'
                   f'<span class="txt">{html.escape(row["text"]) or " "}</span></div>')
    out.append('</div>')
    return ''.join(out)


def contents_entries(selection, kept, docs):
    """One flat entry per class (or module-function group), matching body order."""
    groups = []
    for gi, (title, aliases) in enumerate(GROUPS, 1):
        entries = []
        fi = 0
        for alias in aliases:
            path = selection['paths'][alias]
            methods = [r for r in kept if r['file'] == path]
            notes = [r for (file, _), r in docs.items() if file == path]
            if not methods and not notes:
                continue
            fi += 1
            buckets = OrderedDict()
            for record in sorted(methods, key=lambda r:r['start']):
                buckets.setdefault(record['parent_class'], []).append(record)
            for note in notes:
                buckets.setdefault(note['name'], [])
            for ci, (cls, records) in enumerate(buckets.items(), 1):
                names = []
                for record in records:
                    name = record['name'].removeprefix(cls + '.') if cls else record['name']
                    is_property = any(row['text'].split('#', 1)[0].strip() == '@property' for row in record['rows'])
                    names.append({'name':name, 'qualname':record['name'], 'id':record['id'],
                                  'label':name + ('（属性）' if is_property else '()')})
                entries.append({'file':path, 'class':cls,
                                'label':cls or Path(path).name + '（模块函数）',
                                'id':f'core-class-{gi}-{fi}-{ci}', 'methods':names})
        groups.append({'number':gi, 'title':title, 'entries':entries})
    return groups


def render_contents(groups):
    out = ['<section class="toc"><h1>目录</h1>']
    for group in groups:
        out.append(f'<h2 class="toc-group">{group["number"]}. {html.escape(group["title"])}</h2>')
        for entry in group['entries']:
            target = '#' + entry['id']
            out.append(f'<div class="toc-line"><b><a href="{target}">{name_html(entry["label"])}</a></b>：')
            if entry['methods']:
                out.append('、'.join(f'<a href="#{method["id"]}">{name_html(method["label"])}</a>'
                                    for method in entry['methods']))
            else:
                out.append('类说明')
            out.append(f'<a class="toc-page" href="{target}"></a></div>')
    out.append('</section>')
    return ''.join(out)


def main():
    original, selection, plan, decisions, corrections = load_inputs()
    kept = []
    omitted = []
    for record in original['records']:
        decision = decisions[record['file'], record['name']]
        if decision['omit']:
            omitted.append({'file':record['file'], 'name':record['name'], 'reason':decision['reason']})
        else:
            kept.append({**record, 'rows':excerpt(record, decision, corrections),
                         'focus':decision.get('focus') or decision['reason']})
    retained_lines = {(r['file'], x['line']) for r in kept for x in r['rows'] if x['line'] is not None}
    nested = [r for r in selection['records'] if r['embedded_in'] and (r['file'], r['start']) in retained_lines]
    classes = {(c['file'], c['name']): c for c in selection['classes']}
    docs = {(r['file'], r['name']): r for r in original['class_notes']}
    printed_docs = []
    css = CSS + '\n.gap{color:#666;font-size:8pt;} .focus{font-size:8.3pt;margin:0 0 2mm;break-after:avoid;} .chain{font-family:"DejaVu Sans Mono","Noto Sans CJK SC";font-size:8.5pt;white-space:pre-wrap;margin:0 0 4mm;break-inside:avoid;}\n'
    css += '''
@page contents { @bottom-left { content:'目录'; } }
.toc { page:contents; break-after:page; }
.toc h1 { bookmark-level:1; }
.toc-group { font-size:11pt; margin:5mm 0 2.5mm; bookmark-level:none; }
.toc-group:nth-of-type(3) { break-before:page; }
.toc-line { position:relative; padding-right:10mm; margin:0 0 2.2mm; font-size:8.5pt; line-height:1.55; break-inside:avoid; overflow-wrap:anywhere; }
.toc a { color:#111; text-decoration:none; }
.toc-page { position:absolute; top:0; right:0; font-size:8pt; }
.toc-page::after { content:target-counter(attr(href), page); }
'''
    toc = contents_entries(selection, kept, docs)
    (HERE / 'core_toc.json').write_text(json.dumps({'groups':toc}, ensure_ascii=False, indent=2) + '\n')
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>SGLang 已读核心链路 · 精简打印版</title>',
           f'<style>{css}</style></head><body>', render_contents(toc),
           '<div class="title" id="core-body">SGLang 已读核心链路</div>',
           '<div class="intro">Qwen3.5 dense 文本生成；重点保留已读主链路与相关基础代码、张量说明和学习注释。<br>'
           '默认加载、Overlap 调度、Prefill / Decode、TP、前缀缓存与普通采样；优化路径保留必要入口。<br>'
           '代码为阅读节选，省略处明确标记，不能直接作为完整函数运行。行号对应上一版收录的源码。<br>'
           '每个方法保留关键输入、处理步骤与结果；注释错误仅在本文修正，源码未改动。</div>']
    for gi, (title, aliases) in enumerate(GROUPS, 1):
        out.append(f'<section class="group"><h1>{gi}. {title}</h1>')
        label, chain = CHAINS[gi-1]
        out.append(f'<div class="focus"><b>{label}</b></div><div class="chain">{html.escape(chain)}</div>')
        fi = 0
        for alias in aliases:
            path = selection['paths'][alias]
            methods = [r for r in kept if r['file'] == path]
            class_notes = [r for key, r in docs.items() if key[0] == path]
            if not methods and not class_notes:
                continue
            fi += 1
            short = path.removeprefix('python/sglang/')
            out.append(f'<h2>{gi}.{fi} {name_html(short)}</h2><div class="meta file-context">{html.escape(short)}</div>')
            buckets = OrderedDict()
            for record in sorted(methods, key=lambda r:r['start']):
                buckets.setdefault(record['parent_class'], []).append(record)
            for note in class_notes:
                buckets.setdefault(note['name'], [])
            for ci, (cls, entries) in enumerate(buckets.items(), 1):
                out.append(f'<h3 id="core-class-{gi}-{fi}-{ci}">{gi}.{fi}.{ci} {name_html(cls or "模块级函数")}</h3>')
                if cls:
                    bases = ', '.join(classes.get((path, cls), {}).get('bases', [])) or 'object'
                    out.append(f'<div class="meta">class {name_html(cls)}({html.escape(bases)})</div>')
                    if (path, cls) in docs:
                        note = docs[path, cls]
                        rows = [corrected_row(path, row, corrections) for row in note['rows']]
                        out.append('<div class="class-doc">' + code_rows(rows) + '</div>')
                        printed_docs.append({**note, 'rows':rows})
                for mi, record in enumerate(entries, 1):
                    name = record['name'].removeprefix(cls + '.') if cls else record['name']
                    status = {'D':'注释已读', 'C':'推定已读 · 链路补全', 'B':'推定已读 · 分支补全'}[record['status']]
                    out.append(f'<div class="method" id="{record["id"]}"><h4>{gi}.{fi}.{ci}.{mi} {name_html(name)}()</h4>')
                    out.append(f'<div class="meta">{status} · 方法原始行号 {record["start"]}–{record["end"]}</div>')
                    out.append(f'<p class="focus">{html.escape(record["focus"])}</p>')
                    out.append(code_rows(record['rows']) + '</div>')
        out.append('</section>')
    out.append('</body></html>')
    document = ''.join(out)
    (HERE / 'core_excerpts.html').write_text(document)
    catalog = {'base_catalog_sha256':plan['base_catalog_sha256'], 'records':kept,
               'class_notes':printed_docs, 'nested_methods':nested, 'omitted_methods':omitted}
    (HERE / 'core_excerpts_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    from weasyprint import HTML
    rendered = HTML(string=document, base_url=str(HERE)).render()
    seen = set(); overflow = []
    for i, page in enumerate(rendered.pages):
        for box in page._page_box.descendants():
            element = getattr(box, 'element', None)
            if element is not None and element.get('id', '').startswith('method-'):
                seen.add(element.get('id'))
            if type(box).__name__ == 'TextBox' and (
                box.position_x < 0 or box.position_x + box.width > page.width + 1
                or box.position_y < 0 or box.position_y + box.height > page.height + 1):
                overflow.append({'page':i+1, 'text':box.text})
    assert seen == {r['id'] for r in kept}, 'Missing excerpt in layout'
    assert not overflow, overflow
    temp = OUTPUT.with_suffix('.tmp.pdf')
    rendered.write_pdf(temp, pdf_identifier=True, srgb=True)
    temp.replace(OUTPUT)
    source_rows = sum(x['line'] is not None for r in kept for x in r['rows'])
    applied = [c for c in plan['corrections'] if (c['file'], c['line']) in retained_lines]
    check = {'pdf':str(OUTPUT), 'pages':len(rendered.pages), 'previous_pages':224,
             'standalone_methods':len(kept), 'nested_methods':len(nested), 'omitted_methods':len(omitted),
             'previous_source_rows':sum(len(r['rows']) for r in original['records']), 'source_rows':source_rows,
             'omission_markers':sum(x['line'] is None for r in kept for x in r['rows']),
             'additional_comment_corrections':len(applied), 'layout_overflow':overflow,
             'toc_entries':sum(len(g['entries']) for g in toc),
             'toc_methods':sum(len(e['methods']) for g in toc for e in g['entries'])}
    (HERE / 'core_excerpts_check.json').write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(check, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
