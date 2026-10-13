#!/usr/bin/env python3
"""Print learned methods, preserving source lines while omitting imports."""

import argparse
import ast
import hashlib
import html
import io
import json
import re
import sys
import tokenize
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from quick_learn.pdf_notes.extract_annotations import definitions, extract
from quick_learn.update_learn_progress import extent, measure

OUTPUT = ROOT / "user_guide_zh/SGLang已读核心方法_打印版.pdf"
GROUPS = [
    ("请求与模型加载", ["http", "chat", "tokenizer", "loader"]),
    ("调度与前向执行", ["scheduler", "policy", "batch", "worker", "runner", "forward_batch", "eager"]),
    ("模型结构与基础计算", ["qwen_entry", "qwen_model", "embedding", "linear", "unquant", "norm"]),
    ("注意力与缓存", ["attention", "linear_attention", "pool", "radix"]),
    ("采样与输出", ["logits", "sampler", "detokenizer"]),
]

CSS = """
@page { size:A4; margin:15mm 15mm 16mm;
  @bottom-left { content:string(current_file); font-family:'DejaVu Sans Mono','Noto Sans CJK SC'; font-size:6.5pt; color:#666; }
  @bottom-right { content:counter(page); font-family:'Noto Sans CJK SC'; font-size:8pt; color:#555; }
}
* { box-sizing:border-box; }
body { margin:0; color:#111; font-family:'Noto Sans CJK SC',sans-serif; font-size:9pt; line-height:1.5; }
.title { font-size:16pt; font-weight:bold; margin:0 0 3mm; }
.intro { font-size:8.2pt; color:#444; margin-bottom:3mm; }
h1 { font-size:14pt; margin:0 0 3mm; padding-bottom:2mm; border-bottom:1px solid #333; bookmark-level:1; }
.group + .group { break-before:page; }
h2 { font-size:11pt; margin:5mm 0 2mm; bookmark-level:2; overflow-wrap:anywhere; }
h3 { font-size:10.5pt; margin:4mm 0 2mm; bookmark-level:3; overflow-wrap:anywhere; }
h4 { font-size:9.8pt; font-weight:bold; margin:4mm 0 1.5mm; bookmark-level:4; overflow-wrap:anywhere; }
h1,h2,h3,h4 { break-after:avoid; }
.file-context { string-set:current_file content(text); }
.meta { color:#555; font-size:7.3pt; margin:0 0 1.5mm; overflow-wrap:anywhere; break-after:avoid; }
.code { font-family:'DejaVu Sans Mono','Noto Sans CJK SC',monospace; font-size:9.2pt; line-height:1.38; margin:0 0 2.5mm; }
.row { position:relative; padding-left:10mm; white-space:pre-wrap; overflow-wrap:anywhere; min-height:1.38em; break-inside:avoid; }
.num { position:absolute; left:0; top:0; width:8mm; text-align:right; font-size:6.7pt; color:#777; white-space:nowrap; }
.class-doc { margin:0 0 3mm; }
.class-doc .code { font-size:8.7pt; }
.method { margin-bottom:3mm; }
"""


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def collect():
    snapshot = extract()
    if snapshot["skipped"]:
        raise ValueError(snapshot["skipped"])
    report = measure()
    paths = dict(report["scope"]["sources"])
    paths.update(embedding="python/sglang/srt/layers/vocab_parallel_embedding.py",
                 norm="python/sglang/srt/layers/layernorm.py")
    chosen = {paths[key] for _, keys in GROUPS for key in keys}
    infos = {}
    for path in chosen:
        source = (ROOT / path).read_text()
        if path in snapshot["source_hashes"] and digest(source) != snapshot["source_hashes"][path]:
            raise ValueError(f"Source changed during extraction: {path}")
        tree = ast.parse(source)
        comments = {t.start[0]: t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
                    if t.type == tokenize.COMMENT}
        import_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.update(range(node.lineno, node.end_lineno + 1))
                # Refuse to remove another statement sharing an import's line.
                if source.splitlines()[node.end_lineno-1][node.end_col_offset:].lstrip().startswith(";"):
                    raise ValueError(f"Import shares a line with code: {path}:{node.lineno}")
        infos[path] = {"source":source, "lines":source.splitlines(), "defs":definitions(tree),
                       "imports":import_lines, "comments":comments}
    direct = {(e["file"], e["name"]): e for e in snapshot["entries"]
              if e["file"] in chosen and e["kind"] != "class" and e["direct"]}
    classes = {(e["file"],e["name"]):e for e in snapshot["entries"]
               if e["file"] in chosen and e["kind"] == "class"}
    learned = {(r["file"],r["qualname"]):r for r in report["rows"]
               if r["file"] in chosen and r["learned"]}
    records = []
    for path, name in sorted(direct.keys() | learned.keys()):
        node, parent = infos[path]["defs"][name]
        start,end = extent(node,infos[path]["lines"])
        record = {"file":path,"name":name,"parent_class":parent,"start":start,"end":end,
                  "status":"D" if (path,name) in direct else learned[path,name]["status"],
                  "evidence":direct.get((path,name),{}).get("annotations",[])}
        records.append(record)
    # Selected local functions already occur in their enclosing selected method.
    for entry in records:
        parents = [p for p in records if p["file"] == entry["file"]
                   and p["start"] <= entry["start"] and p["end"] >= entry["end"]
                   and p["name"] != entry["name"]]
        entry["embedded_in"] = min(parents,key=lambda p:p["start"])["name"] if parents else None
    return snapshot, paths, infos, records, classes


def load_corrections(infos):
    path=HERE / "print_comment_corrections.json"
    corrections=json.loads(path.read_text()) if path.exists() else []
    by_line={}
    for c in corrections:
        if c["file"] not in infos:
            continue
        original=infos[c["file"]]["lines"][c["line"]-1]
        if original != c["old"]:
            raise ValueError(f"Stale comment correction: {c['file']}:{c['line']}")
        key=(c["file"],c["line"])
        if key in by_line and by_line[key] != c:
            raise ValueError(f"Conflicting correction: {key}")
        by_line[key]=c
    # All replacements must leave executable tokens unchanged, except docstrings.
    for p,info in infos.items():
        revised=list(info["lines"])
        for (file,line),c in by_line.items():
            if p==file: revised[line-1]=c["new"]
        original_tree=ast.parse(info["source"])
        revised_tree=ast.parse("\n".join(revised))
        class RemoveNotes(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                    return None
                return self.generic_visit(node)
        if ast.dump(RemoveNotes().visit(original_tree)) != ast.dump(RemoveNotes().visit(revised_tree)):
            raise ValueError(f"Correction changed executable code: {p}")
    return by_line


def rows_for(path,start,end,info,corrections,used):
    rows=[]
    for line in range(start,end+1):
        text=info["lines"][line-1]
        key=(path,line)
        if key in corrections:
            text=corrections[key]["new"]
            used.add(key)
        if line in info["imports"]:
            # Retain user comments attached to an omitted import.
            note=info["comments"].get(line)
            if note and re.search(r"[\u3400-\u9fff]",note):
                text=" "*(len(text)-len(text.lstrip()))+note
            else:
                continue
        if not text.strip() and (not rows or not rows[-1]["text"].strip()):
            continue
        rows.append({"line":line,"text":text})
    while rows and not rows[-1]["text"].strip():
        rows.pop()
    # Remove the common class indentation, preserving method-body indentation.
    indent=min((len(r["text"])-len(r["text"].lstrip()) for r in rows if r["text"].strip()),default=0)
    return [{**r,"text":r["text"][indent:]} for r in rows]


def code_rows(rows):
    return '<div class="code">'+''.join(
        f'<div class="row"><span class="num">{r["line"]}</span><span class="txt">{html.escape(r["text"]) or " "}</span></div>'
        for r in rows)+'</div>'


def name_html(name):
    return html.escape(name).replace(".",".<wbr>").replace("_","_<wbr>")


def render(paths,infos,records,classes,corrections):
    used=set(); printed=[]; class_printed=[]
    date=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    out=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>SGLang 已读核心方法 · 打印版</title>',
         f'<style>{CSS}</style></head><body><div class="title">SGLang 已读核心方法</div>',
         f'<div class="intro">{len(records)} 个已读方法／函数条目 · {date}<br>'
         '范围沿用前述 24 个核心文件；注释已读与按约定推定已读分别标记。<br>'
         '保留方法体和学习注释；省略导入，连续空行压缩。嵌套方法随外层打印一次。<br>'
         '左侧为原始行号。明确有误的注释已在本打印版修正；源码未改动。</div>']
    for gi,(label,keys) in enumerate(GROUPS,1):
        out.append(f'<section class="group"><h1>{gi}. {label}</h1>')
        fi=0
        for alias in keys:
            path=paths[alias];info=infos[path]
            entries=[r for r in records if r["file"]==path and not r["embedded_in"]]
            # Class-level learning notes (e.g. RMS example) do not imply any unread method.
            class_only=[c for (p,n),c in classes.items() if p==path and c["direct"]
                        and not any(r["parent_class"]==n for r in entries)]
            if not entries and not class_only:
                continue
            fi+=1; short=path.removeprefix("python/sglang/")
            out.append(f'<h2>{gi}.{fi} {name_html(short)}</h2><div class="meta file-context">{html.escape(short)}</div>')
            buckets=OrderedDict()
            for r in sorted(entries,key=lambda e:e["start"]):
                buckets.setdefault(r["parent_class"],[]).append(r)
            for c in class_only:
                buckets.setdefault(c["name"],[])
            for ci,(cls,methods) in enumerate(buckets.items(),1):
                label=cls or "模块级函数"
                out.append(f'<h3>{gi}.{fi}.{ci} {name_html(label)}</h3>')
                if cls:
                    node=info["defs"][cls][0]
                    bases=", ".join(ast.unparse(b) for b in node.bases) or "object"
                    out.append(f'<div class="meta">class {name_html(cls)}({html.escape(bases)})</div>')
                    doc=node.body[0] if node.body else None
                    if (isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant)
                            and isinstance(doc.value.value,str)):
                        class_entry=classes.get((path,cls),{})
                        # Full original docstring if it contains learning notes; otherwise first paragraph.
                        if class_entry.get("direct"):
                            rows=rows_for(path,doc.lineno,doc.end_lineno,info,corrections,used)
                            out.append('<div class="class-doc">'+code_rows(rows)+'</div>')
                            class_printed.append({"file":path,"name":cls,"rows":rows})
                        elif not methods:
                            continue
                for mi,r in enumerate(methods,1):
                    ident=f'method-{len(printed)}'
                    shortname=r["name"].removeprefix(cls+".") if cls else r["name"]
                    status={"D":"注释已读","C":"推定已读 · 链路补全","B":"推定已读 · 分支补全"}[r["status"]]
                    out.append(f'<div class="method" id="{ident}"><h4>{gi}.{fi}.{ci}.{mi} {name_html(shortname)}()</h4>')
                    out.append(f'<div class="meta">{status} · 原始行号 {r["start"]}–{r["end"]}</div>')
                    embedded=[x["name"] for x in records if x["file"]==path and x["embedded_in"]==r["name"]]
                    if embedded:
                        out.append('<div class="meta">内含已读局部方法：'+', '.join(name_html(n) for n in embedded)+'</div>')
                    rows=rows_for(path,r["start"],r["end"],info,corrections,used)
                    out.append(code_rows(rows)+'</div>')
                    printed.append({**r,"id":ident,"rows":rows})
        out.append('</section>')
    out.append('</body></html>')
    return ''.join(out),printed,class_printed,used


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-selection',action='store_true',help='Reuse a checked snapshot only when source hashes match.')
    args=parser.parse_args()
    cache=HERE/'read_methods_selection.json'
    if args.reuse_selection and cache.exists():
        saved=json.loads(cache.read_text());paths=saved['paths'];records=saved['records']
        classes={(e['file'],e['name']):e for e in saved['classes']};infos={}
        for path,info in saved['sources'].items():
            source=(ROOT/path).read_text()
            if digest(source)!=info['sha256']:raise ValueError(f'Source changed: {path}')
            tree=ast.parse(source)
            infos[path]={**info,'source':source,'lines':source.splitlines(),'defs':definitions(tree),
                         'imports':set(info['imports']),'comments':{int(k):v for k,v in info['comments'].items()}}
    else:
        snapshot,paths,infos,records,classes=collect()
        saved={'created_utc':snapshot['created_utc'],'head':snapshot['head'],'paths':paths,'records':records,
               'classes':list(classes.values()),'sources':{p:{'sha256':digest(i['source']),'imports':sorted(i['imports']),
                                                          'comments':i['comments']} for p,i in infos.items()}}
        cache.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
    corrections=load_corrections(infos)
    document,printed,class_printed,used=render(paths,infos,records,classes,corrections)
    (HERE/'read_methods.html').write_text(document)
    catalog={'records':printed,'class_notes':class_printed,'corrections':[corrections[k] for k in sorted(used)]}
    (HERE/'read_methods_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    from weasyprint import HTML
    rendered=HTML(string=document,base_url=str(HERE)).render()
    ids=set();overflow=[]
    for i,page in enumerate(rendered.pages):
        for box in page._page_box.descendants():
            element=getattr(box,'element',None)
            if element is not None and element.get('id','').startswith('method-'):
                ids.add(element.get('id'))
            if type(box).__name__=='TextBox' and (
                box.position_x < 0 or box.position_x+box.width > page.width+1
                or box.position_y < 0 or box.position_y+box.height > page.height+1):
                overflow.append({'page':i+1,'text':box.text})
    assert ids=={r['id'] for r in printed},'Missing method in PDF layout'
    if overflow:raise ValueError(overflow)
    temporary=OUTPUT.with_suffix('.tmp.pdf')
    rendered.write_pdf(temporary,pdf_identifier=True,srgb=True)
    temporary.replace(OUTPUT)
    result={'pdf':str(OUTPUT),'pages':len(rendered.pages),'selected_methods':len(records),
            'printed_method_blocks':len(printed),'embedded_methods':len(records)-len(printed),
            'direct':sum(r['status']=='D' for r in records),'inferred':sum(r['status']!='D' for r in records),
            'method_files':len({r['file'] for r in records}),'class_notes':len(class_printed),
            'code_rows':sum(len(r['rows']) for r in printed),
            'comment_corrections':sum(corrections[k].get('kind')!='display_escape' for k in used),
            'display_escapes':sum(corrections[k].get('kind')=='display_escape' for k in used),'overflow':overflow}
    (HERE/'read_methods_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
