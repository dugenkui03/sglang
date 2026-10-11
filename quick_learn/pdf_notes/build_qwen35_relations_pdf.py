#!/usr/bin/env python3
"""Append source-checked class diagrams to the one-page Qwen3.5 call tree."""

import html
import json
import re
from pathlib import Path

import pymupdf
from weasyprint import HTML

from build_qwen35_call_tree_pdf import OUTPUT as CALL_TREE_PDF
from build_qwen35_call_tree_pdf import main as build_call_tree

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "user_guide_zh/Qwen3.5推理调用链与核心类关系.pdf"
SOURCES = [HERE / "qwen35_core_inheritance.txt", HERE / "qwen35_core_composition.txt"]
CSS = """
@page { size: A4 portrait; margin: 9mm; }
body { margin: 0; color: #000; font-family: 'Noto Sans CJK SC'; }
.page + .page { break-before: page; }
.group { break-inside: avoid; margin-bottom: 3mm; }
h2 { font-size: 12pt; line-height: 1.3; margin: 0 0 1.5mm;
     padding-bottom: 1mm; border-bottom: 0.4pt solid #777; }
pre { margin: 0; font-family: 'Noto Sans Mono CJK SC'; font-size: 11pt;
      line-height: 1.2; white-space: pre; font-variant-ligatures: none; }
.legend { font-size: 9pt; margin: 0 0 3mm; line-height: 1.4; }
"""


def page_html(text):
    chunks = re.split(r"(?m)^## ", text.strip())
    legend = chunks.pop(0).strip()
    rendered = '<p class="legend">' + html.escape(legend) + '</p>' if legend else ''
    for chunk in chunks:
        title, body = chunk.split('\n', 1)
        rendered += ('<section class="group"><h2>' + html.escape(title)
                     + '</h2><pre>' + html.escape(body.rstrip()) + '</pre></section>')
    return '<article class="page">' + rendered + '</article>'


def main():
    build_call_tree()
    sources = [path.read_text() for path in SOURCES]
    pages = [part for source in sources for part in source.split('===PAGE===')]
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>Qwen3.5 核心类关系</title><style>' + CSS + '</style><body>'
                + ''.join(map(page_html, pages)) + '</body></html>')
    (HERE / 'qwen35_core_relations.html').write_text(document)
    appendix = pymupdf.open(stream=HTML(string=document).write_pdf(), filetype='pdf')
    assert len(appendix) == 4, f'Expected four diagram pages, got {len(appendix)}'
    combined = pymupdf.open(CALL_TREE_PDF)
    combined.insert_pdf(appendix)
    combined.set_metadata({'title': 'Qwen3.5 推理调用链与核心类关系'})
    combined.set_toc([[1, '推理调用链', 1], [1, '继承：模型与计算模块', 2],
                      [1, '继承：执行器、后端与缓存', 3],
                      [1, '组合：模型与 Decoder 层', 4],
                      [1, '引用：后端、缓存与本轮状态', 5]])
    combined.save(OUTPUT, garbage=4, deflate=True)
    normalize = lambda s: re.sub(r'\s+', '', s)
    printed = normalize(''.join(page.get_text() for page in appendix))
    expected = [line.removeprefix('## ') for source in sources
                for line in source.splitlines() if line.strip() and line != '===PAGE===']
    missing = [line for line in expected if normalize(line) not in printed]
    overflow = []
    for number, page in enumerate(combined, 1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if not page.rect.contains(pymupdf.Rect(span['bbox'])):
                        overflow.append({'page': number, 'text': span['text']})
    report = dict(pdf=str(OUTPUT), pages=len(combined), paper='A4 portrait',
                  inheritance_groups=8, composition_groups=7,
                  missing_lines=missing, overflow=overflow)
    (HERE / 'qwen35_relations_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    assert not missing and not overflow, 'PDF content/layout check failed'


if __name__ == '__main__':
    main()
