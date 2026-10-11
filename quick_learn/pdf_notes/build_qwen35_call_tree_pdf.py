#!/usr/bin/env python3
"""Print only the reviewed Qwen3.5 call tree on one portrait page."""

import html
import json
import re
from pathlib import Path

import pymupdf
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "qwen35_inference_call_tree.txt"
OUTPUT = ROOT / "user_guide_zh/Qwen3.5推理调用链_单页竖版.pdf"

CSS = """
@page {
  size: A4 portrait; margin: 5mm;
}
body { margin: 0; color: #000; font-family: 'Noto Sans CJK SC'; }
.tree { margin: 0; font-family: 'Noto Sans Mono CJK SC';
  font-size: 8pt; line-height: 7.7pt; white-space: pre;
  font-variant-ligatures: none; }
"""


def main():
    lines = SOURCE.read_text().splitlines()
    split = lines.index("", 4)
    tree = lines[5:split]
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                '<title>Qwen3.5 单卡推理调用链</title><style>' + CSS + '</style></head><body>'
                '<pre class="tree">' + html.escape('\n'.join(tree)) + '</pre></body></html>')
    (HERE / 'qwen35_inference_call_tree_portrait.html').write_text(document)
    HTML(string=document, base_url=str(HERE)).write_pdf(OUTPUT)
    doc = pymupdf.open(OUTPUT)
    normalize = lambda s: re.sub(r"\s+", "", s)
    printed = normalize(''.join(page.get_text() for page in doc))
    missing = [line for line in tree if line.strip() and normalize(line) not in printed]
    overflow = []
    for number, page in enumerate(doc, 1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width or y1 > page.rect.height:
                        overflow.append({'page': number, 'text': span['text']})
    report = dict(pdf=str(OUTPUT), pages=len(doc), paper='A4 portrait',
                  tree_font_pt=8, line_height_pt=7.7, margin_mm=5,
                  title=False, header=False, notes=False,
                  source_lines=len(lines), tree_lines=len(tree),
                  missing_lines=missing, overflow=overflow)
    (HERE / 'qwen35_call_tree_portrait_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    assert not missing and not overflow, 'PDF content/layout check failed'
    assert len(doc) == 1 and doc[0].rect.width < doc[0].rect.height, 'Expected one portrait page'


if __name__ == '__main__':
    main()
