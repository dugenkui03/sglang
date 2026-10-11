#!/usr/bin/env python3
"""Render the four reviewed Qwen3.5 call trees as four A4 portrait pages."""

import html
import json
import re
from pathlib import Path

import pymupdf
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qwen35_prefill_decode_trees.md"
OUTPUT = HERE.parents[1] / "user_guide_zh/Qwen3.5推理链路_Prefill与Decode四图.pdf"
# Fill each page with its tree, keeping a printable 5 mm bottom margin.
TREE_HEIGHT_PT = 297 * 72 / 25.4 - (8 + 5) * 72 / 25.4 - 23 - 2
CSS = """
@page { size: A4 portrait; margin: 8mm 8mm 5mm; }
body { margin: 0; color: #000; font-family: 'Noto Sans CJK SC'; }
.page { break-before: page; }
.page:first-child { break-before: auto; }
h2 { font-size: 12pt; margin: 0 0 3pt; line-height: 15pt; font-weight: 700; }
.tree { margin-top: 5pt; font-size: 10pt; }
.line { white-space: pre; break-inside: avoid; }
.prefix { font-family: 'Noto Sans Mono CJK SC'; font-variant-ligatures: none; }
.stage { font-weight: 700; }
"""


def normalize(value):
    return re.sub(r"\s+", "", value)


def main():
    sections = re.findall(
        r"^## (.*?)\n\n(.*?)\n\n```text\n(.*?)\n```",
        SOURCE.read_text(), re.M | re.S,
    )
    assert len(sections) == 4
    body = []
    line_heights = []
    for title, scope, tree in sections:
        line_height = round(TREE_HEIGHT_PT / len(tree.splitlines()), 2)
        line_heights.append(line_height)
        lines = []
        for line in tree.splitlines():
            prefix, text = re.match(r"^([ │├└─→]*)(.*)$", line).groups()
            stage = ' class="stage"' if re.search("[①②③④]", text) else ""
            lines.append('<div class="line"><span class="prefix">'
                         + html.escape(prefix) + '</span><span' + stage + '>'
                         + html.escape(text) + '</span></div>')
        body.append('<section class="page"><h2>' + html.escape(title) + '</h2>'
                    + f'<div class="tree" style="line-height:{line_height}pt">'
                    + ''.join(lines) + '</div></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>Qwen3.5：Prefill 与 Decode 四图</title><style>' + CSS
                + '</style><body>' + ''.join(body) + '</body></html>')
    (HERE / "qwen35_prefill_decode_trees.html").write_text(document)
    HTML(string=document, base_url=str(HERE)).write_pdf(OUTPUT)
    doc = pymupdf.open(OUTPUT)
    issues = []
    if len(doc) != 4:
        issues.append({"unexpected_page_count": len(doc)})
    page_info = []
    for number, page in enumerate(doc, 1):
        text = normalize(page.get_text())
        if number <= len(sections):
            title, scope, tree = sections[number - 1]
            missing = [line for line in (title + '\n' + tree).splitlines()
                       if line.strip() and normalize(line) not in text]
            if missing:
                issues.append({"page": number, "missing": missing})
        boxes = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    boxes.append(span['bbox'])
                    if x0 < 10 or y0 < 10 or x1 > page.rect.width - 10 or y1 > page.rect.height - 10:
                        issues.append({"page": number, "overflow": span['text']})
        page_info.append({"page": number, "width_pt": page.rect.width,
                          "height_pt": page.rect.height,
                          "content_bottom_pt": max(box[3] for box in boxes),
                          "bottom_whitespace_mm": round(
                              (page.rect.height - max(box[3] for box in boxes)) * 25.4 / 72, 2)})
        page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).save(
            HERE / f"qwen35_prefill_decode_preview_{number}.png")
    doc.set_toc([[1, title, i + 1] for i, (title, _, _) in enumerate(sections)])
    doc.saveIncr()
    report = {"pdf": str(OUTPUT), "pages": len(doc), "font_pt": 10,
              "line_heights_pt": line_heights, "introductory_descriptions": False,
              "page_info": page_info, "issues": issues}
    (HERE / "qwen35_prefill_decode_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    assert not issues, "PDF text or layout check failed"


if __name__ == '__main__':
    main()
