#!/usr/bin/env python3
"""Print the supplied attention decoder call tree on one A4 page."""
import html
import json
import re
from pathlib import Path

import pymupdf
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qwen35_attention_decoder_forward.txt"
OUTPUT = HERE.parents[1] / "user_guide_zh/Qwen3.5全注意力层_forward调用链.pdf"


def main():
    lines = SOURCE.read_text().splitlines()
    line_height = round(((297 - 8 - 6) * 72 / 25.4 - 2) / len(lines), 2)
    css = f"""
    @page {{ size: A4 portrait; margin: 8mm 8mm 6mm; }}
    body {{ margin: 0; color: #000; font-family: 'Noto Sans CJK SC';
            font-size: 11.5pt; line-height: {line_height}pt; }}
    .line {{ white-space: pre; }}
    .prefix {{ font-family: 'Noto Sans Mono CJK SC'; font-variant-ligatures: none; }}
    .stage {{ font-weight: 700; }}
    """
    body = []
    for number, line in enumerate(lines):
        prefix, text = re.match(r"^([ │├└─→]*)(.*)$", line).groups()
        stage = ' class="stage"' if number == 0 or re.search("[①②③④⑤⑥]", text) else ""
        body.append('<div class="line"><span class="prefix">' + html.escape(prefix)
                    + '</span><span' + stage + '>' + html.escape(text) + '</span></div>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>Qwen3.5 全注意力层 forward 调用链</title><style>' + css
                + '</style><body>' + ''.join(body) + '</body></html>')
    (HERE / "qwen35_attention_decoder_forward.html").write_text(document)
    HTML(string=document, base_url=str(HERE)).write_pdf(OUTPUT)
    doc = pymupdf.open(OUTPUT)
    normalize = lambda s: re.sub(r"\s+", "", s)
    printed = normalize(''.join(page.get_text() for page in doc))
    missing = [line for line in lines if normalize(line) not in printed]
    overflow = []
    for number, page in enumerate(doc, 1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 10 or y0 < 10 or x1 > page.rect.width - 10 or y1 > page.rect.height - 10:
                        overflow.append({'page': number, 'text': span['text']})
        page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).save(
            HERE / f"qwen35_attention_decoder_preview_{number}.png")
    report = dict(pdf=str(OUTPUT), pages=len(doc), font_pt=11.5,
                  line_height_pt=line_height, missing=missing, overflow=overflow)
    (HERE / "qwen35_attention_decoder_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    assert len(doc) == 1 and not missing and not overflow


if __name__ == '__main__':
    main()
