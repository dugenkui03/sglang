"""绘制可复现的中文幻灯片；文字、流程图与时序图均保留为结构化源数据。"""

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH, HEIGHT = 1920, 1080
INK = '#34443B'
MUTED = '#65735F'
ACCENT = '#57735D'
PALE = '#E7EDDF'
CIRCLED = '①②③④⑤⑥⑦⑧⑨'


def step_marker(value):
    """将一开始的步骤号绘成紧凑标记；超过九步时退回普通数字。"""
    number = int(value)
    return CIRCLED[number - 1] if 1 <= number <= len(CIRCLED) else f'{number}.'


def wrapped(text, font, width):
    """按实际字体宽度换行，中文无需依赖空格。"""
    lines = []
    for paragraph in str(text).splitlines() or ['']:
        current = ''
        tokens = re.findall(r'[A-Za-z0-9_]+|.', paragraph)
        for token in tokens:
            if font.getlength(token) > width:
                pieces = list(token)
            else:
                pieces = [token]
            for char in pieces:
                if current and font.getlength(current + char) > width:
                    lines.append(current)
                    current = char.lstrip()
                else:
                    current += char
        lines.append(current)
    return lines


class Canvas:
    def __init__(self, font_path):
        self.font_path = str(font_path)
        self.image = Image.new('RGB', (WIDTH, HEIGHT), '#F2F1E8')
        self.draw = ImageDraw.Draw(self.image)

    def text(
        self,
        text,
        box,
        size=32,
        minimum=22,
        color=INK,
        center=False,
        preserve_tokens=False,
    ):
        x, y, w, h = box
        minimum = min(minimum, size)
        for point in range(size, minimum - 1, -1):
            font = ImageFont.truetype(self.font_path, point)
            if preserve_tokens and point > minimum:
                tokens = re.findall(r'[A-Za-z0-9_]+', str(text))
                if any(font.getlength(token) > w for token in tokens):
                    continue
            lines = wrapped(text, font, w)
            line_height = int(point * 1.45)
            if len(lines) * line_height <= h:
                break
        else:
            raise ValueError(f'幻灯片内容放不下，请拆页或缩短文字：{text[:100]}')
        if center:
            y += (h - len(lines) * line_height) / 2
        for line in lines:
            left = x + (w - font.getlength(line)) / 2 if center else x
            self.draw.text((left, y), line, font=font, fill=color, anchor='lt')
            y += line_height

    def arrow(self, start, end, color=ACCENT, dashed=False):
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        if dashed:
            for offset in range(0, int(length), 18):
                stop = min(offset + 10, length)
                self.draw.line((x1 + ux * offset, y1 + uy * offset, x1 + ux * stop, y1 + uy * stop), fill=color, width=3)
        else:
            self.draw.line((*start, *end), fill=color, width=4)
        self.draw.polygon([(x2, y2), (x2 - 16 * ux - 8 * uy, y2 - 16 * uy + 8 * ux), (x2 - 16 * ux + 8 * uy, y2 - 16 * uy - 8 * ux)], fill=color)

    def card(self, title, body, box):
        x, y, w, h = box
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=PALE, outline='#C4CFBA', width=2)
        self.text(title, (x + 22, y + 18, w - 44, 70), size=32, minimum=23, color=ACCENT)
        self.text(body, (x + 22, y + 96, w - 44, h - 114), size=27, minimum=22)


def draw_relations(page, visual, bounds):
    """NOTE 用带类型的连线区分数据传递、持有、调用与继承。"""
    x, y, w, h = bounds
    boxes = {n['id']: (x + n['x'] * w, y + n['y'] * h, n['w'] * w, n['h'] * h) for n in visual['nodes']}
    labels = []
    for edge_index, edge in enumerate(visual['edges'], 1):
        ax, ay, aw, ah = boxes[edge['from']]
        bx, by, bw, bh = boxes[edge['to']]
        ac, bc = (ax + aw / 2, ay + ah / 2), (bx + bw / 2, by + bh / 2)
        horizontal = abs(ac[1] - bc[1]) < 5
        if horizontal:
            direction = 1 if bc[0] > ac[0] else -1
            start, end = (ac[0] + direction * aw / 2, ac[1]), (bc[0] - direction * bw / 2, bc[1])
        else:
            direction = 1 if bc[1] > ac[1] else -1
            start, end = (ac[0], ac[1] + direction * ah / 2), (bc[0], bc[1] - direction * bh / 2)
        length = math.dist(start, end)
        if length < 24:
            raise ValueError('关系图连线过短，请增加节点间距')
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        if edge['kind'] == 'inherits':
            # NOTE 空心三角指向父类，避免把组合误画成继承。
            base = (end[0] - ux * 20, end[1] - uy * 20)
            page.draw.line((*start, *base), fill=ACCENT, width=3)
            triangle = [end, (base[0] - uy * 10, base[1] + ux * 10), (base[0] + uy * 10, base[1] - ux * 10), end]
            page.draw.polygon(triangle, fill='#F2F1E8')
            page.draw.line(triangle, fill=ACCENT, width=3)
        else:
            page.arrow(start, end, dashed=edge['kind'] == 'calls')
            if edge['kind'] == 'contains':
                diamond = [start, (start[0] + ux * 12 - uy * 8, start[1] + uy * 12 + ux * 8), (start[0] + ux * 24, start[1] + uy * 24), (start[0] + ux * 12 + uy * 8, start[1] + uy * 12 - ux * 8), start]
                page.draw.polygon(diamond, fill='#F2F1E8')
                page.draw.line(diamond, fill=ACCENT, width=3)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        if 'label_box' in edge:
            lx, ly, lw, lh = edge['label_box']
            label_box = (x + lx * w, y + ly * h, lw * w, lh * h)
        elif horizontal:
            label_box = (min(start[0], end[0]) + 4, my - 60, abs(start[0] - end[0]) - 8, 55)
        else:
            label_box = (mx + 18, my - 35, min(310, x + w - mx - 18), 75)
        labels.append((f'{step_marker(edge.get("number", edge_index))} {edge["label"]}', label_box))
    for node in visual['nodes']:
        nx, ny, nw, nh = boxes[node['id']]
        page.draw.rounded_rectangle((nx, ny, nx + nw, ny + nh), radius=14, fill=PALE, outline='#C4CFBA', width=2)
        body = node.get('body')
        title_height = nh * .53 if body else nh - 20
        page.text(
            node['title'],
            (nx + 14, ny + 10, nw - 28, title_height),
            size=29,
            minimum=18,
            color=ACCENT,
            center=True,
            preserve_tokens=True,
        )
        if body:
            page.text(body, (nx + 14, ny + 14 + title_height, nw - 28, nh - title_height - 22), size=24, minimum=19, center=True)
    for label, box in labels:
        lx, ly, lw, lh = box
        page.draw.rounded_rectangle(
            (lx + min(20, lw * .18), ly + 4, lx + lw - min(20, lw * .18), ly + lh - 4),
            radius=8,
            fill='#F2F1E8',
        )
        page.text(
            label,
            box,
            size=24,
            minimum=16,
            color=MUTED,
            center=True,
            preserve_tokens=True,
        )


def draw_code(page, visual, bounds):
    """NOTE 代码区保持浅色；编号注释直接解释步骤，脱离左栏也能读懂。"""
    x, y, w, h = bounds
    page.draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill='#E5EBDD')
    lines = visual['text'].splitlines()
    row = min(48, (h - 32) / max(1, len(lines)))
    original_font = page.font_path
    mono_font = Path(page.font_path).with_name('NotoSansMonoCJK-Regular.ttc')
    if mono_font.is_file():
        page.font_path = str(mono_font)
    for i, line in enumerate(lines, 1):
        top = y + 16 + (i - 1) * row
        is_explanation = bool(re.match(r'\s*#\s*[①②③④⑤⑥]', line))
        if is_explanation:
            page.draw.rectangle(
                (x + 8, top - 2, x + w - 8, top + row), fill='#F0EBDD'
            )
            page.draw.rectangle((x + 8, top - 2, x + 14, top + row), fill='#A78555')
        if i in visual.get('highlight_lines', []):
            page.draw.rectangle(
                (x + 14, top - 2, x + w - 8, top + row), fill='#E7EEE1'
            )
        page.text(
            f'{i:2d}  {line}',
            (x + 16, top, w - 32, row),
            size=min(28, int(row / 1.45)),
            minimum=17,
            color='#6F5537' if is_explanation else INK,
        )
    page.font_path = original_font


def render_slide(slide, lesson, index, target, font_path, lesson_dir, source_labels, timing=None):
    page = Canvas(font_path)
    draw = page.draw
    draw.rectangle((0, 0, WIDTH, 14), fill=ACCENT)
    page.text('SGLang  /  QUICK LEARN', (80, 45, 1300, 42), size=26, color=ACCENT)
    page.text(f'{index + 1:02d} / {len(lesson["slides"]):02d}', (1660, 45, 180, 42), size=25, color=MUTED)
    page.text(slide['title'], (80, 108, 1760, 90), size=56, minimum=38)
    draw.line((80, 222, 1840, 222), fill='#D2D9C9', width=2)

    bullets = slide['bullets']
    layout = slide.get('layout', 'split')
    if layout == 'wide_visual':
        if slide.get('code_excerpt'):
            raise ValueError('wide_visual 页面请把源码作为主图，避免图、代码与说明争抢空间')
        columns = 2 if len(bullets) <= 4 else 3
        rows = math.ceil(len(bullets) / columns)
        cell_w = 1760 / columns
        row_height = min(68, 135 / rows)
        for i, bullet in enumerate(bullets):
            column, row = i % columns, i // columns
            left, top = 82 + column * cell_w, 772 + row * (row_height + 8)
            explicit = re.match(r'^([①②③④⑤⑥⑦⑧⑨])\s*', bullet)
            marker = explicit.group(1) if explicit else step_marker(i + 1)
            content = bullet[explicit.end():] if explicit else bullet
            page.text(marker, (left, top, 36, row_height), size=26, minimum=22, color=ACCENT)
            page.text(content, (left + 40, top, cell_w - 54, row_height), size=27, minimum=21)
        x, y, w, h = 80, 245, 1760, 500
    else:
        compact_bullets = bool(slide.get('code_excerpt')) and len(bullets) > 2
        if slide.get('code_excerpt'):
            row_height = 112 if len(bullets) <= 2 else max(48, 250 // len(bullets))
        else:
            row_height = min(150, 600 // len(bullets))
        for i, bullet in enumerate(bullets):
            top = 265 + i * row_height
            explicit = re.match(r'^([①②③④⑤⑥⑦⑧⑨])\s*', bullet)
            marker = explicit.group(1) if explicit else step_marker(i + 1)
            content = bullet[explicit.end():] if explicit else bullet
            page.text(marker, (82, top, 36, row_height - 8), size=27 if compact_bullets else 29, minimum=23, color=ACCENT)
            page.text(content, (122, top, 592, row_height - 8), size=29 if compact_bullets else 32, minimum=23 if compact_bullets else 26)

        if slide.get('code_excerpt'):
            code_top = 520 if len(bullets) <= 2 else 265 + len(bullets) * row_height + 16
            draw_code(page, slide['code_excerpt'], (84, code_top, 620, 880 - code_top))
        x, y, w, h = 735, 260, 1105, 620

    visual = slide['visual']
    kind = visual['type']
    if kind == 'relations':
        draw_relations(page, visual, (x, y, w, h))
    elif kind == 'cards':
        items = visual['items']
        columns = 2 if len(items) > 1 else 1
        rows = math.ceil(len(items) / columns)
        cw, ch = (w - (columns - 1) * 20) / columns, (h - (rows - 1) * 20) / rows
        for i, item in enumerate(items):
            page.card(item['title'], item['body'], (x + (i % columns) * (cw + 20), y + (i // columns) * (ch + 20), cw, ch))
    elif kind == 'flow':
        steps = visual['steps']
        gap = 36
        height = (h - gap * (len(steps) - 1)) / len(steps)
        for i, step in enumerate(steps):
            top = y + i * (height + gap)
            draw.rounded_rectangle((x, top, x + w, top + height), radius=14, fill=PALE, outline='#C4CFBA', width=2)
            page.text(step_marker(i + 1), (x + 16, top + 16, 76, height - 32), size=34, color=ACCENT, center=True)
            if len(steps) == 4:
                page.text(step['title'], (x + 112, top + 12, 390, height - 24), size=29, minimum=22, color=ACCENT)
                page.text(step['body'], (x + 532, top + 12, w - 552, height - 24), size=24, minimum=21)
            else:
                page.text(step['title'], (x + 112, top + 12, w - 136, 48), size=31, minimum=24, color=ACCENT)
                page.text(step['body'], (x + 112, top + 60, w - 136, height - 65), size=24, minimum=21)
            if i + 1 < len(steps):
                page.arrow((x + w / 2, top + height + 4), (x + w / 2, top + height + gap - 4))
                page.text(
                    f'{step_marker(i + 1)}→{step_marker(i + 2)}',
                    (x + w / 2 + 22, top + height + 3, 80, gap - 4),
                    size=18,
                    minimum=16,
                    color=ACCENT,
                    center=True,
                )
    elif kind == 'sequence':
        participants = visual['participants']
        box_width = min(230, (w - 20) / len(participants) - 10)
        centers = {
            p['id']: x + box_width / 2 + i * (w - box_width) / (len(participants) - 1)
            for i, p in enumerate(participants)
        }
        for part in participants:
            cx = centers[part['id']]
            draw.rounded_rectangle((cx - box_width / 2, y, cx + box_width / 2, y + 70), radius=10, fill=PALE)
            page.text(part['label'], (cx - box_width / 2 + 8, y + 8, box_width - 16, 54), size=25, minimum=17, center=True, preserve_tokens=True)
            for yy in range(int(y + 80), int(y + h), 18):
                draw.line((cx, yy, cx, yy + 9), fill='#BBC8B1', width=2)
        step_height = (h - 96) / len(visual['messages'])
        for i, message in enumerate(visual['messages']):
            yy = y + 116 + (i + .65) * step_height
            start, end = centers[message['from']], centers[message['to']]
            is_return = message.get('return', False)
            color = MUTED if is_return else ACCENT
            if start == end:
                edge = min(start + 80, x + w - 8)
                draw.line((start, yy - 14, edge, yy - 14, edge, yy), fill=color, width=3)
                page.arrow((edge, yy), (start, yy), color, is_return)
                label_x, label_w = x + 15, w - 30
            else:
                page.arrow((start, yy), (end, yy), color, is_return)
                label_x, label_w = min(start, end) + 10, abs(end - start) - 20
            page.text(
                f'{step_marker(i + 1)} {message["label"]}',
                (label_x, yy - step_height + 7, label_w, step_height - 9),
                size=24,
                minimum=16,
                color=color,
                center=True,
                preserve_tokens=True,
            )
    elif kind == 'code':
        draw_code(page, visual, (x, y, w, h))
    elif kind == 'image':
        with Image.open(lesson_dir / visual['path']) as source:
            source = ImageOps.contain(source.convert('RGB'), (w, h))
            page.image.paste(source, (x + (w - source.width) // 2, y + (h - source.height) // 2))
    # 画面底部不再重复来源、章节和时长；来源保存在独立 source-index.md。
    # 880 以下留给视频字幕，避免烧录字幕遮住主图。
    page.image.save(target)
    return page.image


def contact_sheet(paths, target):
    columns = min(3, len(paths))
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new('RGB', (columns * 640, rows * 360), '#D2D9C9')
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            sheet.paste(im.resize((640, 360)), ((i % columns) * 640, (i // columns) * 360))
    sheet.save(target)


def slide_document(paths, target, title):
    """NOTE 将同一套页面另存为可直接翻阅的幻灯片文件。"""
    pages = []
    try:
        for path in paths:
            with Image.open(path) as source:
                pages.append(source.convert('RGB'))
        pages[0].save(target, save_all=True, append_images=pages[1:], resolution=144.0, title=title)
    finally:
        for page in pages:
            page.close()
