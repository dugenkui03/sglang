"""NOTE 将经源码核对的课程、逐段配音和字幕合成为学习视频。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from task import TaskExecution

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def source_info(source):
    """【Step 1】以当前文件定位符号，不复用文档里的旧行号。"""
    path = (REPO / source['path']).resolve()
    require(path.is_relative_to(REPO) and path.is_file(), f'无效源码路径：{path}')
    symbol = source.get('symbol', '')
    line = 1
    if symbol:
        nodes = ast.parse(path.read_text(encoding='utf-8')).body
        for name in symbol.split('.'):
            found = next((node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name), None)
            require(found is not None, f'源码符号不存在：{source["path"]}:{symbol}')
            line, nodes = found.lineno, found.body
    return {'path': str(path.relative_to(REPO)), 'symbol': symbol, 'line': line, 'sha256': sha256(path)}


def validate(lesson, lesson_dir):
    require(lesson.get('version') == 1, '课程 version 必须为 1')
    require(nonempty(lesson.get('title')), '缺少课程标题')
    require(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', lesson.get('id', '')), '课程 id 只能含小写字母、数字、连字符和下划线')
    slides = lesson.get('slides', [])
    require(isinstance(slides, list) and slides, '课程至少包含一页')
    identifiers = set()
    for slide in slides:
        sid = slide.get('id', '')
        require(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', sid) and sid not in identifiers, f'无效或重复页名：{sid}')
        identifiers.add(sid)
        require(nonempty(slide.get('title')), f'{sid} 缺少标题')
        require(1 <= len(slide.get('bullets', [])) <= 6 and all(nonempty(x) for x in slide['bullets']), f'{sid} 需要 1–6 条重点')
        if slide.get('code_excerpt'):
            excerpt = slide['code_excerpt'].get('text', '')
            require(nonempty(excerpt) and len(excerpt.splitlines()) <= 12,
                    f'{sid} 方法旁注最多十二行代码')
        require(slide.get('sources'), f'{sid} 缺少源码来源')
        require(slide.get('narration'), f'{sid} 缺少讲稿')
        for segment in slide['narration']:
            require(nonempty(segment.get('text')) and nonempty(segment.get('spoken_text', segment['text'])), f'{sid} 存在空讲稿')
        visual = slide.get('visual', {})
        kind = visual.get('type')
        require(kind in ('cards', 'flow', 'sequence', 'code', 'image', 'relations'), f'{sid} 图示类型无效')
        if kind in ('cards', 'flow'):
            items = visual.get('items' if kind == 'cards' else 'steps', [])
            require((1 if kind == 'cards' else 2) <= len(items) <= 4, f'{sid} 图示节点数量无效')
            require(all(nonempty(x.get('title')) and nonempty(x.get('body')) for x in items), f'{sid} 图示文字不完整')
        elif kind == 'sequence':
            participants = visual.get('participants', [])
            ids = [p['id'] for p in participants]
            require(2 <= len(ids) <= 5 and len(set(ids)) == len(ids), f'{sid} 时序参与者数量无效或重复')
            require(all(nonempty(p.get('label')) for p in participants), f'{sid} 时序参与者缺少名称')
            messages = visual.get('messages', [])
            require(1 <= len(messages) <= 6, f'{sid} 时序消息需要 1–6 条')
            require(all(m.get('from') in ids and m.get('to') in ids and nonempty(m.get('label')) for m in messages), f'{sid} 时序消息引用无效')
        elif kind == 'relations':
            nodes = visual.get('nodes', [])
            ids = [node['id'] for node in nodes]
            require(2 <= len(nodes) <= 6 and len(ids) == len(set(ids)), f'{sid} 关系图需要 2–6 个唯一节点')
            for node in nodes:
                require(nonempty(node.get('title')), f'{sid} 关系图节点缺少标题')
                require(0 <= node['x'] < 1 and 0 <= node['y'] < 1 and node['w'] > 0 and node['h'] > 0 and node['x'] + node['w'] <= 1.001 and node['y'] + node['h'] <= 1.001, f'{sid} 关系图节点超出绘图区')
            require(visual.get('edges'), f'{sid} 关系图缺少连线')
            for edge in visual['edges']:
                require(edge.get('from') in ids and edge.get('to') in ids and edge['from'] != edge['to'] and nonempty(edge.get('label')), f'{sid} 关系图连线无效')
                require(edge.get('kind') in ('flow', 'contains', 'inherits', 'calls'), f'{sid} 关系图连线类型无效')
        elif kind == 'code':
            require(nonempty(visual.get('text')) and len(visual['text'].splitlines()) <= 16, f'{sid} 代码需要 1–16 行')
        else:
            require((lesson_dir / visual.get('path', '')).is_file(), f'{sid} 图片不存在')
    for first in range(0, len(slides) - 1):
        pair = slides[first:first + 2]
        require(
            any(page['visual']['type'] == 'code' or page.get('code_excerpt') for page in pair),
            f'第 {first + 1}–{first + 2} 页至少需要一页源码讲解',
        )


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def subtitle_text(text, font_path):
    from PIL import ImageFont
    from render import wrapped

    # NOTE 字幕整段显示，时间取该段实际配音；不伪造逐字对齐。
    lines = wrapped(text, ImageFont.truetype(str(font_path), 38), 1710)
    require(len(lines) <= 2, f'字幕超过两行，请拆分讲稿：{text}')
    return '\n'.join(lines)


def narration_markdown(lesson):
    """NOTE 讲稿只包含可讲述内容；源码证据和制作计划分别保存。"""
    result = [f'# {lesson["title"]}', '']
    for i, slide in enumerate(lesson['slides'], 1):
        result += [f'## {i}. {slide["title"]}', '']
        for j, segment in enumerate(slide['narration'], 1):
            result += [f'<!-- tts:{slide["id"]}:{j:03d} -->', segment['text'], '']
    return '\n'.join(result)


def source_index(lesson, sources):
    result = [f'# {lesson["title"]}：源码索引', '', '- 本文件保存讲解依据；口播内容见 narration.md。', '']
    for i, slide in enumerate(lesson['slides'], 1):
        result += [f'## {i}. {slide["title"]}', '']
        result += [f'- [{s["symbol"] or Path(s["path"]).name}]({REPO / s["path"]}:{s["line"]}) — `{s["path"]}`' for s in sources[i - 1]]
        if slide.get('diagram_source'):
            result += [f'- 图示底图：{slide["diagram_source"]}']
        result += ['']
    return '\n'.join(result)


def time_plan(lesson, durations=None):
    result, position = [], 0
    for i, slide in enumerate(lesson['slides']):
        estimate = float(slide.get('estimated_seconds', sum(len(s.get('spoken_text', s['text'])) for s in slide['narration']) / 4.3 + len(slide['narration']) * .2))
        duration = durations[i] if durations is not None else estimate
        require(duration > 0, f'{slide["id"]} 的讲解时长必须大于零')
        result.append({'section': slide.get('section', '学习回顾'), 'start': position, 'end': position + duration, 'seconds': duration, 'estimated_seconds': estimate, 'actual': durations is not None})
        position += duration
    for row in result:
        row['total'] = position
    return result


def minute_label(seconds):
    value = round(seconds)
    return f'{value // 60:02d}:{value % 60:02d}'


def storyboard(lesson, timing):
    result = [f'# {lesson["title"]}：章节与节奏', '', f'- 共 {len(timing)} 页，预计总时长 {sum(t["estimated_seconds"] for t in timing) / 60:.1f} 分钟。']
    if timing[0]['actual']:
        result += [f'- 实际中文配音成片时长：{minute_label(timing[0]["total"])}。']
    else:
        result += ['- 当前为讲解计划；配音完成后以真实音频时长校准。']
    result += ['- 按章节衔接方法、对象与数据变化；复杂方法独立展开，章节末回到请求主线。', '', '| 页 | 章节 | 本页讲解重点 | 预计秒数 | 本次时间段 |', '|---|---|---|---:|---|']
    for i, (slide, row) in enumerate(zip(lesson['slides'], timing), 1):
        result += [f'| {i} | {row["section"]} | {slide["title"]} | {row["estimated_seconds"]:.0f} | {minute_label(row["start"])}–{minute_label(row["end"])} |']
    return '\n'.join(result) + '\n'


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def build(args, task):
    config_path = ROOT / 'config.local.json'
    require(config_path.is_file(), '请先运行 scripts/setup_omnivoice.py 完成本机安装')
    config = json.loads(config_path.read_text())
    font = Path(config['font'])
    require(font.is_file(), f'中文字体不存在：{font}')
    lesson_path = args.lesson.expanduser()
    if not lesson_path.is_absolute():
        lesson_path = ROOT / lesson_path
    lesson_path = lesson_path.resolve()
    lesson = json.loads(lesson_path.read_text(encoding='utf-8'))
    validate(lesson, lesson_path.parent)
    require(lesson['id'] == task.task_id, '课程标识与任务不符，不能混用其他任务的课程')
    if args.limit_slides is not None:
        require(args.limit_slides > 0, '--limit-slides 必须为正整数')
        require(args.output_dir is not None, '短样片需用 --output-dir 指定独立输出目录')
        lesson['slides'] = lesson['slides'][:args.limit_slides]
    output = task.output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # NOTE 旧成片单独保留，避免本次只渲染或失败后仍把旧视频当成新结果。
    previous_media = ('video.mp4', 'narration.wav', 'subtitles.srt', 'slides.ffconcat')
    previous_tracks = list((output / 'audio').glob('*-full.wav'))
    if any((output / name).exists() for name in previous_media) or previous_tracks:
        previous = output / 'previous' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous.mkdir(parents=True)
        for filename in previous_media:
            if (output / filename).is_file():
                (output / filename).replace(previous / filename)
        if previous_tracks:
            (previous / 'audio').mkdir()
            for track in previous_tracks:
                track.replace(previous / 'audio' / track.name)
        for filename in ('manifest.json', 'lesson.json', 'narration.md'):
            if (output / filename).is_file():
                shutil.copy2(output / filename, previous / filename)
    (output / 'slides').mkdir(exist_ok=True)
    (output / 'audio').mkdir(exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = {'status': 'building', 'created_at': datetime.now(timezone.utc).isoformat(), 'lesson': str(lesson_path), 'lesson_sha256': sha256(lesson_path), 'lesson_id': lesson['id'], 'limit_slides': args.limit_slides, 'slides': [], 'quality': {'manual_listening': 'pending', 'note': '技术检查不等于听感、读音与内容验收。'}}
    manifest.update(task_id=task.task_id, task_file=str(task.execution_file), output=str(output))

    def publish_manifest():
        save_json(manifest_path, manifest)
        task.update(manifest)
    try:
        manifest['repository_revision'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        manifest['repository_revision'] = None
    publish_manifest()
    try:
        from render import contact_sheet, render_slide, slide_document

        # 【Step 2】先完成排版和来源检查，再投入较慢的模型合成。
        sources = [[source_info(s) for s in slide['sources']] for slide in lesson['slides']]
        timing = time_plan(lesson)
        manifest['timing_plan'] = timing
        paths = []
        expected_names = {f'{i + 1:02d}-{slide["id"]}.png' for i, slide in enumerate(lesson['slides'])}
        for previous_slide in (output / 'slides').glob('*.png'):
            if re.fullmatch(r'\d+-[a-z0-9_-]+\.png', previous_slide.name) and previous_slide.name not in expected_names:
                stale = output / 'previous' / 'slides'
                stale.mkdir(parents=True, exist_ok=True)
                previous_slide.replace(stale / previous_slide.name)
        for i, slide in enumerate(lesson['slides']):
            target = output / 'slides' / f'{i + 1:02d}-{slide["id"]}.png'
            labels = [f'{Path(s["path"]).name}:{s["line"]} {s["symbol"]}' for s in sources[i]]
            render_slide(slide, lesson, i, target, font, lesson_path.parent, labels, timing[i])
            paths.append(target)
            for segment in slide['narration']:
                subtitle_text(segment['text'], font)
            manifest['slides'].append({'id': slide['id'], 'image': str(target.relative_to(output)), 'sources': sources[i], 'segments': []})
        contact_sheet(paths, output / 'contact-sheet.png')
        slide_document(paths, output / 'slides.pdf', lesson['title'])
        (output / 'narration.md').write_text(narration_markdown(lesson), encoding='utf-8')
        (output / 'source-index.md').write_text(source_index(lesson, sources), encoding='utf-8')
        (output / 'storyboard.md').write_text(storyboard(lesson, timing), encoding='utf-8')
        save_json(output / 'lesson.json', lesson)
        publish_manifest()
        print(f'已渲染 {len(paths)} 页：{output / "contact-sheet.png"}', flush=True)
        if args.slides_only:
            manifest['status'] = 'slides_only'
            publish_manifest()
            return output

        for executable in ('ffmpeg', 'ffprobe'):
            require(shutil.which(executable), f'缺少程序：{executable}')
        import numpy as np
        import soundfile as sf
        import speech
        from parallel_speech import available_cpu_count, synthesize_jobs

        runtime_dir = Path(config['runtime_dir'])
        budget = args.threads if args.threads is not None else config.get('threads', 8)
        workers = args.workers
        require(budget > 0 and workers > 0, '--threads 与 --workers 必须为正整数')
        require(args.speech_retries >= 0, '--speech-retries 不能为负数')
        require(budget <= available_cpu_count(), '配音总线程预算超过当前进程有效配额')
        require(workers <= budget, '配音工作进程数不能超过总线程预算')
        segment_count = sum(len(slide['narration']) for slide in lesson['slides'])
        workers = min(workers, segment_count)
        threads = max(1, budget // workers)
        signature = speech.engine_signature(runtime_dir)
        manifest['speech'] = {'backend': config.get('backend', 'audio8'), 'runtime_dir': str(runtime_dir),
                              'signature': signature, 'threads': threads, 'thread_budget': budget,
                              'workers': workers, 'retries': args.speech_retries,
                              'total_segments': segment_count, 'completed_segments': 0,
                              'cached_segments': 0, 'segments': {}}
        jobs, pending, records = [], [], {}
        # NOTE 先收集全部句子与有效缓存；生成顺序不改变讲稿与字幕顺序。
        for i, slide in enumerate(lesson['slides']):
            for j, segment in enumerate(slide['narration']):
                text = segment.get('spoken_text', segment['text'])
                key = f'{slide["id"]}:{j + 1:03d}'
                cache_key = hashlib.sha256(json.dumps({'text': text, 'engine': signature, 'seed': 42,
                                                       'threads': threads}, sort_keys=True).encode()).hexdigest()
                audio_path = output / 'audio' / f'{slide["id"]}-{j + 1:02d}-{cache_key[:12]}.wav'
                require(audio_path.resolve().is_relative_to(output), '配音路径不能链接到任务目录外')
                job = {'key': key, 'cache_key': cache_key, 'text': text, 'path': str(audio_path),
                       'seed': 42, 'slide_index': i, 'segment_index': j}
                jobs.append(job)
                records[key] = {'segment_index': j + 1, 'text': segment['text'], 'spoken_text': text,
                                'audio': str(audio_path.relative_to(output))}
                meta_path, metadata = audio_path.with_suffix('.json'), None
                if not args.force_audio and audio_path.is_file() and meta_path.is_file():
                    try:
                        candidate = json.loads(meta_path.read_text(encoding='utf-8'))
                        if candidate.get('cache_key') == cache_key and candidate.get('audio_sha256') == sha256(audio_path):
                            audio, rate = sf.read(audio_path, dtype='float32')
                            if audio.ndim == 1 and audio.size and np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 1e-5:
                                metadata = candidate
                    except (ValueError, OSError, RuntimeError):
                        pass
                if metadata is None:
                    pending.append(job)
                    manifest['speech']['segments'][key] = {'status': 'pending'}
                else:
                    records[key].update(status='cached', speech_metrics=metadata)
                    manifest['slides'][i]['segments'].append(records[key])
                    manifest['speech']['segments'][key] = {'status': 'cached'}
                    manifest['speech']['completed_segments'] += 1
                    manifest['speech']['cached_segments'] += 1
                    print(f'复用配音：{key}', flush=True)
        jobs_by_key = {job['key']: job for job in jobs}

        def speech_progress(event):
            key = event['key']
            manifest['speech']['segments'][key] = {k: v for k, v in event.items() if k != 'metadata'}
            if event['status'] == 'complete':
                record = records[key]
                record.update(status='complete', speech_metrics=event['metadata'])
                page = manifest['slides'][jobs_by_key[key]['slide_index']]
                page['segments'].append(record)
                page['segments'].sort(key=lambda entry: entry['segment_index'])
                manifest['speech']['completed_segments'] += 1
            publish_manifest()

        manifest['status'] = 'synthesizing'
        publish_manifest()
        if pending:
            require(config.get('backend') == 'omnivoice', '请运行 setup_omnivoice.py；Audio8 当前未通过可用性检查')
            synthesize_jobs(pending, runtime_dir, threads=threads, workers=workers,
                            retries=args.speech_retries, on_progress=speech_progress)
        manifest['status'] = 'assembling'
        publish_manifest()
        sample_rate = None
        timeline = 0
        subtitles, tracks, durations = [], [], []
        for i, slide in enumerate(lesson['slides']):
            chunks = []
            slide_start = timeline
            for j, segment in enumerate(slide['narration']):
                text = segment.get('spoken_text', segment['text'])
                record = records[f'{slide["id"]}:{j + 1:03d}']
                audio_path = output / record['audio']
                audio, rate = sf.read(audio_path, dtype='float32')
                require(audio.ndim == 1 and audio.size > 0 and np.isfinite(audio).all(), f'音频无效：{audio_path}')
                require(float(np.max(np.abs(audio))) > 1e-5, f'音频为静音：{audio_path}')
                if sample_rate is None:
                    sample_rate = rate
                require(rate == sample_rate, '配音采样率不一致，请重新生成')
                gap = np.zeros(round(rate * (0.2 if j else 0.15)), dtype=np.float32)
                chunks += [gap, audio]
                timeline += gap.size / rate
                start = timeline
                timeline += audio.size / rate
                subtitles.append(f'{len(subtitles) + 1}\n{timestamp(start)} --> {timestamp(timeline)}\n{subtitle_text(segment["text"], font)}\n')
                record.update(start_seconds=start, end_seconds=timeline)
                publish_manifest()
            tail = np.zeros(round(sample_rate * 0.3), dtype=np.float32)
            chunks.append(tail)
            timeline += tail.size / sample_rate
            track = np.concatenate(chunks)
            tracks.append(track)
            durations.append(track.size / sample_rate)
            sf.write(output / 'audio' / f'{slide["id"]}-full.wav', track, sample_rate, subtype='PCM_16')
            manifest['slides'][i].update(start_seconds=slide_start, end_seconds=timeline)

        # 【Step 3】同一套真实采样长度同时驱动字幕和画面切换。
        timing = time_plan(lesson, durations)
        manifest['timing_actual'] = timing
        for i, slide in enumerate(lesson['slides']):
            labels = [f'{Path(s["path"]).name}:{s["line"]} {s["symbol"]}' for s in sources[i]]
            render_slide(slide, lesson, i, paths[i], font, lesson_path.parent, labels, timing[i])
        contact_sheet(paths, output / 'contact-sheet.png')
        slide_document(paths, output / 'slides.pdf', lesson['title'])
        (output / 'storyboard.md').write_text(storyboard(lesson, timing), encoding='utf-8')
        sf.write(output / 'narration.wav', np.concatenate(tracks), sample_rate, subtype='PCM_16')
        (output / 'subtitles.srt').write_text('\n'.join(subtitles), encoding='utf-8')
        concat = ['ffconcat version 1.0']
        for path, duration in zip(paths, durations):
            concat += [f"file '{path.relative_to(output).as_posix()}'", f'duration {duration:.9f}']
        concat += [f"file '{paths[-1].relative_to(output).as_posix()}'"]
        (output / 'slides.ffconcat').write_text('\n'.join(concat) + '\n', encoding='utf-8')
        filters = 'fps=24'
        if not args.no_burn_subtitles:
            # 红色字幕由用户指定；浅米色描边与护眼页面保持协调并确保可读。
            filters += ",subtitles=subtitles.srt:force_style='FontName=Noto Sans CJK SC,FontSize=10,PrimaryColour=&H00323CD2,OutlineColour=&H00E8F1F2,BorderStyle=1,Outline=2,Shadow=0,MarginV=18'"
        temporary_video = output / '.video-building.mp4'
        print(f'合成视频，共 {timeline:.1f} 秒', flush=True)
        run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-threads', '4', '-filter_threads', '4', '-f', 'concat', '-safe', '0', '-i', 'slides.ffconcat', '-i', 'narration.wav', '-vf', filters, '-t', f'{timeline:.9f}', '-c:v', 'libx264', '-threads', '4', '-preset', 'veryfast', '-tune', 'stillimage', '-crf', '22', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-movflags', '+faststart', temporary_video.name], cwd=output)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(temporary_video)], text=True))
        require(any(s['codec_type'] == 'audio' for s in probe['streams']), '成片缺少声音')
        require(any(s['codec_type'] == 'video' and s['width'] == 1920 and s['height'] == 1080 for s in probe['streams']), '成片分辨率错误')
        require(abs(float(probe['format']['duration']) - timeline) < 0.15, '成片与讲稿时长不一致')
        temporary_video.replace(output / 'video.mp4')
        manifest.update(status='complete', duration_seconds=timeline, video_sha256=sha256(output / 'video.mp4'), subtitles_burned=not args.no_burn_subtitles, video_probe=probe)
        publish_manifest()
        return output
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc) or '用户或调用进程中断；已完成配音保留为缓存。')
        publish_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lesson', type=Path, help='tasks/<任务ID>/lesson.json，可用绝对路径')
    parser.add_argument('--slides-only', action='store_true')
    parser.add_argument('--threads', type=int, help='配音总线程预算，默认 8；自动分给工作进程')
    parser.add_argument('--workers', type=int, default=2, help='配音工作进程数，默认 2，每进程只加载一次模型')
    parser.add_argument('--speech-retries', type=int, default=1, help='失败片段额外重试次数，默认 1')
    parser.add_argument('--force-audio', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-burn-subtitles', action='store_true')
    parser.add_argument('--limit-slides', type=int, help='只生成前几页；需同时用 --output-dir 指定独立输出目录')
    args = parser.parse_args()
    try:
        lesson_path = args.lesson.expanduser()
        if not lesson_path.is_absolute():
            lesson_path = ROOT / lesson_path
        with TaskExecution(lesson_path) as task:
            task.start(args)
            output = build(args, task)
    except KeyboardInterrupt:
        print('生成已中断，已完成的逐段配音可以复用。', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'生成失败：{exc}', file=sys.stderr)
        sys.exit(1)
    print(f'产物目录：{output}', flush=True)


if __name__ == '__main__':
    main()
