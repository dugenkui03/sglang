"""NOTE 为每次视频制作创建独立任务，任务记录与产物均留在忽略目录内。"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks'
BEGIN = '<!-- execution:start -->'
END = '<!-- execution:end -->'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def create_task(name, title, request=''):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', name):
        raise ValueError('任务名称须为一至六十四位小写字母、数字、连字符或下划线')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    identifier = f'{stamp}-{name}-{uuid4().hex[:8]}'
    directory = TASKS / identifier
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'outputs').mkdir()
    record = {'version': 1, 'task_id': identifier, 'title': title, 'created_at': now(),
              'request': request, 'status': 'planning', 'runs': []}
    write_json(directory / 'task.json', record)
    write_json(directory / 'lesson.json', {'version': 1, 'id': identifier,
               'title': title, 'subtitle': '', 'slides': []})
    template = (ROOT / 'templates/TASK.md').read_text(encoding='utf-8')
    for key, value in {'TASK_ID': identifier, 'TITLE': title, 'CREATED_AT': record['created_at'],
                       'REQUEST': request or '待记录本次用户要求与讨论结论。'}.items():
        template = template.replace('{{' + key + '}}', value)
    (directory / 'TASK.md').write_text(template, encoding='utf-8')
    return directory


class TaskExecution:
    """NOTE 一份课程绑定一个任务；同一任务同时只允许一个构建进程。"""

    def __init__(self, lesson_path):
        self.lesson_path = Path(lesson_path).resolve()
        self.directory = self.lesson_path.parent
        if self.directory.parent != TASKS.resolve() or self.lesson_path.name != 'lesson.json':
            raise ValueError('课程必须位于 tasks/<任务ID>/lesson.json；请先运行 scripts/task.py')
        self.task_id = self.directory.name
        self.record_path = self.directory / 'task.json'
        self.execution_file = self.directory / 'TASK.md'
        self.lock = None
        self.run = None
        self._read_record()

    def _read_record(self):
        if not self.record_path.is_file() or not self.execution_file.is_file():
            raise ValueError('任务缺少 TASK.md 或 task.json，请先建立独立任务')
        self.record = json.loads(self.record_path.read_text(encoding='utf-8'))
        if self.record.get('task_id') != self.task_id:
            raise ValueError('任务标识与目录不符，拒绝混用任务记录')
        content = self.execution_file.read_text(encoding='utf-8')
        if content.count(BEGIN) != 1 or content.count(END) != 1 or content.index(BEGIN) > content.index(END):
            raise ValueError('TASK.md 的执行记录标记缺失或重复，请恢复模板标记')

    def __enter__(self):
        self.lock = (self.directory / '.build.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._read_record()
        except Exception:
            self.lock.close()
            raise RuntimeError('任务记录无效或已有构建正在运行；禁止混写同一任务')
        return self

    def output_path(self, requested=None):
        base = self.directory / 'outputs'
        if base.resolve() != base:
            raise ValueError('任务产物目录不能链接到其他位置')
        output = (Path(requested).expanduser() if requested else base / 'full').resolve()
        if not output.is_relative_to(base) or output == base:
            raise ValueError('输出必须位于本任务的 outputs/ 子目录，不能写入其他任务或公共目录')
        return output

    def start(self, args):
        self.run = {'id': uuid4().hex[:12], 'started_at': now(), 'status': 'building',
                    'options': {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
        self.record['runs'].append(self.run)
        self.update({'status': 'building'})

    def update(self, manifest):
        if self.run is None:
            return
        self.run['status'] = manifest['status']
        self.run['updated_at'] = now()
        if 'slides' in manifest:
            self.run['completed_segments'] = sum(len(p['segments']) for p in manifest['slides'])
        for key in ('output', 'lesson_sha256', 'repository_revision', 'duration_seconds', 'error', 'speech'):
            if key in manifest:
                self.run[key] = manifest[key]
        self.record['status'] = self.run['status']
        write_json(self.record_path, self.record)
        lines = [f'- 当前状态：`{self.record["status"]}`', f'- 最近更新：{now()}', '',
                 '| 执行 | 开始时间 | 状态 | 已完成配音段数 | 产物目录 |', '|---|---|---|---:|---|']
        for run in self.record['runs']:
            output = run.get('output')
            link = f'[{Path(output).name}]({Path(output).relative_to(self.directory).as_posix()}/)' if output else '尚未生成'
            lines.append(f'| {run["id"]} | {run["started_at"]} | {run["status"]} | {run.get("completed_segments", 0)} | {link} |')
        if self.run.get('error'):
            lines += ['', '- 最近错误：' + self.run['error'].replace('\n', ' ')]
        content = self.execution_file.read_text(encoding='utf-8')
        before, rest = content.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        temporary = self.execution_file.with_suffix('.md.tmp')
        temporary.write_text(before + BEGIN + '\n\n' + '\n'.join(lines) + '\n\n' + END + after, encoding='utf-8')
        temporary.replace(self.execution_file)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None and self.run is not None:
                self.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                             'error': str(exc) or '构建中断；已生成的本任务片段可复用。'})
        finally:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='任务短名，每次调用仍会创建新的唯一任务')
    parser.add_argument('--title', required=True, help='视频主题')
    parser.add_argument('--request', default='', help='本次用户要求；后续在 TASK.md 中补充方案与进度')
    args = parser.parse_args()
    try:
        print(create_task(args.name, args.title, args.request))
    except (OSError, ValueError) as exc:
        print(f'创建任务失败：{exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
