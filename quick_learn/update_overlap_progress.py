#!/usr/bin/env python3
"""Preview the Overlap reading route; --write saves it, --check checks freshness."""

import argparse
import subprocess
import sys
from collections import Counter

if __package__:
    from .update_learn_progress import ROOT, counts, measure, percentage, validate_scope
else:
    from update_learn_progress import ROOT, counts, measure, percentage, validate_scope

SCOPE_FILE = "quick_learn/LEARN.overlap.scope.json"
REPORT_FILE = "LEARN.overlap.md"
STATUS = {"D": "✅ 已读", "C": "☑️ 推定 · C", "B": "☑️ 推定 · B", "U": "⬜ 未读"}


def validate_route(scope):
    validate_scope(scope)
    nodes = {node[0]: node for branch in scope["branches"] for node in branch["nodes"]}
    if len(nodes) != scope["expected_method_count"]:
        raise ValueError(
            "Route denominator changed; update the version and scope explicitly"
        )
    listed = [
        ident for section in scope["route_sections"] for ident in section["nodes"]
    ]
    if set(listed) != nodes.keys() or any(n != 1 for n in Counter(listed).values()):
        raise ValueError(
            "Every scope method must appear exactly once in the reading route"
        )
    section_ids = [section["id"] for section in scope["route_sections"]]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Duplicate route section")
    if not set(scope["priority_nodes"]) <= nodes.keys():
        raise ValueError("Next-reading priorities must refer to route methods")
    if any("event_loop_normal" in node[2] for node in nodes.values()):
        raise ValueError("The normal scheduler loop is outside this route")
    if any(chain["id"] == "normal_schedule" for chain in scope["chains"]):
        raise ValueError("Do not infer Overlap progress through the normal-loop chain")


def measure_overlap(root=ROOT):
    report = measure(root, scope_file=SCOPE_FILE)
    validate_route(report["scope"])
    return report


def cell(value):
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def method_cells(row):
    cls, _, method = row["qualname"].rpartition(".")
    name = cell(method).replace("_", r"\_")
    return cell(cls or "—"), f"[{name}]({row['file']}#L{row['line']})"


def ordered_rows(report):
    by_id = {row["id"]: row for row in report["rows"]}
    return [
        by_id[ident]
        for section in report["scope"]["route_sections"]
        for ident in section["nodes"]
    ]


def next_rows(report, limit=10):
    by_id = {row["id"]: row for row in report["rows"]}
    priority = report["scope"]["priority_nodes"]
    order = {row["id"]: i for i, row in enumerate(ordered_rows(report))}
    rest = [ident for ident in order if ident not in priority]
    selected = [
        by_id[ident] for ident in priority + rest if by_id[ident]["status"] == "U"
    ][:limit]
    return sorted(selected, key=lambda row: order[row["id"]])


def render_summary(report):
    total = counts(report["rows"])
    by_id = {row["id"]: row for row in report["rows"]}
    lines = [
        f"**Overlap 路线阅读覆盖：{total['learned']}/{total['total']}（{percentage(total)}），待覆盖 {total['U']} 个。**",
        "",
        f"直接注释 D：{total['D']}；链路补全 C：{total['C']}；分支补全 B：{total['B']}。",
        "",
        "| 阅读阶段 | 已覆盖 / 总数 | 待覆盖 | 进度 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for section in report["scope"]["route_sections"]:
        count = counts([by_id[ident] for ident in section["nodes"]])
        lines.append(
            f"| {cell(section['label'])} | {count['learned']}/{count['total']} | {count['U']} | {percentage(count)} |"
        )
    return "\n".join(lines)


def render_route(report):
    scope = report["scope"]
    rows = ordered_rows(report)
    by_id = {row["id"]: row for row in rows}
    numbers = {row["id"]: i for i, row in enumerate(rows, 1)}
    rule = scope["branch_threshold"]
    lines = [
        "# SGLang Overlap 学习路线与现状",
        "",
        scope["scope"],
        "",
        render_summary(report),
        "",
        "## 统计口径与阅读约定",
        "",
        scope["origin"],
        "",
        "- 每个方法等权计 1；范围内每个方法在完整路线中只出现一次。按方法粒度判定，不对方法内部每个条件分支单独计分。类名与方法名分列，模块级函数的类名为 `—`。",
        "- ✅ 已读（D）：源码相对固定基线新增中文学习说明；☑️ 推定（C/B）：按约定补全；⬜ 未读（U）：当前统计未覆盖。未读不等于断言实际没看过。",
        f"- 先 D，再一轮链路 C，最后一轮分支 B：同一概念分支至少 {rule['min_direct']} 个 D，且 (D+C)/分支方法数 ≥ {rule['numerator']}/{rule['denominator']}；C/B 不作新端点。只在本路线白名单中独立计算。",
        "- 启动准备单列模型构造与权重加载，完成后再进入请求路线；请求调度只沿 event_loop_overlap，共享方法内只看相应可达分支。AI 文档、图片、聊天次数不计分，也不要求考试。",
        "- 编号是阅读顺序，不是串行调用栈；Prefill/Decode、两类模型层、Eager/CUDA Graph 和结束/未结束请求都是分支。通信跨越进程，后台接收和 GPU 执行可能并发。",
        "- 通常先提交本批，再处理上一批；关闭本轮重叠时提前处理上一批。只有存在 delay_sample_func 时，才在上一批处理后采样本批。",
        "",
        scope["representatives"],
        "",
        "排除范围：",
        "",
        *[f"- {item}" for item in scope["excluded"]],
        "",
        "## 接下来重点看",
        "",
        "按固定优先级从当前未覆盖方法中选择，再按路线顺序展示；学习后重算会自动跳过已覆盖项。序号对应下方完整路线。",
        "",
        "| 序号 | 状态 | 类名 | 方法名 | 作用 |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in next_rows(report):
        cls, method = method_cells(row)
        lines.append(
            f"| {numbers[row['id']]} | {STATUS[row['status']]} | {cls} | {method} | {cell(row['concept'])} |"
        )
    if not next_rows(report):
        lines += ["", "本路线已全部覆盖。"]

    for section in scope["route_sections"]:
        section_rows = [by_id[ident] for ident in section["nodes"]]
        total = counts(section_rows)
        lines += [
            "",
            f"## {section['label']}",
            "",
            f"**已覆盖 {total['learned']}/{total['total']}（{percentage(total)}），待覆盖 {total['U']}。**",
            "",
            section["note"],
            "",
            "| 序号 | 状态 | 类名 | 方法名 | 作用 |",
            "| ---: | --- | --- | --- | --- |",
        ]
        for row in section_rows:
            cls, method = method_cells(row)
            lines.append(
                f"| {numbers[row['id']]} | {STATUS[row['status']]} | {cls} | {method} | {cell(row['concept'])} |"
            )

    lines += [
        "",
        "## 重算与后续查询",
        "",
        "以后询问“学习路线”“学习现状”“学了多少”“接下来读什么”时，默认使用此模型准备与 Overlap 路线。先从当前源码重算，再报告百分比、阶段表和逐方法清单；保持类名/方法名分列、一个方法一行。全核心总览见 LEARN.methods.md，不能混用两套范围的百分比。",
        "",
        "```bash",
        "python quick_learn/update_overlap_progress.py          # 只预览阶段进度",
        "python quick_learn/update_overlap_progress.py --route  # 预览完整方法路线",
        "python quick_learn/update_overlap_progress.py --write  # 更新本文",
        "python quick_learn/update_overlap_progress.py --check  # 检查是否与源码一致",
        "```",
        "",
        "本文件自动生成；修改阅读范围、顺序或重点请编辑 [路线范围](quick_learn/LEARN.overlap.scope.json)，调整统计范围时递增版本。手动改文档不会产生已读证据。",
        "",
        "<details>",
        "<summary>展开逐方法判定依据与来源</summary>",
        "",
        "| 序号 | 判定 | 依据 |",
        "| ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {numbers[row['id']]} | {row['status']} | {cell(row['evidence'])} |"
        )
    lines += [
        "",
        "</details>",
        "",
        f"- 路线版本：{scope['route_id']} v{scope['version']}；固定分母：{scope['expected_method_count']}。",
        "- 统计对象：当前工作区，包含未提交修改；日期不作为已读证据。",
        f"- HEAD：`{report['head']}`。",
        f"- 固定对照基线：`{scope['baseline_commit']}`。",
        f"- 源码指纹：`{report['source_sha256']}`。",
        f"- 范围指纹：`{report['scope_sha256']}`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="Update only LEARN.overlap.md"
    )
    mode.add_argument(
        "--check", action="store_true", help="Fail when LEARN.overlap.md is stale"
    )
    parser.add_argument(
        "--route", action="store_true", help="Print the complete method route"
    )
    args = parser.parse_args()
    report = measure_overlap()
    rendered = render_route(report)
    print(rendered if args.route else render_summary(report))
    output = ROOT / REPORT_FILE
    if args.check or args.write:
        stale = not output.exists() or output.read_text(encoding="utf-8") != rendered
        if args.check:
            if stale:
                print(f"Stale: {REPORT_FILE}", file=sys.stderr)
                return 1
            print("Overlap progress is current.")
        elif stale:
            output.write_text(rendered, encoding="utf-8")
            print(f"Updated: {REPORT_FILE}")
        else:
            print("Overlap progress is already current.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        sys.exit(f"Overlap progress update failed: {error}")
