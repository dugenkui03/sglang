#!/usr/bin/env python3
"""Recompute the fixed LEARN core-reading scope without importing SGLang.

Run without arguments to preview, with --write to update the two reports, or
with --check to check that the reports match the current working tree.
"""

import argparse
import ast
import difflib
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
BEGIN = "<!-- LEARN-PROGRESS:START -->"
END = "<!-- LEARN-PROGRESS:END -->"


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def definitions(tree):
    """Qualified module/class definitions; do not flatten nested functions."""
    result = {}

    def visit(node, prefix=""):
        if isinstance(node, FUNCTION):
            result.setdefault(prefix + node.name, []).append(node)
            return
        if isinstance(node, ast.ClassDef):
            prefix += node.name + "."
        for child in ast.iter_child_nodes(node):
            visit(child, prefix)

    visit(tree)
    return result


def annotation_fragments(source, tree):
    """Chinese comments and standalone explanation strings, never UI strings."""
    lines = source.splitlines()
    evidence = {
        token.start[0]: token.string.strip()
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and CJK.search(token.string)
    }
    for node in ast.walk(tree):
        # Includes multiline docstrings and comment-like unassigned string blocks.
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            evidence.update(
                {
                    line: lines[line - 1].strip()
                    for line in range(node.lineno, node.end_lineno + 1)
                    if CJK.search(lines[line - 1])
                }
            )
    return evidence


def annotation_lines(source, tree):
    return set(annotation_fragments(source, tree))


def new_annotation_lines(current, previous, changed):
    # A moved comment, or a code edit beside an unchanged inline comment, does
    # not create new reading evidence. Preserve counts for duplicate comments.
    additions = Counter(current.values()) - Counter(previous.values())
    return sorted(
        line
        for line, text in current.items()
        if line in changed and additions[text] > 0
    )


def added_lines(before, after):
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    )
    return {
        line
        for tag, _a, _b, start, end in matcher.get_opcodes()
        if tag in ("insert", "replace")
        for line in range(start + 1, end + 1)
    }


def extent(node, lines):
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    while start > 1:
        prior = lines[start - 2]
        indent = len(prior) - len(prior.lstrip())
        if not prior.lstrip().startswith("#") or indent != node.col_offset:
            break
        start -= 1
    end = node.end_lineno
    # AST end positions exclude trailing comment-only lines inside the method.
    for pos in range(end, len(lines)):
        line = lines[pos]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if line.lstrip().startswith("#") and indent > node.col_offset:
            end = pos + 1
        else:
            break
    return start, end


def own_evidence(node, source, candidates):
    lines = source.splitlines()
    start, end = extent(node, lines)
    own = set(range(start, end + 1))
    for child in ast.walk(node):
        if child is not node and isinstance(child, (*FUNCTION, ast.ClassDef)):
            nested_start, nested_end = extent(child, lines)
            own.difference_update(range(nested_start, nested_end + 1))
    return sorted(own & candidates)


def validate_scope(scope):
    if scope["node_columns"] != ["id", "source", "qualname", "concept"]:
        raise ValueError("Unexpected node schema")
    threshold = scope["branch_threshold"]
    if not (
        0 < threshold["numerator"] <= threshold["denominator"]
        and threshold["min_direct"] >= 2
    ):
        raise ValueError("Invalid branch threshold")
    nodes, branches, targets = set(), set(), set()
    for branch in scope["branches"]:
        if branch["id"] in branches or not branch["nodes"]:
            raise ValueError(f"Duplicate/empty branch: {branch['id']}")
        branches.add(branch["id"])
        if branch["stage"] not in scope["stages"]:
            raise ValueError(f"Unknown stage: {branch['stage']}")
        for ident, alias, qualname, concept in branch["nodes"]:
            if alias not in scope["sources"] or not concept:
                raise ValueError(f"Invalid source/concept: {ident}")
            target = (scope["sources"][alias], qualname)
            if ident in nodes or target in targets:
                raise ValueError(f"Duplicate core method: {ident}")
            nodes.add(ident)
            targets.add(target)
    if not nodes:
        raise ValueError("Empty core scope")
    chains = set()
    for chain in scope["chains"]:
        path = chain["nodes"]
        if (
            chain["id"] in chains
            or len(path) < 2
            or len(path) != len(set(path))
            or not set(path) <= nodes
            or chain["kind"] not in ("call", "call/process", "ipc", "data")
        ):
            raise ValueError(f"Invalid core chain: {chain['id']}")
        chains.add(chain["id"])


def infer_coverage(scope, direct):
    """D -> a single chain pass -> a single independent branch pass."""
    all_ids = {node[0] for b in scope["branches"] for node in b["nodes"]}
    if not set(direct) <= all_ids:
        raise ValueError("Evidence outside the core scope")
    status = {ident: "D" if ident in direct else "U" for ident in all_ids}
    reasons = {
        ident: [f"新增中文说明 L{','.join(map(str, lines))}"]
        for ident, lines in direct.items()
    }
    for chain in scope["chains"]:
        path = chain["nodes"]
        anchors = [i for i, ident in enumerate(path) if ident in direct]
        for left, right in pairwise(anchors):
            for ident in path[left + 1 : right]:
                status[ident] = "C"
                reasons.setdefault(ident, []).append(
                    f"chain={chain['id']};anchors={path[left]}->{path[right]}"
                )

    known = {ident for ident, state in status.items() if state in ("D", "C")}
    rule = scope["branch_threshold"]
    for branch in scope["branches"]:
        ids = {node[0] for node in branch["nodes"]}
        num_direct = len(ids & direct.keys())
        num_known = len(ids & known)
        if (
            num_direct >= rule["min_direct"]
            and num_known * rule["denominator"] >= len(ids) * rule["numerator"]
        ):
            for ident in ids - known:
                status[ident] = "B"
                reasons[ident] = [
                    (
                        f"branch={branch['id']};direct={num_direct};"
                        f"covered_before_branch={num_known}/{len(ids)}"
                    )
                ]
    return status, reasons


def measure(root=ROOT):
    scope_text = (root / "quick_learn" / "LEARN.scope.json").read_text(encoding="utf-8")
    scope = json.loads(scope_text)
    validate_scope(scope)
    baseline = scope["baseline_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", baseline):
        raise ValueError("Pin baseline_commit to a full Git commit SHA")
    git(root, "cat-file", "-e", baseline + "^{commit}")
    baseline_paths = set(
        git(root, "ls-tree", "-r", "--name-only", baseline).splitlines()
    )
    head = git(root, "rev-parse", "HEAD").strip()
    paths = sorted(set(scope["sources"].values()))
    # Read each worktree source exactly once; do not mix scans of changing files.
    sources = {path: (root / path).read_text(encoding="utf-8") for path in paths}
    indexes, candidates, annotations, baselines = {}, {}, {}, {}
    fingerprint = hashlib.sha256()
    for path, source in sources.items():
        fingerprint.update(path.encode() + b"\0" + source.encode() + b"\0")
        tree = ast.parse(source, filename=path)
        indexes[path] = definitions(tree)
        before = (
            git(root, "show", f"{baseline}:{path}") if path in baseline_paths else ""
        )
        before_tree = ast.parse(before, filename=path)
        annotations[path] = annotation_fragments(source, tree)
        candidates[path] = set(annotations[path]) & added_lines(before, source)
        baselines[path] = (
            before,
            definitions(before_tree),
            annotation_fragments(before, before_tree),
        )

    direct, rows = {}, []
    for branch in scope["branches"]:
        for ident, alias, qualname, concept in branch["nodes"]:
            path = scope["sources"][alias]
            matches = indexes[path].get(qualname, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one definition for {path}:{qualname}, got {len(matches)}; "
                    "update the scope explicitly instead of silently changing the denominator"
                )
            node = matches[0]
            current_lines = own_evidence(node, sources[path], set(annotations[path]))
            before, old_definitions, old_annotations = baselines[path]
            previous = old_definitions.get(qualname, [])
            if len(previous) > 1:
                raise ValueError(f"Ambiguous baseline definition: {path}:{qualname}")
            previous_lines = (
                own_evidence(previous[0], before, set(old_annotations))
                if previous
                else []
            )
            evidence = new_annotation_lines(
                {line: annotations[path][line] for line in current_lines},
                {line: old_annotations[line] for line in previous_lines},
                candidates[path],
            )
            if evidence:
                direct[ident] = evidence
            rows.append(
                {
                    "id": ident,
                    "stage": branch["stage"],
                    "branch": branch["id"],
                    "file": path,
                    "qualname": qualname,
                    "line": node.lineno,
                    "concept": concept,
                }
            )
    status, reasons = infer_coverage(scope, direct)
    for row in rows:
        row["status"] = status[row["id"]]
        row["learned"] = int(row["status"] != "U")
        row["evidence"] = " | ".join(
            reasons.get(row["id"], ["无直接证据，且未满足补全规则"])
        )
    return {
        "scope": scope,
        "rows": rows,
        "head": head,
        "source_sha256": fingerprint.hexdigest(),
        "scope_sha256": hashlib.sha256(scope_text.encode()).hexdigest(),
    }


def counts(rows):
    result = Counter(row["status"] for row in rows)
    result["total"] = len(rows)
    result["learned"] = sum(result[key] for key in ("D", "C", "B"))
    return result


def percentage(count):
    return f"{100 * count['learned'] / count['total']:.1f}%" if count["total"] else "—"


def render_methods(report):
    scope, rows = report["scope"], report["rows"]
    total = counts(rows)
    rule = scope["branch_threshold"]
    lines = [
        "# SGLang 核心链路阅读覆盖清单；由 quick_learn/update_learn_progress.py --write 生成。",
        f"# scope_version={scope['version']}; baseline_commit={scope['baseline_commit']}",
        f"# source=current_worktree_including_uncommitted; HEAD={report['head']}",
        f"# source_sha256={report['source_sha256']}",
        f"# scope_sha256={report['scope_sha256']}",
        f"# {scope['scope']}",
        f"# {scope['representatives']}",
        "# D=直接注释；C=两个 D 之间的链路补全；B=分支补全；U=待覆盖。",
        (
            f"# 分支规则：至少 {rule['min_direct']} 个 D，且补全前 (D+C)/N >= "
            f"{rule['numerator']}/{rule['denominator']}；C/B 不作为新的链路端点。"
        ),
        "# 每个方法只属于一个分支、等权计 1；learned=1 包含 D/C/B，不表示考试或能力认证。",
        (
            f"# 已覆盖={total['learned']}/{total['total']} ({percentage(total)}); "
            f"D={total['D']}; C={total['C']}; B={total['B']}; U={total['U']}"
        ),
    ]
    for branch in scope["branches"]:
        count = counts([row for row in rows if row["branch"] == branch["id"]])
        lines.append(
            f"# branch={branch['id']} ({branch['label']}): "
            f"{count['learned']}/{count['total']}; "
            f"D={count['D']},C={count['C']},B={count['B']},U={count['U']}"
        )
    columns = [
        "learned",
        "status",
        "stage",
        "branch",
        "id",
        "file",
        "qualname",
        "line",
        "concept",
        "evidence",
    ]
    lines.append("\t".join(columns))
    for row in rows:
        lines.append(
            "\t".join(
                str(row[key]).replace("\t", " ").replace("\n", " ") for key in columns
            )
        )
    return "\n".join(lines) + "\n"


def render_summary(report):
    scope, rows = report["scope"], report["rows"]
    total = counts(rows)
    lines = [
        f"**当前核心链路阅读覆盖：{total['learned']}/{total['total']}（{percentage(total)}）。**",
        "",
        (
            f"其中：直接注释 **{total['D']}** 个，链路补全 **{total['C']}** 个，"
            f"分支补全 **{total['B']}** 个；待覆盖 **{total['U']}** 个。"
            "该数值表示按约定视为已读的核心方法比例。"
        ),
        "",
        "| 核心阶段 | 直接注释 D | 链路补全 C | 分支补全 B | 已覆盖 / 总数 | 覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, label in scope["stages"].items():
        count = counts([row for row in rows if row["stage"] == stage])
        lines.append(
            f"| {stage} {label} | {count['D']} | {count['C']} | {count['B']} | "
            f"{count['learned']}/{count['total']} | {percentage(count)} |"
        )
    lines += [
        "",
        (
            f"统计范围版本：v{scope['version']}，{len(scope['branches'])} 个分支，"
            f"{len(scope['sources'])} 个源码文件中的白名单方法。"
            f"读取当前工作区（含未提交修改），HEAD `{report['head'][:12]}`，"
            f"固定对照基线 `{scope['baseline_commit'][:12]}`。"
        ),
        (
            f"源码指纹 `{report['source_sha256'][:16]}`；范围指纹 `{report['scope_sha256'][:16]}`。"
            "完整指纹和逐方法依据见 [LEARN.methods.txt](LEARN.methods.txt)。"
        ),
    ]
    return "\n".join(lines)


def replace_summary(document, summary):
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("LEARN.md must contain exactly one progress marker pair")
    start = document.index(BEGIN) + len(BEGIN)
    end = document.index(END)
    if end < start:
        raise ValueError("Reversed progress markers")
    return document[:start] + "\n\n" + summary + "\n\n" + document[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="Update LEARN.md and LEARN.methods.txt"
    )
    mode.add_argument(
        "--check", action="store_true", help="Fail if either generated report is stale"
    )
    args = parser.parse_args()
    report = measure()
    summary = render_summary(report)
    print(summary)
    if not (args.write or args.check):
        return 0
    learn = ROOT / "LEARN.md"
    outputs = {
        ROOT / "LEARN.methods.txt": render_methods(report),
        learn: replace_summary(learn.read_text(encoding="utf-8"), summary),
    }
    stale = [
        path
        for path, text in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        if stale:
            print("Stale: " + ", ".join(path.name for path in stale), file=sys.stderr)
            return 1
        print("Generated progress is current.")
    else:
        for path in stale:
            path.write_text(outputs[path], encoding="utf-8")
        print(
            "Updated: " + (", ".join(path.name for path in stale) or "already current")
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        sys.exit(f"Progress update failed: {error}")
