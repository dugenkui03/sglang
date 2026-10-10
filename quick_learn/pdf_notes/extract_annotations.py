#!/usr/bin/env python3
"""Snapshot added Chinese learning comments without importing SGLang."""

import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quick_learn.update_learn_progress import (  # noqa: E402
    added_lines,
    annotation_fragments,
    git,
    new_annotation_lines,
    own_evidence,
)


def definitions(tree):
    result = {}

    def visit(node, prefix="", parent_class=None):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            result[name] = (node, parent_class)
            if isinstance(node, ast.ClassDef):
                parent_class = name
            prefix = name + "."
        for child in ast.iter_child_nodes(node):
            visit(child, prefix, parent_class)

    visit(tree)
    return result


def extract():
    scope = json.loads((ROOT / "quick_learn/LEARN.scope.json").read_text())
    baseline = scope["baseline_commit"]
    baseline_paths = set(git(ROOT, "ls-tree", "-r", "--name-only", baseline).splitlines())
    paths = subprocess.run(
        ["rg", "-l", r"[\p{Han}]", "python", "-g", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    hints = {}
    for branch in scope["branches"]:
        for _, alias, name, concept in branch["nodes"]:
            hints[scope["sources"][alias], name] = concept
    entries, module_notes, skipped = [], [], []
    snapshots = {}
    for path in sorted(paths):
        source = (ROOT / path).read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            skipped.append({"file": path, "reason": str(exc)})
            continue
        before = git(ROOT, "show", f"{baseline}:{path}") if path in baseline_paths else ""
        previous_tree = ast.parse(before)
        current = annotation_fragments(source, tree)
        previous = annotation_fragments(before, previous_tree)
        changed = added_lines(before, source)
        current_defs = definitions(tree)
        previous_defs = definitions(previous_tree)
        found = {}
        claimed = set()
        for name, (node, parent_class) in current_defs.items():
            current_lines = own_evidence(node, source, set(current))
            previous_lines = (
                own_evidence(previous_defs[name][0], before, set(previous))
                if name in previous_defs else []
            )
            evidence = new_annotation_lines(
                {n: current[n] for n in current_lines},
                {n: previous[n] for n in previous_lines}, changed,
            )
            if not evidence:
                continue
            claimed.update(evidence)
            found[name] = {
                "file": path, "name": name, "line": node.lineno,
                "kind": "class" if isinstance(node, ast.ClassDef) else "method" if parent_class else "function",
                "parent_class": parent_class, "direct": True,
                "hint": hints.get((path, name), ""),
                "bases": [ast.unparse(b) for b in node.bases] if isinstance(node, ast.ClassDef) else [],
                "annotations": [{"line": n, "text": current[n]} for n in evidence],
            }
        for entry in list(found.values()):
            parent = entry["parent_class"]
            while parent and parent not in found:
                node, grandparent = current_defs[parent]
                found[parent] = {
                    "file": path, "name": parent, "line": node.lineno, "kind": "class",
                    "parent_class": grandparent, "direct": False, "hint": "",
                    "bases": [ast.unparse(b) for b in node.bases], "annotations": [],
                }
                parent = grandparent
        all_added = set(new_annotation_lines(current, previous, changed))
        unclaimed = all_added - claimed
        # Exclude comments owned by any definition: these are handled above.
        for node, _ in current_defs.values():
            unclaimed -= set(own_evidence(node, source, set(current)))
        if unclaimed:
            module_notes.append({"file": path, "annotations": [
                {"line": n, "text": current[n]} for n in sorted(unclaimed)
            ]})
        if found or unclaimed:
            snapshots[path] = hashlib.sha256(source.encode()).hexdigest()
        entries.extend(found.values())
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "head": git(ROOT, "rev-parse", "HEAD").strip(), "baseline": baseline,
        "rule": "相对固定基线新增的中文注释与独立说明字符串；方法归属类另列，不作推定已读；包含未提交修改。",
        "source_hashes": snapshots, "skipped": skipped,
        "entries": sorted(entries, key=lambda e: (e["file"], e["line"])),
        "module_notes": module_notes,
    }


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("annotations.json")
    snapshot = extract()
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    from collections import Counter
    print(json.dumps({"output": str(destination), "counts": dict(Counter(e["kind"] for e in snapshot["entries"])),
                      "source_files": len(snapshot["source_hashes"]), "skipped": snapshot["skipped"]}, ensure_ascii=False))
