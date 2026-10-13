"""Preserve complete source blocks around every learning-comment anchor."""

import ast
import hashlib
import inspect
import io
import re
import tokenize
from pathlib import Path

from extract_annotations import definitions
from quick_learn.update_learn_progress import own_evidence

ROOT = Path(__file__).resolve().parents[2]


def collect_source(snapshot):
    sources = {}
    for path, expected in snapshot["source_hashes"].items():
        source = (ROOT / path).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise ValueError(f"Source changed; refresh annotation snapshot: {path}")
        tree = ast.parse(source)
        comments = {
            token.start[0]: token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }
        strings = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                strings.append({
                    "line": node.lineno, "end": node.end_lineno,
                    "text": inspect.cleandoc(node.value.value), "kind": "string",
                })
        sources[path] = {
            "source": source, "lines": source.splitlines(), "tree": tree,
            "defs": definitions(tree), "comments": comments, "strings": strings,
        }
    return sources


def comment_blocks(info, owned_lines):
    blocks = []
    for line in sorted(set(info["comments"]) & set(owned_lines)):
        raw = info["lines"][line - 1]
        # Inline notes retain their parameter/assignment so their subject is clear.
        text = (re.sub(r"^# ?", "", info["comments"][line])
                if raw.lstrip().startswith("#") else raw.lstrip())
        if blocks and all(not s.strip() for s in info["lines"][blocks[-1]["end"]:line-1]):
            gap = line - blocks[-1]["end"]
            blocks[-1]["text"] += "\n" * gap + text
            blocks[-1]["end"] = line
        else:
            blocks.append({"line": line, "end": line, "text": text, "kind": "comment"})
    return blocks


def own_blocks(info, node=None):
    owned = set(range(1, len(info["lines"]) + 1))
    if node is not None:
        owned = set(own_evidence(node, info["source"], owned))
    blocks = comment_blocks(info, owned)
    blocks += [s.copy() for s in info["strings"] if s["line"] in owned]
    return sorted(blocks, key=lambda b: b["line"])


def doc_block(info, node):
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return None
    return next((s.copy() for s in info["strings"] if s["line"] == node.body[0].lineno), None)


def paragraphs_first(block):
    return {**block, "text": re.split(r"\n\s*\n", block["text"])[0], "evidence_lines": []}


def excerpts_for(entry, sources, entries):
    info = sources[entry["file"]]
    node = info["defs"][entry["name"]][0] if entry["kind"] != "module" else None
    evidence = {a["line"] for a in entry["annotations"]}
    selected = []
    for block in own_blocks(info, node):
        anchors = evidence & set(range(block["line"], block["end"] + 1))
        if anchors:
            selected.append({**block, "evidence_lines": sorted(anchors)})

    # Existing English class/method summary remains useful, without copying an
    # unrelated upstream argument reference or repeating an entire child method.
    doc = doc_block(info, node) if node is not None else None
    if doc and not any(b["kind"] == "string" and b["line"] == doc["line"] for b in selected):
        first = paragraphs_first(doc)
        if first["text"].strip():
            selected.insert(0, first)
    if not selected and entry["kind"] == "class":
        children = [e for e in entries if e["file"] == entry["file"]
                    and e["parent_class"] == entry["name"] and e["kind"] != "class"]
        for child in children:
            child_blocks = excerpts_for(child, sources, entries)
            if child_blocks:
                selected = [{**paragraphs_first(child_blocks[0]), "from_method": child["name"]}]
                break
    if not selected:
        raise ValueError(f"No source explanation for {entry['file']}:{entry['name']}")
    covered = {n for block in selected for n in block["evidence_lines"]}
    if evidence != covered:
        raise ValueError(f"Missing anchors for {entry['name']}: {sorted(evidence-covered)}")
    return selected


def validate_excerpts(snapshot, sources, catalog, module_catalog):
    cache = {}
    expected = set()
    covered = set()
    for entry, record in list(zip(snapshot["entries"], catalog, strict=True)) + list(zip(snapshot["module_notes"], module_catalog, strict=True)):
        path = entry["file"]
        if path not in cache:
            cache[path] = [b["text"] for b in own_blocks(sources[path])]
        expected.update((path, a["line"]) for a in entry["annotations"])
        for block in record["excerpts"]:
            if not any(block["text"] in original for original in cache[path]):
                raise ValueError(f"Non-literal source excerpt: {path}:{block['line']}")
            covered.update((path, n) for n in block["evidence_lines"])
    if expected != covered:
        raise ValueError(f"Missing learning comments: {sorted(expected-covered)}")
    return {"learning_comment_anchors": len(expected), "covered_anchors": len(covered), "missing": []}
