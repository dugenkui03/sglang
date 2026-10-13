"""Focused regression tests for reading-evidence and inference boundaries."""

import ast
import copy
import unittest

from quick_learn.update_learn_progress import (
    BEGIN,
    END,
    added_lines,
    annotation_lines,
    definitions,
    infer_coverage,
    new_annotation_lines,
    own_evidence,
    replace_summary,
    validate_scope,
)


def make_scope(branches, chains=()):
    return {
        "node_columns": ["id", "source", "qualname", "concept"],
        "sources": {"src": "example.py"},
        "stages": {"1": "core"},
        "branch_threshold": {"numerator": 2, "denominator": 3, "min_direct": 2},
        "branches": [
            {
                "id": f"b{i}",
                "stage": "1",
                "nodes": [[name, "src", name, "core concept"] for name in names],
            }
            for i, names in enumerate(branches)
        ],
        "chains": [
            {"id": f"c{i}", "nodes": list(names), "kind": "call"}
            for i, names in enumerate(chains)
        ],
    }


class InferenceTests(unittest.TestCase):
    def test_chain_fills_gap_but_not_sibling_or_tail(self):
        scope = make_scope(
            [["a"], ["b"], ["c"], ["tail"], ["sibling"]],
            [["a", "b", "c", "tail"]],
        )
        status, reasons = infer_coverage(scope, {"a": [1], "c": [3]})
        self.assertEqual(
            status, {"a": "D", "b": "C", "c": "D", "tail": "U", "sibling": "U"}
        )
        self.assertIn("anchors=a->c", reasons["b"][0])

    def test_single_endpoint_does_not_fill_chain(self):
        status, _ = infer_coverage(make_scope(["abc"], ["abc"]), {"a": [1]})
        self.assertEqual(status, {"a": "D", "b": "U", "c": "U"})

    def test_chain_inference_is_not_a_new_anchor(self):
        scope = make_scope(["a", "b", "c", "x", "d"], ["abc", "bxd"])
        status, _ = infer_coverage(scope, {"a": [1], "c": [3], "d": [5]})
        self.assertEqual(status["b"], "C")
        self.assertEqual(status["x"], "U")

    def test_branch_at_exact_threshold_is_complete(self):
        status, reasons = infer_coverage(make_scope(["abc"]), {"a": [1], "b": [2]})
        self.assertEqual(status["c"], "B")
        self.assertIn("covered_before_branch=2/3", reasons["c"][0])

    def test_branch_below_threshold_stays_open(self):
        status, _ = infer_coverage(make_scope(["abcd"]), {"a": [1], "b": [2]})
        self.assertEqual(status["c"], "U")
        self.assertEqual(status["d"], "U")

    def test_branch_uses_chain_coverage_but_requires_two_direct_nodes(self):
        scope = make_scope(["abcd", "x", "y"], ["xab"])
        status, _ = infer_coverage(scope, {"x": [1], "b": [2], "c": [3]})
        self.assertEqual(status["a"], "C")
        self.assertEqual(status["d"], "B")
        scope = make_scope(["abcd", "x", "y"], ["xaby"])
        status, _ = infer_coverage(scope, {"x": [1], "y": [2], "c": [3]})
        self.assertEqual(status["a"], "C")
        self.assertEqual(status["d"], "U")

    def test_branch_inference_does_not_spread_to_another_branch(self):
        scope = make_scope(["abc", "xyz"], ["cxy"])
        status, _ = infer_coverage(scope, {"a": [1], "b": [2], "y": [3]})
        self.assertEqual(status["c"], "B")
        self.assertEqual(status["x"], "U")
        self.assertEqual(status["z"], "U")
        reverse = copy.deepcopy(scope)
        reverse["branches"].reverse()
        self.assertEqual(
            infer_coverage(reverse, {"a": [1], "b": [2], "y": [3]})[0], status
        )


class EvidenceTests(unittest.TestCase):
    def evidence(self, source, name, before=""):
        tree = ast.parse(source)
        candidates = annotation_lines(source, tree) & added_lines(before, source)
        return own_evidence(definitions(tree)[name][0], source, candidates)

    def test_multiline_docstring_and_unassigned_explanation(self):
        source = 'def f():\n    """Forward pass.\n    执行采样\n    """\n    """补充说明"""\n    return 1\n'
        self.assertEqual(self.evidence(source, "f"), [3, 5])

    def test_chinese_runtime_strings_are_not_reading_evidence(self):
        source = 'def f():\n    value = "普通数据"\n    print("用户输出")\n    raise ValueError("错误信息")\n'
        self.assertEqual(self.evidence(source, "f"), [])

    def test_unchanged_annotations_do_not_count(self):
        before = "def f():\n    # 已有中文注释\n    return 1\n"
        source = before.replace("return 1", "return 2")
        self.assertEqual(self.evidence(source, "f", before), [])

    def test_code_edit_or_comment_move_is_not_new_annotation(self):
        self.assertEqual(
            new_annotation_lines({4: "# 执行采样"}, {2: "# 执行采样"}, {4}), []
        )
        self.assertEqual(
            new_annotation_lines({4: "# 执行采样"}, {4: "# 执行采样"}, {4}), []
        )
        self.assertEqual(
            new_annotation_lines({4: "# 执行采样"}, {4: "# 旧说明"}, {4}), [4]
        )

    def test_duplicate_annotation_added_to_method_is_new_evidence(self):
        self.assertEqual(
            new_annotation_lines(
                {2: "# 执行采样", 4: "# 执行采样"}, {2: "# 执行采样"}, {4}
            ),
            [4],
        )

    def test_nested_annotations_are_not_attributed_to_parent(self):
        source = 'def outer():\n    # 内部函数的说明\n    def inner():\n        """内部函数说明"""\n        pass\n    class Nested:\n        """内部类说明"""\n        pass\n    return inner\n'
        self.assertEqual(self.evidence(source, "outer"), [])

    def test_leading_and_trailing_comments_belong_to_correct_method(self):
        source = "def previous():\n    return 1\n    # 上个方法的尾注\n\n# 当前方法说明\n@decorator\ndef current():\n    return 2\n"
        self.assertEqual(self.evidence(source, "previous"), [3])
        self.assertEqual(self.evidence(source, "current"), [5])

    def test_class_or_module_docstring_does_not_mark_methods(self):
        source = '"""模块说明"""\nclass C:\n    """类说明"""\n    def method(self):\n        pass\n'
        self.assertEqual(self.evidence(source, "C.method"), [])


class ScopeAndOutputTests(unittest.TestCase):
    def test_scope_rejects_duplicate_method_and_invalid_chain(self):
        validate_scope(make_scope(["abc"], ["abc"]))
        with self.assertRaises(ValueError):
            validate_scope(make_scope(["ab", "a"]))
        with self.assertRaises(ValueError):
            validate_scope(make_scope(["ab"], ["abc"]))

    def test_only_generated_summary_is_replaced(self):
        document = f"User notes\n{BEGIN}\nOld stats\n{END}\nMore user notes\n"
        result = replace_summary(document, "New stats")
        self.assertEqual(
            result, f"User notes\n{BEGIN}\n\nNew stats\n\n{END}\nMore user notes\n"
        )
        with self.assertRaises(ValueError):
            replace_summary("User notes without markers", "New stats")


if __name__ == "__main__":
    unittest.main()
