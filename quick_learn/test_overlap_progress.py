"""Check that route filtering cannot silently change the progress denominator."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quick_learn.update_learn_progress import ROOT, infer_coverage, measure
from quick_learn.update_overlap_progress import (
    SCOPE_FILE,
    method_cells,
    next_rows,
    ordered_rows,
    validate_route,
)


class OverlapRouteTests(unittest.TestCase):
    def setUp(self):
        self.scope = json.loads((ROOT / SCOPE_FILE).read_text(encoding="utf-8"))

    def test_fixed_scope_contains_overlap_methods_and_no_normal_loop(self):
        validate_route(self.scope)
        names = {
            node[2] for branch in self.scope["branches"] for node in branch["nodes"]
        }
        self.assertEqual(len(names), 127)
        self.assertIn("Scheduler.event_loop_overlap", names)
        self.assertIn("Scheduler.launch_batch_sample_if_needed", names)
        self.assertIn("GenerationBatchResult.copy_to_cpu", names)
        self.assertIn("Qwen3_5ForConditionalGeneration.__init__", names)
        self.assertIn("Qwen3_5ForConditionalGeneration.load_weights", names)
        self.assertIn("Qwen3VLForConditionalGeneration.forward", names)
        self.assertIn("general_mm_embed_routine", names)
        self.assertIn("LogitsProcessor._compute_lm_head", names)
        self.assertNotIn("Qwen3_5ForConditionalGeneration.forward", names)
        self.assertNotIn("Scheduler.event_loop_normal", names)
        self.assertNotIn("DecodeCudaGraphRunner.capture", names)

    def test_missing_or_repeated_route_row_fails_instead_of_changing_total(self):
        for missing in (True, False):
            scope = copy.deepcopy(self.scope)
            nodes = scope["route_sections"][0]["nodes"]
            if missing:
                nodes.pop()
            else:
                nodes.append(nodes[0])
            with self.assertRaises(ValueError):
                validate_route(scope)

    def test_normal_loop_inference_is_rejected_even_with_valid_nodes(self):
        scope = copy.deepcopy(self.scope)
        scope["chains"].append(
            {
                "id": "normal_schedule",
                "kind": "call",
                "nodes": ["overlap_loop", "next_batch"],
            }
        )
        with self.assertRaises(ValueError):
            validate_route(scope)

    def test_no_future_map_init_evidence_is_borrowed_to_complete_request_branch(self):
        status, _ = infer_coverage(self.scope, {"future_stash": [560]})
        self.assertEqual(status["future_publish"], "U")
        with self.assertRaises(ValueError):
            infer_coverage(self.scope, {"future_init": [259]})

    def test_constructor_evidence_does_not_complete_weights_or_inference(self):
        status, _ = infer_coverage(
            self.scope,
            {
                "load_model": [1],
                "qwen_parent_init": [2],
                "qwen_model_init": [3],
            },
        )
        self.assertEqual(status["qwen_entry_init"], "C")
        for ident in (
            "load_weights_postprocess",
            "qwen_load_weights",
            "qwen_gdn_init",
            "causal_lm",
            "qwen_model",
            "process_logits",
        ):
            self.assertEqual(status[ident], "U", ident)

    def test_text_embedding_bridge_uses_forward_evidence(self):
        status, _ = infer_coverage(self.scope, {"causal_lm": [1], "qwen_model": [2]})
        self.assertEqual(status["qwen_embed"], "C")
        self.assertEqual(status["qwen_parent_init"], "U")
        self.assertEqual(status["compute_lm_head"], "U")

    def test_next_reading_omits_direct_and_inferred_methods(self):
        ids = [n for s in self.scope["route_sections"] for n in s["nodes"]]
        report = {
            "scope": self.scope,
            "rows": [{"id": ident, "status": "U"} for ident in ids],
        }
        by_id = {row["id"]: row for row in report["rows"]}
        for ident, status in zip(self.scope["priority_nodes"][:3], ("D", "C", "B")):
            by_id[ident]["status"] = status
        self.assertEqual(next_rows(report, 1)[0]["id"], self.scope["priority_nodes"][3])
        self.assertEqual([r["id"] for r in ordered_rows(report)], ids)

    def test_method_link_preserves_dunder_name_and_separates_class(self):
        cls, link = method_cells(
            {
                "qualname": "Req.__init__",
                "file": "example.py",
                "line": 12,
            }
        )
        self.assertEqual(cls, "Req")
        self.assertEqual(link, r"[\_\_init\_\_](example.py#L12)")

    def test_measure_reads_requested_scope_and_its_sources(self):
        scope = {
            "baseline_commit": "0" * 40,
            "node_columns": ["id", "source", "qualname", "concept"],
            "sources": {"src": "route.py"},
            "stages": {"1": "route"},
            "branch_threshold": {"numerator": 2, "denominator": 3, "min_direct": 2},
            "branches": [
                {
                    "id": "route",
                    "stage": "1",
                    "nodes": [
                        ["f", "src", "f", "request"],
                    ],
                }
            ],
            "chains": [],
        }
        before = "def f():\n    return 1\n"
        after = "def f():\n    # 学习说明\n    return 1\n"

        def fake_git(root, *args):
            if args[0] == "ls-tree":
                return "route.py\n"
            if args[0] == "rev-parse":
                return "1" * 40
            if args[0] == "show":
                return before
            return ""

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "route.json").write_text(json.dumps(scope), encoding="utf-8")
            (root / "route.py").write_text(after, encoding="utf-8")
            with patch("quick_learn.update_learn_progress.git", side_effect=fake_git):
                report = measure(root, scope_file="route.json")
            self.assertEqual(report["rows"][0]["status"], "D")
            self.assertEqual(report["rows"][0]["evidence"], "新增中文说明 L2")


if __name__ == "__main__":
    unittest.main()
