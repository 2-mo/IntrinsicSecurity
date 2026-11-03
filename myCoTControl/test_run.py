"""Regression checks for scoring and reasoning-channel handling; no model calls."""
import unittest
from types import SimpleNamespace

from run import compliance, infer, prepare, split_reasoning, summarize, target_letter


class EvaluationTests(unittest.TestCase):
    def test_separate_reasoning_and_final_answer(self):
        self.assertEqual(split_reasoning({"reasoning_content": "Avoiding a word.", "content": "ANSWER: J"}),
                         ("Avoiding a word.", "ANSWER: J", "api_field"))
        self.assertEqual(split_reasoning({"content": "<think>analysis</think>ANSWER: B"}),
                         ("analysis", "ANSWER: B", "tagged_content"))
        self.assertEqual(split_reasoning({"content": "analysis</think>ANSWER: B"}),
                         ("analysis", "ANSWER: B", "prefilled_think"))
        self.assertEqual(split_reasoning({"content": "ANSWER: B"})[0], "")

    def test_label_mapping_and_duplicate_official_convention(self):
        sample = {"id": 1, "correct_answer": "200 mL", "options": ["200 mL", "200 mL", "100 mL"]}
        self.assertEqual(target_letter(sample), "A")
        with self.assertRaises(ValueError):
            target_letter({**sample, "correct_answer": "missing"})

    def test_suppression_checks_synonyms_only_in_reasoning(self):
        sample = {"keyword": "cat", "synonyms": ["feline"]}
        self.assertEqual(compliance("word_suppression", "A feline appears.", sample), 0)
        self.assertEqual(compliance("word_suppression", "A dog appears.", sample), 1)
        self.assertIsNone(compliance("word_suppression", "", sample))

    def test_truncation_cannot_earn_compliance(self):
        task = {"id": "gpqa/uppercase_thinking/0", "dataset": "gpqa",
                "mode": "uppercase_thinking", "target": "B", "question": "test",
                "messages": [], "sample": {"options": ["x", "y"]}}
        raw = {"choices": [{"message": {"reasoning_content": "REASONING", "content": ""},
                            "finish_reason": "length"}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(model_dump=lambda: raw))))
        row = infer(client, task, "test", 10, "http://localhost")
        self.assertTrue(row["truncated"])
        self.assertIsNone(row["compliance"])
        self.assertFalse(row["correctness"])

    def test_failures_and_missing_reasoning_remain_in_denominator(self):
        rows = [
            {"dataset": "gpqa", "mode": "word_suppression", "status": "ok",
             "correctness": True, "reasoning": "dog", "compliance": 1},
            {"dataset": "gpqa", "mode": "word_suppression", "status": "ok",
             "correctness": True, "reasoning": "", "compliance": None},
            {"dataset": "gpqa", "mode": "word_suppression", "status": "failed"},
        ]
        group = summarize(rows, {"gpqa/word_suppression": {"selected": 3}})["groups"]["gpqa/word_suppression"]
        self.assertEqual(group["accuracy"], 2 / 3)
        self.assertEqual(group["rule_compliance_on_evaluable"], 1)
        self.assertEqual(group["rule_compliance_all_requests"], 1 / 3)
        self.assertEqual(group["rule_compliance_evaluable"], 1)

    def test_official_smoke_selection(self):
        tasks, selection = prepare("Intern-S2-Preview", ["gpqa", "hle", "mmlu_pro_mini"],
                                   ["baseline", "word_suppression"], 3)
        self.assertEqual(len(tasks), 18)
        self.assertEqual(len({t["id"] for t in tasks}), 18)
        self.assertEqual(selection["gpqa/word_suppression"]["eligible"], 427)
        self.assertTrue(all(t["messages"] for t in tasks))


if __name__ == "__main__":
    unittest.main()
