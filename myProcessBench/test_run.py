import sys
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myProcessBench.run import extract_answer, generate, metrics


class ProcessBenchTests(unittest.TestCase):
    def test_import_does_not_load_other_benchmarks_or_serving_libraries(self):
        root = Path(__file__).resolve().parents[1]
        code = '''
import importlib.abc, sys
class IndependentClient(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'ripo', 'pi_engine', 'myCoTControl', 'run_cceval', 'vllm'}:
            raise ImportError('Unavailable in the ProcessBench standalone bundle')
sys.meta_path.insert(0, IndependentClient())
from myProcessBench.run import main
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(root)}, text=True)

    def test_generation_scores_final_channel_and_preserves_reasoning(self):
        raw = {"choices": [{"message": {"reasoning_content": r"Maybe \boxed{0}",
                                        "content": r"Final: \boxed{-1}"}, "finish_reason": "stop"}]}
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda: raw)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        task = {"id": "gsm8k-0", "split": "gsm8k", "label": -1, "steps": 3, "prompt": "Critique"}
        row = generate(client, task, "remote-model", 100)
        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["match"])
        self.assertEqual(row["prediction"], -1)
        self.assertEqual(row["reasoning_source"], "api_field")
        self.assertEqual(calls[0]["model"], "remote-model")
        self.assertEqual(calls[0]["seed"], 42)

    def test_extracts_last_boxed_valid_index(self):
        self.assertEqual(extract_answer(r"Maybe \boxed{1}, final \boxed{-1}", 4), -1)
        self.assertIsNone(extract_answer(r"\boxed{4}", 4))
        self.assertIsNone(extract_answer(r"\boxed{-2}", 4))
        self.assertIsNone(extract_answer("paragraph 1 is incorrect", 4))

    def test_official_harmonic_mean_and_failure_denominator(self):
        rows = [
            {"split": "gsm8k", "label": 1, "match": True, "status": "ok", "prediction": 1},
            {"split": "gsm8k", "label": 2, "match": False, "status": "failed", "prediction": None},
            {"split": "gsm8k", "label": -1, "match": True, "status": "ok", "prediction": -1},
        ]
        result = metrics(rows, {"gsm8k": 3})
        self.assertTrue(result['complete'])
        self.assertEqual(result['groups']['gsm8k']['error_accuracy'], .5)
        self.assertEqual(result['groups']['gsm8k']['correct_accuracy'], 1)
        self.assertAlmostEqual(result['mean_f1'], 2 / 3)
        self.assertEqual(result['groups']['gsm8k']['failed'], 1)

    def test_absent_class_does_not_produce_f1(self):
        rows = [{"split": "math", "label": 0, "match": False, "status": "ok", "prediction": None}]
        self.assertIsNone(metrics(rows, {'math': 1})['mean_f1'])


if __name__ == '__main__':
    unittest.main()
