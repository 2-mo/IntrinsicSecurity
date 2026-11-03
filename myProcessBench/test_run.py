import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myProcessBench.run import extract_answer, metrics


class ProcessBenchTests(unittest.TestCase):
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
