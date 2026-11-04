from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from baseline_json import run
from test_baseline_json import sample


class BaselineJSONRunTests(unittest.TestCase):
    def evaluate_output(self, text):
        client = Mock()
        client.complete_with_metadata.return_value = SimpleNamespace(
            responses=[{"role": "assistant", "content": text}],
            latency_sec=1.0, usage={"completion_tokens": 20},
            endpoint="http://localhost/v1", model="test-model",
        )
        with patch.object(run, "baseline_messages", return_value=[]):
            return run.run_one(client, sample(), data_path=Path('.'),
                               template_path=Path('prompt.txt'), media_root=Path('.'),
                               max_tokens=100)

    def test_valid_json_preserves_reasoning_for_evaluation(self):
        result = self.evaluate_output(json.dumps({"reasoning": "P implies Q", "final_answer": "B"}))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["correct"])
        self.assertEqual(result["reasoning"], "P implies Q")

    def test_malformed_or_out_of_contract_outputs_are_not_scored(self):
        for text in ('Answer: B', '{"reasoning":"x","final_answer":"E"}',
                     '{"reasoning":7,"final_answer":"B"}',
                     '{"reasoning":"x","final_answer":"B","extra":true}'):
            with self.subTest(text=text):
                result = self.evaluate_output(text)
                self.assertEqual(result["status"], "invalid_json")
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["output"], text)

    def test_current_ripo_client_receives_explicit_configuration(self):
        config = run.ModelConfig(endpoints=("http://localhost/v1",), model="test", pool_concurrency=4, seed=42)
        with patch.object(run.ripo_client, "PIVLM") as backend:
            self.assertIs(run.resolve_model_config(config), config)
            run.build_ripo_client_from_config(config, max_tokens=8192, temperature=0.0)
        actual_config = backend.call_args.args[0]
        self.assertEqual(actual_config.endpoints, config.endpoints)
        self.assertEqual(actual_config.request_seed, 42)
        self.assertEqual(actual_config.max_tokens, 8192)

    def test_current_ripo_rejects_unresolved_auto_endpoint(self):
        with patch.object(run, "ripo_client", SimpleNamespace()):
            with self.assertRaisesRegex(ValueError, "explicit --endpoints"):
                run.resolve_model_config(run.ModelConfig(endpoints=("auto",)))


if __name__ == "__main__":
    unittest.main()
