from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import RateLimitError

from baseline_json import run as baseline
from logicam_json import run as logicam
from muslr_core.inference import JSONClient, ModelConfig, image_block, prepare_messages
from test_baseline_json import sample


def completion(output: str | None) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "test-completion", "object": "chat.completion", "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": output, "reasoning_content": "private reasoning",
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


class InferenceTests(unittest.TestCase):
    def test_imports_work_when_ripo_and_pi_engine_are_unavailable(self):
        source = Path(__file__).resolve().parents[1]
        code = '''
import importlib.abc, sys
class NoRipo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'ripo', 'pi_engine'}:
            raise ImportError('RIPO is unavailable')
sys.meta_path.insert(0, NoRipo())
import baseline_json.run, logicam_json.run, legacy.free_text_run
import evaluation.evaluate, result_dashboard.app
'''
        subprocess.run([sys.executable, "-c", code], check=True,
                       env={**os.environ, "PYTHONPATH": str(source), "RIPO_ROOT": "/missing-ripo"},
                       capture_output=True, text=True)

    def test_data_images_are_portable_and_input_messages_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image with spaces.png"
            image.write_bytes(b"example-image")
            original = [{"role": "user", "content": [image_block(image)]}]
            prepared = prepare_messages(original, "data")
            url = prepared[0]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            self.assertEqual(base64.b64decode(url.partition(",")[2]), image.read_bytes())
            self.assertEqual(prepare_messages(original, "file"), original)
            self.assertTrue(original[0]["content"][0]["image_url"]["url"].startswith("file://"))

    def test_both_runners_send_schema_and_images_through_real_sdk(self):
        real_http_client = httpx.Client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").write_bytes(b"sample-image")
            (root / "logicam").mkdir()
            prompt = "Official example\nImage: " + base64.b64encode(b"\xff\xd8\xffexample").decode() + "\nEnd"
            template = root / "prompt.txt"
            template.write_text(prompt)
            (root / "logicam/prompt.txt").write_text(prompt)
            for runner, structured, kwargs in (
                (baseline, {"reasoning": "P implies Q", "final_answer": "B"}, {"template_path": template}),
                (logicam, {"reasoning_steps": ["P implies Q"], "final_answer": "B"}, {"template_dir": root}),
            ):
                with self.subTest(runner=runner.__name__):
                    requests = []

                    def handle(request):
                        requests.append(request)
                        return completion(json.dumps(structured))

                    with patch("muslr_core.inference.httpx", SimpleNamespace(Client=lambda **kw: real_http_client(
                        transport=httpx.MockTransport(handle), **kw,
                    ))):
                        config = ModelConfig(endpoints=("http://127.0.0.1:8000/v1",), model="test-model", api_key="test-key")
                        with JSONClient(config, max_tokens=8192) as client:
                            result = runner.run_one(client, sample(), data_path=root, media_root=root / "media",
                                                    max_tokens=8192, **kwargs)
                    self.assertEqual(result["status"], "completed", result)
                    self.assertTrue(result["correct"])
                    self.assertEqual(result["usage"]["completion_tokens"], 20)
                    self.assertEqual(result["usage"]["endpoint"], config.endpoints[0])
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0].url.path, "/v1/chat/completions")
                    self.assertEqual(requests[0].headers["authorization"], "Bearer test-key")
                    payload = json.loads(requests[0].content)
                    self.assertEqual(payload["response_format"], runner.response_format_for(sample()))
                    self.assertEqual(payload["max_tokens"], 8192)
                    self.assertEqual(payload["temperature"], 0)
                    content = payload["messages"][-1]["content"]
                    images = [block["image_url"]["url"] for block in content if block["type"] == "image_url"]
                    self.assertEqual(len(images), 2)
                    self.assertTrue(all(url.startswith("data:image/") for url in images))
                    self.assertEqual(base64.b64decode(images[-1].partition(",")[2]), b"sample-image")

    def test_provider_failure_is_single_attempt_and_releases_endpoint_slot(self):
        real_http_client = httpx.Client
        requests = []

        def handle(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, json={"error": {"message": "busy", "type": "rate_limit"}})
            return completion(None)

        with patch("muslr_core.inference.httpx", SimpleNamespace(Client=lambda **kw: real_http_client(
            transport=httpx.MockTransport(handle), **kw,
        ))):
            config = ModelConfig(endpoints=("http://127.0.0.1:8000/v1",), model="test", pool_concurrency=1)
            with JSONClient(config, max_tokens=32) as client:
                http_client = client._clients[0]._client
                with self.assertRaises(RateLimitError):
                    client.complete_with_metadata([], response_format={"type": "json_object"})
                self.assertEqual(len(requests), 1)
                self.assertEqual(client._slots.qsize(), 1)
                # Empty final content stays empty; never substitute the reasoning channel.
                result = client.complete_with_metadata([], response_format={"type": "json_object"})
                self.assertEqual(result.output, "")
            self.assertTrue(http_client.is_closed)

    def test_multiple_endpoints_are_used(self):
        real_http_client = httpx.Client
        with patch("muslr_core.inference.httpx", SimpleNamespace(Client=lambda **kw: real_http_client(
            transport=httpx.MockTransport(lambda request: completion("{}")), **kw,
        ))):
            config = ModelConfig(endpoints=("http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"),
                                 model="test", pool_concurrency=1)
            self.assertEqual(config.max_workers, 2)
            with JSONClient(config, max_tokens=32) as client:
                endpoints = [client.complete_with_metadata([], response_format={"type": "json_object"}).endpoint
                             for _ in range(2)]
            self.assertEqual(tuple(endpoints), config.endpoints)

    def test_explicit_endpoint_required(self):
        for endpoints in ((), ("auto",), ("file:///tmp/model",)):
            with self.subTest(endpoints=endpoints), self.assertRaisesRegex(ValueError, "explicit --endpoints"):
                ModelConfig(endpoints=endpoints, model="test")


if __name__ == "__main__":
    unittest.main()
