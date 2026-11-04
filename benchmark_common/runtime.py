"""Checkpoint and response handling without benchmark or serving dependencies."""

import json
import re


def read_checkpoint(path):
    # U+2028/U+2029 inside a JSON string are data, not JSONL record boundaries.
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def split_reasoning(message):
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip(), content, "api_field"
    match = re.search(r"<(think|output_reasoning)>(.*?)</\1>", content, re.S)
    if match:
        return match[2].strip(), content[match.end():].strip(), "tagged_content"
    # Some templates prefill <think>, so generated text has only </think>.
    if "</think>" in content:
        reasoning, final = content.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), final.strip(), "prefilled_think"
    return "", content, "missing"
