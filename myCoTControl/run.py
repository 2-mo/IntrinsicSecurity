#!/usr/bin/env python3
"""Evaluate CoT-Control QA through local OpenAI-compatible HTTP endpoints.

Uses the official data loader, prompts, answer extractor and rule graders.
Like myMuSLR, requests run concurrently and results checkpoint to JSONL.
Default: first three eligible questions per dataset, baseline + word suppression.
Use --limit 0 for all eligible questions. Agent tasks and paid LLM judges are
outside this runner; rule compliance is not the full paper's judged metric.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "datasets/CoT-Control/CoT-Control-QA"
sys.path.insert(0, str(OFFICIAL))
from run_cceval import create_prompt_for_sample, get_control_value, load_evaluation_data
from llm import LLMResponse, extract_answer_from_response, format_messages
from grading import (
    GRADE_FUNC_MAP, grade_word_suppression, grade_repeat_sentences,
    grade_end_of_sentence, grade_meow_between_words,
)

DATASETS = {name: OFFICIAL / "datasets" / f"{name}_w_keyword.csv"
            for name in ("gpqa", "hle", "mmlu_pro_mini")}
MODES = ("baseline", "word_suppression", "multiple_word_suppression",
         "uppercase_thinking", "lowercase_thinking", "alternating_case",
         "repeat_sentences", "end_of_sentence", "meow_between_words")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_letter(sample):
    """Resolve labels without the upstream loader's silent fallback to A."""
    answer = sample["correct_answer"].strip()
    options = sample.get("options") or []
    if not options:
        raise ValueError(f"No multiple-choice options for sample {sample['id']}")
    if re.fullmatch(r"\(?[A-Z]\)?", answer.upper()):
        letter = answer.upper().strip("()")
        if ord(letter) - ord("A") < len(options):
            return letter
    matches = [i for i, option in enumerate(options) if str(option).strip() == answer]
    if not matches:
        raise ValueError(f"Unmatched label for sample {sample['id']}: {answer!r}")
    # Match the official first-match convention when a row duplicates an option.
    return chr(ord("A") + matches[0])


def split_reasoning(message):
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip(), content, "api_field"
    match = re.search(r"<(think|output_reasoning)>(.*?)</\1>", content, re.S)
    if match:
        return match[2].strip(), content[match.end():].strip(), "tagged_content"
    # Some templates prefill <think>, so decoded generated text has only </think>.
    if "</think>" in content:
        reasoning, final = content.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), final.strip(), "prefilled_think"
    return "", content, "missing"


def compliance(mode, reasoning, sample):
    if mode == "baseline" or not reasoning.strip():
        return None
    if mode in ("word_suppression", "multiple_word_suppression"):
        return grade_word_suppression(reasoning, sample["keyword"], sample.get("synonyms", []))
    if mode == "repeat_sentences":
        return grade_repeat_sentences(reasoning, get_control_value(mode, sample))
    if mode == "end_of_sentence":
        return grade_end_of_sentence(reasoning, get_control_value(mode, sample))
    if mode == "meow_between_words":
        return grade_meow_between_words(reasoning)
    return GRADE_FUNC_MAP[mode](reasoning)


def prepare(model, datasets, modes, limit):
    tasks, selection = [], {}
    for dataset in datasets:
        for mode in modes:
            samples = load_evaluation_data(DATASETS[dataset], False, mode)
            available = len(samples)
            samples = samples[:limit] if limit else samples
            selection[f"{dataset}/{mode}"] = {"eligible": available, "selected": len(samples)}
            for sample in samples:
                system, prompt = create_prompt_for_sample(
                    sample, mode, model, False, "none", False, False, False)
                tasks.append({
                    "id": f"{dataset}/{mode}/{sample['id']}",
                    "dataset": dataset, "mode": mode, "sample": sample,
                    "target": target_letter(sample), "question": prompt,
                    "messages": format_messages(prompt, system or None),
                })
    return tasks, selection


def infer(client, task, model, max_tokens, endpoint):
    start = time.monotonic()
    row = {k: task[k] for k in ("id", "dataset", "mode", "target", "question")}
    row.update(model=model, endpoint=endpoint)
    try:
        response = client.chat.completions.create(
            model=model, messages=task["messages"], max_tokens=max_tokens, temperature=0)
        raw = response.model_dump()
        choice = raw["choices"][0]
        reasoning, final, source = split_reasoning(choice["message"])
        answer = extract_answer_from_response(LLMResponse(content=final))
        if answer and ord(answer) - ord("A") >= len(task["sample"]["options"]):
            answer = None
        truncated = choice.get("finish_reason") == "length"
        score = compliance(task["mode"], reasoning, task["sample"]) if not truncated else None
        row.update(
            status="ok", reasoning=reasoning, response=final, reasoning_source=source,
            extracted_answer=answer, correctness=answer == task["target"],
            compliance=score, finish_reason=choice.get("finish_reason"),
            truncated=truncated, usage=raw.get("usage"), raw_response=raw,
            meta_discussion=None,
        )
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}",
                   correctness=False, compliance=None)
    row["latency_seconds"] = time.monotonic() - start
    return row


def summarize(rows, expected):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['dataset']}/{row['mode']}"].append(row)
    result = {}
    for key, members in groups.items():
        ok = [r for r in members if r["status"] == "ok"]
        scored = [r for r in members if r.get("compliance") is not None]
        controlled = members[0]["mode"] != "baseline"
        result[key] = {
            "completed": len(members), "expected": expected[key]["selected"],
            "failed": len(members) - len(ok),
            "accuracy": sum(r.get("correctness", False) for r in members) / len(members),
            "answer_coverage": sum(bool(r.get("extracted_answer")) for r in members) / len(members),
            "reasoning_coverage": sum(bool(r.get("reasoning")) for r in members) / len(members),
            "truncated": sum(r.get("truncated", False) for r in members),
            "rule_compliance_evaluable": len(scored),
            "rule_compliance_on_evaluable": sum(r["compliance"] for r in scored) / len(scored) if scored else None,
            "rule_compliance_all_requests": sum(r["compliance"] for r in scored) / len(members) if controlled else None,
            "correct_and_rule_compliant_all_requests": sum(
                r.get("correctness", False) and r.get("compliance") == 1 for r in members
            ) / len(members) if controlled else None,
        }
    return {"groups": result, "meta_discussion_judge": "not run",
            "label_policy": "official first exact match; GPQA row 74 has duplicate correct options",
            "scope": "CoT-Control QA; rule compliance only; not the paper's full judged metric",
            "missing_or_truncated_reasoning": "not evaluable; counts as unsuccessful in all-request compliance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Intern-S2-Preview")
    parser.add_argument("--model-path", type=Path,
                        default=Path("/data3/Intern-S2-Preview"))
    parser.add_argument("--endpoints", default="http://127.0.0.1:22000/v1")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["baseline", "word_suppression"])
    parser.add_argument("--limit", type=int, default=3, help="Per dataset/mode; 0 = all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=25000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cot_control/intern_s2_smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or args.max_tokens < 1:
        parser.error("limit >= 0, workers >= 1 and max-tokens >= 1 are required")
    if args.dry_run and args.evaluate_only:
        parser.error("--dry-run and --evaluate-only are mutually exclusive")
    endpoints = [x.strip().rstrip("/") for x in args.endpoints.split(",") if x.strip()]
    if not endpoints:
        parser.error("At least one endpoint is required")
    tasks, selection = prepare(args.model, args.datasets, args.modes, args.limit)
    spec = {"model": args.model, "model_path": str(args.model_path),
            "max_tokens": args.max_tokens, "temperature": 0,
            "task_hash": digest(tasks), "selection": selection,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "metadata.json"
    predictions = args.output / "predictions.jsonl"
    if predictions.exists() and meta_path.exists() and json.loads(meta_path.read_text()) != spec:
        parser.error("Run settings changed; choose a new --output directory")
    if predictions.exists() and not meta_path.exists():
        parser.error("Predictions exist without run metadata; choose a new --output directory")
    meta_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    (args.output / "prompts.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    print(json.dumps(selection, indent=2), flush=True)
    if args.dry_run:
        print(f"Dry run: {len(tasks)} requests prepared; no inference was performed.")
        return
    rows = {}
    if predictions.exists():
        for line in predictions.read_text().splitlines():
            row = json.loads(line)
            rows[row["id"]] = row
    clients = []
    try:
        pending = [t for t in tasks if rows.get(t["id"], {}).get("status") != "ok"]
        if not args.evaluate_only and pending:
            for endpoint in endpoints:
                # Local model traffic must not accidentally use an external proxy.
                client = OpenAI(base_url=endpoint, api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
                                timeout=args.timeout, max_retries=2,
                                http_client=httpx.Client(trust_env=False, timeout=args.timeout))
                clients.append(client)
                available = [m.id for m in client.models.list().data]
                if args.model not in available:
                    raise RuntimeError(f"{endpoint} serves {available}, requested {args.model!r}")
            with predictions.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(infer, clients[i % len(clients)], task,
                                       args.model, args.max_tokens, endpoints[i % len(endpoints)])
                           for i, task in enumerate(pending)]
                for future in as_completed(futures):
                    row = future.result()
                    rows[row["id"]] = row
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{len(rows)}/{len(tasks)} {row['id']} {row['status']}", flush=True)
        if not rows:
            raise RuntimeError("No predictions available; no evaluation metrics can be computed")
        selected_ids = {t["id"] for t in tasks}
        summary = summarize([r for k, r in rows.items() if k in selected_ids], selection)
        summary["complete"] = selected_ids <= rows.keys()
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if not summary["complete"] or any(r["status"] != "ok" for r in rows.values()):
            raise RuntimeError("Run incomplete or contains failed requests; see predictions.jsonl")
    except Exception as error:
        (args.output / "last_error.json").write_text(json.dumps({
            "error": f"{type(error).__name__}: {error}",
            "model_path_exists": args.model_path.is_dir(),
            "local_weight_files": len(list(args.model_path.glob("*.safetensors"))) + len(list(args.model_path.glob("*.bin"))),
            "endpoints": endpoints,
        }, indent=2) + "\n")
        raise
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    main()
