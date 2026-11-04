#!/usr/bin/env python3
"""ProcessBench critic evaluation using the official template and metric.

Reports exact first-error accuracy, no-error accuracy, and their harmonic mean
(called F1 in the official code). HTTP requests checkpoint to resumable JSONL.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
sys.path.insert(0, str(ROOT))
from myCoTControl.run import read_checkpoint, split_reasoning

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")


def extract_answer(text, steps):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    try:
        value = int(matches[-1].strip())
    except ValueError:
        return None
    return value if -1 <= value < steps else None


def metrics(rows, expected):
    groups = {}
    for split in SPLITS:
        group = [r for r in rows if r["split"] == split]
        if not group:
            continue
        error = [r for r in group if r["label"] != -1]
        correct = [r for r in group if r["label"] == -1]
        a = sum(r.get("match", False) for r in error) / len(error) if error else None
        b = sum(r.get("match", False) for r in correct) / len(correct) if correct else None
        f1 = 2 * a * b / (a + b) if a is not None and b is not None and a + b else (
            0.0 if a is not None and b is not None else None)
        groups[split] = {
            "expected": expected[split], "completed": len(group),
            "error_samples": len(error), "correct_samples": len(correct),
            "error_accuracy": a, "correct_accuracy": b, "f1": f1,
            "exact_match_accuracy": sum(r.get("match", False) for r in group) / len(group),
            "parsed": sum(r.get("prediction") is not None for r in group),
            "failed": sum(r["status"] == "failed" for r in group),
            "truncated": sum(r.get("finish_reason") == "length" for r in group),
        }
    f1s = [g["f1"] for g in groups.values() if g["f1"] is not None]
    return {"groups": groups, "mean_f1": sum(f1s) / len(f1s) if f1s else None,
            "metric_definition": "Official harmonic mean of exact-error-index accuracy and no-error accuracy; all requests in denominator",
            "complete": all(len([r for r in rows if r["split"] == s]) == n for s, n in expected.items())}


def generate(client, task, model, max_tokens):
    started = time.monotonic()
    row = {k: task[k] for k in ("id", "split", "label")}
    try:
        result = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": task["prompt"]}],
            temperature=0, seed=42, max_tokens=max_tokens)
        raw = result.model_dump()
        choice = raw["choices"][0]
        reasoning, final, source = split_reasoning(choice["message"])
        prediction = extract_answer(final, task["steps"])
        row.update(status="ok", prediction=prediction, match=prediction == row["label"],
                   generated_critique=final, reasoning=reasoning, reasoning_source=source,
                   finish_reason=choice.get("finish_reason"), usage=raw.get("usage"), raw_response=raw)
    except Exception as error:
        row.update(status="failed", prediction=None, match=False, error=f"{type(error).__name__}: {error}")
    row["latency_seconds"] = time.monotonic() - started
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/ProcessBench")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/intern_s2/processbench")
    p.add_argument("--base-url", default="http://127.0.0.1:22100/v1")
    p.add_argument("--model", default="Intern-S2-Preview")
    p.add_argument("--limit", type=int, default=0, help="Per split; 0=all; limited runs balance error/no-error")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    if args.limit < 0 or args.workers < 1 or args.max_tokens < 1:
        p.error("Invalid limit/workers/max-tokens")
    template = (Path(__file__).parent / "upstream/templates/critique_template.txt").read_text().strip()
    tasks, expected = [], {}
    for split in SPLITS:
        data = json.loads((args.data / f"{split}.json").read_text())
        if args.limit:
            error = [r for r in data if r["label"] != -1]
            correct = [r for r in data if r["label"] == -1]
            selected = error[:(args.limit + 1) // 2] + correct[:args.limit // 2]
        else:
            selected = data
        expected[split] = len(selected)
        for row in selected:
            tagged = "\n\n".join(f"<paragraph_{i}>\n{step}\n</paragraph_{i}>" for i, step in enumerate(row["steps"]))
            tasks.append({"id": row["id"], "split": split, "label": row["label"],
                          "steps": len(row["steps"]),
                          "prompt": template.format(problem=row["problem"], tagged_response=tagged)})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "predictions.jsonl"
    meta = args.output / "metadata.json"
    settings = {"model": args.model, "model_path": "/data3/Intern-S2-Preview",
                "temperature": 0, "seed": 42, "max_tokens": args.max_tokens,
                "official_code_revision": "e8024636bcabdf8bd514440551b531d3f90dd18b",
                "expected": expected, "task_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()}
    if path.exists() and (not meta.exists() or json.loads(meta.read_text()) != settings):
        p.error("Run settings changed or metadata missing; use a new --output directory")
    meta.write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)
    if args.dry_run:
        return
    rows = {}
    if path.exists():
        for row in read_checkpoint(path):
            rows[row["id"]] = row
    pending = [t for t in tasks if rows.get(t["id"], {}).get("status") != "ok"]
    if pending and not args.evaluate_only:
        with OpenAI(base_url=args.base_url, api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
                    max_retries=2, http_client=httpx.Client(trust_env=False, timeout=1800)) as client:
            models = [m.id for m in client.models.list().data]
            if args.model not in models:
                raise RuntimeError(f"Requested {args.model}, server offers {models}")
            with path.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(generate, client, task, args.model, args.max_tokens) for task in pending]
                for future in as_completed(futures):
                    row = future.result()
                    rows[row["id"]] = row
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{len(rows)}/{len(tasks)} {row['id']} {row['status']} match={row['match']}", flush=True)
    if not rows:
        raise RuntimeError("No predictions to evaluate")
    summary = metrics(list(rows.values()), expected)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"] or any(r["status"] == "failed" for r in rows.values()):
        raise RuntimeError("Incomplete run or request failures; rerun to resume")


if __name__ == "__main__":
    main()
