"""Run fresh Qwen3-VL-8B inference and evaluate its predictions on MuSLR."""

import argparse
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from engine.backend import Backend
from engine.dataset import questions, references
from engine.method import infer

HERE = Path(__file__).resolve().parent
MODEL = "Qwen3-VL-8B-Instruct"
OFFICIAL_SHA = "31bf227e30193a39d4e699470f81d8b7c0b2c6fa8de2b8c1acf07c633f2fa01d"
ENCODER = "sentence-transformers/all-mpnet-base-v2"
ENCODER_REV = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DEFAULT_WEIGHTS = HERE / "official_weights.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_official_images(directory, samples):
    expected = "1883bf10ba0667f00ba9f8b7197ea2869754e166456dde24d9d3ba50a922a066"
    fingerprint = hashlib.sha256()
    for path in sorted({s.image for s in samples}, key=lambda p: p.as_posix()):
        relative = path.relative_to(directory).as_posix()
        fingerprint.update(f"{digest(path)}  {relative}\n".encode())
    if fingerprint.hexdigest() != expected:
        raise ValueError("Official images differ from the pinned MuSLR revision")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load_weights(path):
    weights = json.loads(path.read_text())
    expected = json.loads(DEFAULT_WEIGHTS.read_text())
    if weights.keys() != expected.keys():
        raise ValueError("Both task types require weights")
    for task, values in weights.items():
        if (
            values.keys() != expected[task].keys()
            or not all(
                type(v) in (int, float) and 0 <= v <= 1e6 for v in values.values()
            )
            or not any(values.values())
        ):
            raise ValueError(f"Invalid weights for {task}")
    return weights


def predict(args, samples, weights):
    backend = Backend(args.endpoint, MODEL, args.workers)
    name = "features" if args.stage == "calibrate" else "predictions"
    final = args.output / (name + ".jsonl")
    partial = args.output / (name + ".partial.jsonl")
    records = {}
    try:
        with (
            ThreadPoolExecutor(max_workers=args.workers) as pool,
            partial.open("x", encoding="utf-8") as stream,
        ):
            jobs = {
                pool.submit(infer, backend, s, weights, args.stage == "calibrate"): s.id
                for s in samples
            }
            try:
                for i, job in enumerate(as_completed(jobs), 1):
                    row = job.result()
                    records[row["id"]] = row
                    print(json.dumps(row, ensure_ascii=False), file=stream, flush=True)
                    if i == 1 or i % 25 == 0 or i == len(samples):
                        print(f"{i}/{len(samples)} {name}", flush=True)
            except BaseException:
                for job in jobs:
                    job.cancel()
                raise
        temp = final.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            for s in samples:
                print(json.dumps(records[s.id], ensure_ascii=False), file=stream)
        temp.replace(final)
        partial.unlink()
    finally:
        backend.client.close()
    return final


def score(args, samples, predictions, dataset_sha):
    import torch
    from huggingface_hub import snapshot_download
    from rich.console import Console
    from engine.scoring import Sample, compute_roscoe, reasoning_pairs, resolve_device

    refs = references(args.dataset)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != {s.id for s in samples}:
        raise ValueError(
            "Prediction IDs must cover every requested question exactly once"
        )
    gold = {}
    correct = 0
    for s in samples:
        row = by_id[s.id]
        labels = [v[0] for v in s.choices] or ["True", "False", "Unknown"]
        answer, reasoning = refs[s.id]
        if (
            row["prediction"] not in labels
            or not row["reasoning"].strip()
            or answer.upper() not in [v.upper() for v in labels]
        ):
            raise ValueError(f"Invalid prediction or reference for {s.id}")
        correct += row["prediction"].upper() == answer.upper()
        gold[s.id] = Sample(s.id, s.context, reasoning)
    pairs = reasoning_pairs(gold, by_id)
    if len(pairs) != len(samples):
        raise ValueError("Missing reference or generated explanation")
    encoder = args.encoder_path or snapshot_download(
        ENCODER,
        revision=ENCODER_REV,
        allow_patterns=[
            "config.json",
            "tokenizer*.json",
            "vocab.txt",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    )
    device = resolve_device(args.device)
    torch.set_num_threads(4)
    metric = compute_roscoe(
        pairs, model_name=str(encoder), device=device, batch_size=16, console=Console()
    )
    result = dict(
        model=MODEL,
        n=len(samples),
        correct=correct,
        accuracy=correct / len(samples),
        roscoe=metric["score"],
        roscoe_variant=metric["paper_score_mapping"],
        metadata_sha256=dataset_sha,
        predictions_sha256=digest(predictions),
        full_official=(dataset_sha == OFFICIAL_SHA and len(samples) == 1093),
    )
    write_json(args.output / "scores.json", result)
    print(
        f"{correct}/{len(samples)} | accuracy {result['accuracy']:.4%} | ROSCOE {result['roscoe']:.9f}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory with metadata.csv and original images",
    )
    p.add_argument(
        "--endpoint",
        action="append",
        help="vLLM URL ending in /v1. Repeat for multiple servers",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--output", type=Path, default=Path("runs/official"))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder-path", type=Path)
    p.add_argument(
        "--stage", choices=["run", "infer", "score", "calibrate"], default="run"
    )
    p.add_argument("--limit", type=int, help="First N rows, for a smoke test")
    p.add_argument(
        "--external",
        action="store_true",
        help="Use an external CSV with the same schema",
    )
    args = p.parse_args()
    args.endpoint = args.endpoint or [
        os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
    ]
    dataset_sha = digest(args.dataset / "metadata.csv")
    if not args.external and dataset_sha != OFFICIAL_SHA:
        p.error("Official metadata.csv does not match the pinned MuSLR revision")
    if args.stage == "calibrate" and (not args.external or dataset_sha == OFFICIAL_SHA):
        p.error("calibrate expects an external dataset")
    samples = questions(args.dataset)
    if dataset_sha == OFFICIAL_SHA:
        verify_official_images(args.dataset, samples)
    if (not args.external and len(samples) != 1093) or len(
        {s.id for s in samples}
    ) != len(samples):
        p.error("Incomplete or duplicated questions")
    if (
        args.workers < 1
        or args.limit is not None
        and not 0 < args.limit <= len(samples)
    ):
        p.error("Invalid workers or limit")
    samples = samples[: args.limit]
    if not samples:
        p.error("Empty dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        predictions = args.output / "predictions.jsonl"
        if args.stage != "score":
            if any(x.name != ".lock" for x in args.output.iterdir()):
                p.error("Choose an empty output directory for a new inference run")
            weights = load_weights(args.weights)
            write_json(
                args.output / "run.json",
                dict(
                    model=MODEL,
                    metadata_sha256=dataset_sha,
                    n=len(samples),
                    weights=weights,
                    stage=args.stage,
                ),
            )
            predictions = predict(args, samples, weights)
        if args.stage in ("run", "score"):
            manifest = json.loads((args.output / "run.json").read_text())
            if (manifest["metadata_sha256"], manifest["n"]) != (
                dataset_sha,
                len(samples),
            ):
                p.error("Dataset differs from this prediction run")
            score(args, samples, predictions, dataset_sha)
        elif args.stage == "calibrate":
            from engine.calibration import select_weights

            select_weights(args.dataset, predictions, args.output / "weights.json")


if __name__ == "__main__":
    main()
