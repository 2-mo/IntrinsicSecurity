#!/usr/bin/env python3
"""Run the official MuSLR baseline or LogiCAM prompt against an OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from muslr import CompletionClient, load_samples, metadata_for_samples, run_samples, selected_samples
from ripo_batch import RipoBatchClient


DEFAULT_ENDPOINTS = (
    "http://127.0.0.1:22000/v1",
    "http://127.0.0.1:22001/v1",
    "http://127.0.0.1:22002/v1",
)


def default_template_dir() -> Path:
    env_root = os.environ.get("MUSLR_SOURCE_ROOT")
    if env_root:
        return Path(env_root) / "prompts"
    return Path(__file__).resolve().parents[1].parent / "MuSLR" / "prompts"


def default_local_media_root() -> Path:
    value = os.environ.get("MUSLR_LOCAL_MEDIA_ROOT")
    if value:
        return Path(value)
    return Path("/data4/r1/workspace/anom-pi-engine/outputs/muslr_media")


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def parse_endpoints(value: str) -> tuple[str, ...]:
    endpoints = tuple(part.strip() for part in value.split(",") if part.strip())
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    return endpoints


def default_image_root(data_path: Path, explicit_image_root: Path | None) -> Path | None:
    if explicit_image_root is not None:
        return explicit_image_root
    if data_path.is_dir() and (data_path / "images").is_dir():
        return data_path / "images"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="GitHub JSON or Hugging Face JSON export.")
    parser.add_argument("--output", required=True, type=Path, help="Checkpointed JSONL output path.")
    parser.add_argument("--method", choices=("baseline", "logicam"), required=True)
    parser.add_argument("--image-root", type=Path, help="Directory containing GitHub image files.")
    parser.add_argument("--template-dir", type=Path, default=default_template_dir())
    parser.add_argument(
        "--media-mode",
        choices=("local", "data-url"),
        default="local",
        help="Use local file:// image paths by default; data-url is available as a fallback.",
    )
    parser.add_argument(
        "--local-media-root",
        type=Path,
        default=default_local_media_root(),
        help="Local mirror directory under vLLM --allowed-local-media-path.",
    )
    parser.add_argument("--batch-backend", choices=("ripo", "openai"), default="ripo")
    parser.add_argument(
        "--endpoints",
        default=",".join(DEFAULT_ENDPOINTS),
        help="Comma-separated RIPO endpoints; each endpoint maps to one serving process.",
    )
    parser.add_argument("--ripo-root", type=Path, default=Path("/data4/r1/workspace/cwork/projects/RIPO"))
    parser.add_argument("--endpoint-concurrency", type=int, default=1)
    parser.add_argument("--base-url", default=DEFAULT_ENDPOINTS[0])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default="anom-pi")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3, help="Outer batch workers; use 3 for the three serving endpoints.")
    parser.add_argument("--ids", help="Comma-separated IDs to run.")
    parser.add_argument("--domains", help="Comma-separated domain filter, case-insensitive.")
    parser.add_argument("--symbols", help="Comma-separated logic symbols, such as pl,nm,fol.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Limit selected samples after filters and sharding.")
    parser.add_argument("--no-resume", action="store_true", help="Do not retain prior JSONL records.")
    parser.add_argument(
        "--retry-statuses",
        help="Comma-separated prior statuses to rerun in place, for example: unparsed,failed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and write metadata only.")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.endpoint_concurrency < 1:
        parser.error("--endpoint-concurrency must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    retry_statuses = parse_csv(args.retry_statuses)
    allowed_retry_statuses = {"unparsed", "failed", "skipped_missing_image"}
    if retry_statuses and not retry_statuses <= allowed_retry_statuses:
        parser.error(
            "--retry-statuses only supports: " + ",".join(sorted(allowed_retry_statuses))
        )
    if retry_statuses and args.no_resume:
        parser.error("--retry-statuses cannot be combined with --no-resume")
    if not args.template_dir.is_dir():
        parser.error(f"Prompt directory does not exist: {args.template_dir}")
    try:
        endpoints = parse_endpoints(args.endpoints)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_backend == "ripo" and args.media_mode != "local":
        parser.error("RIPO batch backend requires --media-mode local so images stay on local paths.")
    if args.batch_backend == "ripo" and not args.ripo_root.is_dir():
        parser.error(f"RIPO root does not exist: {args.ripo_root}")

    image_root = default_image_root(args.data, args.image_root)
    samples = load_samples(args.data)
    ids = {part.strip() for part in args.ids.split(",") if part.strip()} if args.ids else None
    selected = selected_samples(
        samples,
        ids=ids,
        domains=parse_csv(args.domains),
        symbols=parse_csv(args.symbols),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    if not selected:
        parser.error("No samples selected.")

    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_for_samples(selected, args_dict), indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} of {len(samples)} samples. Metadata: {metadata_path}")
    if args.dry_run:
        return

    if args.batch_backend == "ripo":
        client = RipoBatchClient(
            endpoints=endpoints,
            ripo_root=args.ripo_root,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            endpoint_concurrency=args.endpoint_concurrency,
        )
    else:
        client = CompletionClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    try:
        counters = run_samples(
            samples=selected,
            output_path=args.output,
            completion_client=client,
            method=args.method,
            template_dir=args.template_dir,
            dataset_dir=args.data if args.data.is_dir() else args.data.parent,
            image_root=image_root,
            media_mode=args.media_mode,
            local_media_root=args.local_media_root if args.media_mode == "local" else None,
            workers=args.workers,
            resume=not args.no_resume,
            retry_statuses=retry_statuses,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    print(json.dumps(counters, indent=2))
    if counters["failed"] or counters["skipped_missing_image"]:
        raise SystemExit("Inference contains failed requests or missing images; evaluation was not started.")


if __name__ == "__main__":
    main()
