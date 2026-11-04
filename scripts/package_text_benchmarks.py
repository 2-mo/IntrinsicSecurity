#!/usr/bin/env python3
"""Export independent CoT-Control QA and ProcessBench HTTP evaluation bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def export(name: str, runner: str, extras: list[str], output_dir: Path) -> Path:
    files = [f"{runner}/{filename}" for filename in ("run.py", "test_run.py", "requirements.txt")]
    files.extend(("benchmark_common/__init__.py", "benchmark_common/runtime.py"))
    files.extend(extras)
    path = output_dir / f"{name}-standalone.zip"
    prefix = f"{name}-standalone/"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(ROOT / runner / "README.md", prefix + "README.md")
        archive.write(ROOT / runner / ".env.example", prefix + ".env.example")
        archive.writestr(prefix + ".gitignore", ".env\n.venv/\noutputs/\n__pycache__/\n*.py[cod]\n")
        for filename in sorted(files):
            archive.write(ROOT / filename, prefix + filename)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/share")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cot = "datasets/CoT-Control/"
    cot_files = [cot + name for name in ("LICENSE", "README.md", "download_manifest.json")]
    cot_files.extend(cot + "CoT-Control-QA/" + name for name in (
        "README.md", "run_cceval.py", "llm.py", "prompts.py", "grading.py",
        "datasets/gpqa_w_keyword.csv", "datasets/hle_w_keyword.csv", "datasets/mmlu_pro_mini_w_keyword.csv",
    ))
    process_files = ["datasets/ProcessBench/" + name for name in (
        "README.md", "download_manifest.json", "gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json",
    )]
    process_files.append("myProcessBench/upstream/templates/critique_template.txt")
    for name, runner, files in (
        ("CoT-Control", "myCoTControl", cot_files),
        ("ProcessBench", "myProcessBench", process_files),
    ):
        print(export(name, runner, files, args.output_dir))


if __name__ == "__main__":
    main()
