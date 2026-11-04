#!/usr/bin/env python3
"""Export portable MuSLR sources using an allowlist, without host deployments."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/share/myMuSLR-standalone.zip")
    args = parser.parse_args()
    source = ROOT / "myMuSLR"
    paths = [source / name for name in ("README.md", "requirements.txt", ".env.example", ".gitignore", "run.py", "evaluate.py")]
    paths.extend(
        path for path in (source / "src").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".html"}
        and "__pycache__" not in path.parts
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=path.relative_to(ROOT))
    print(f"Exported {len(paths)} source files: {args.output}")


if __name__ == "__main__":
    main()
