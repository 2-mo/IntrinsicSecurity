#!/usr/bin/env python3
"""Compatibility entry point for evaluation jobs launched before the package move."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation.evaluate import main

if __name__ == "__main__":
    main()
