#!/usr/bin/env python3
"""Compatibility entry point for existing free-text experiment processes/scripts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legacy.free_text_run import main

if __name__ == "__main__":
    main()
