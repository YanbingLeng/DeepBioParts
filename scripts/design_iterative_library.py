#!/usr/bin/env python3
"""Run deficit-directed library selection with NRP-only generator feedback."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.library_design.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
