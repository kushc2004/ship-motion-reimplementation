#!/usr/bin/env python3
"""Print JSON cache status for a benchmark stage; non-zero when not reusable."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipmotion.artifacts import cached_stage_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("primary", "phase2"))
    args = parser.parse_args()
    output_root = Path(os.environ.get("SHIPMOTION_OUTPUT_ROOT", ROOT / "outputs"))
    result = cached_stage_status(ROOT, output_root, args.stage)
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
