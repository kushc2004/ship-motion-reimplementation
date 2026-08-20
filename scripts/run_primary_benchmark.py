#!/usr/bin/env python3
"""Run M3--M7, or reuse a verified immutable artifact cache."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shipmotion.benchmark import run_benchmark, write_comparison
from shipmotion.artifacts import cached_stage_status

root = Path(__file__).resolve().parents[1]
output_root = Path(os.environ.get("SHIPMOTION_OUTPUT_ROOT", root / "outputs"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun even if the cached artifact verifies")
    args = parser.parse_args()
    status = cached_stage_status(root, output_root, "primary")
    if status["complete"] and not args.force:
        print(f"Reusing verified M3--M7 artifact: {status['path']}")
        return
    if not status["complete"]:
        print(f"M3--M7 cache unavailable: {status['reason']}; running benchmark.")
    out_dir = output_root / "primary_benchmark_m37"
    result = run_benchmark(root, out_dir)
    table, comparison = write_comparison(result, out_dir)
    print(table.to_string(index=False))
    print(comparison)


if __name__ == "__main__":
    main()
