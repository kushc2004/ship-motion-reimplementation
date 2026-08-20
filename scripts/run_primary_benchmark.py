#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shipmotion.benchmark import run_benchmark, write_comparison

root = Path(__file__).resolve().parents[1]
out_dir = root / "outputs" / "primary_benchmark_m37"
result = run_benchmark(root, out_dir)
table, comparison = write_comparison(result, out_dir)
print(table.to_string(index=False))
print(comparison)
