#!/usr/bin/env python3
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shipmotion.data.pipeline import (SplitConfig, _write_table, assign_run_splits,
                                      build_dataset_artifacts, fit_train_scaler,
                                      interpolate_blocks, make_windows, split_raw,
                                      assert_no_leakage)

root = Path(__file__).resolve().parents[1]
artifacts = build_dataset_artifacts(root, root / "data")
for dataset_name in ("BTP-1", "BTP-2"):
    source_mask = artifacts["rows"].source_file.str.contains("/" + dataset_name + "/", regex=False)
    run_mask = artifacts["runs"].source_file.str.contains("/" + dataset_name + "/", regex=False)
    rows = artifacts["rows"][source_mask].copy()
    runs = artifacts["runs"][run_mask].copy()
    if rows.empty: continue
    if len(runs) >= 3:
        split_rows, summary = assign_run_splits(rows, runs)
        strategy = "run_heldout"
    else:
        split_rows, summary = split_raw(rows, config=SplitConfig())
        strategy = "purged_temporal"
    base = root / "data" / "splits" / dataset_name
    _write_table(split_rows, base.with_name(base.name + "_raw.parquet"))
    _write_table(summary, base.with_name(base.name + "_summary.parquet"))
    interp = interpolate_blocks(split_rows)
    _write_table(interp, base.with_name(base.name + "_interpolated.parquet"))
    scaler = fit_train_scaler(interp, base.with_name(base.name + "_train_scaler.json"))
    windows = make_windows(interp)
    assert_no_leakage(windows)
    _write_table(windows, base.with_name(base.name + "_windows.parquet"))
    (base.with_name(base.name + "_metadata.json")).write_text(json.dumps({
        "dataset": dataset_name, "strategy": strategy, "input_steps": 100,
        "output_steps": 50, "stride": 10, "hz": 10, "purge_steps": 150,
        "scaler_fit_split": "train", "window_count": len(windows),
        "window_counts": windows.groupby("split").size().to_dict(),
        "run_ids": sorted(runs.run_id.tolist()), "scaler": scaler,
    }, indent=2, default=str) + "\n")
print(json.dumps({"runs": len(artifacts["runs"]), "rows": len(artifacts["rows"]),
                  "eligible_files": [r["source_file"] for r in artifacts["reports"] if r["eligible"]]}, indent=2))
