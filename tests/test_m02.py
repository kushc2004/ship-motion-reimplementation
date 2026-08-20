import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from shipmotion.data.pipeline import (SplitConfig, assign_run_splits, assert_no_leakage,
    detect_runs, fit_train_scaler, inspect_csv_layout, interpolate_blocks, make_windows, split_raw)


def sample_rows():
    blocks = []
    for speed, start in [(2, 0), (5, 0), (8, 0)]:
        t = np.arange(0, 40, 0.2)
        blocks.append(pd.DataFrame({"time_s": t, "wave_m": np.sin(t), "ship_speed_kn": speed,
            "surge_m": t, "sway_m": t + 1, "heave_m": t + 2, "roll_deg": t + 3,
            "pitch_deg": t + 4, "yaw_deg": t + 5}))
    return pd.concat(blocks, ignore_index=True)


def test_reset_creates_runs_and_shared_clock_is_not_joined():
    rows, runs = detect_runs(sample_rows(), "synthetic.csv")
    assert len(runs) == 3
    assert rows.groupby("run_id").size().tolist() == [200, 200, 200]
    assert rows.duplicated(["run_id", "time_s"]).sum() == 0
    assert rows.time_s.duplicated().sum() > 0


def test_run_heldout_never_crosses_runs():
    rows, runs = detect_runs(sample_rows(), "synthetic.csv")
    split, _ = assign_run_splits(rows, runs)
    assert split.groupby("run_id").split.nunique().max() == 1


def test_temporal_purge_and_raw_coverage_are_disjoint():
    rows, _ = detect_runs(sample_rows(), "synthetic.csv")
    split, summary = split_raw(rows, config=SplitConfig(purge_steps=10, hz=10))
    assert (summary.purged_raw > 0).all()
    interpolated = interpolate_blocks(split)
    windows = make_windows(interpolated, input_steps=5, output_steps=3, stride=2)
    assert_no_leakage(windows)
    for run_id, block in windows.groupby("run_id"):
        assert block.raw_coverage_start.min() >= 0
        source = rows[rows.run_id == run_id]
        assert block.raw_coverage_end.max() <= int(source.source_row_index.max())


def test_scaler_is_train_only(tmp_path):
    rows, _ = detect_runs(sample_rows(), "synthetic.csv")
    split, _ = split_raw(rows, config=SplitConfig(purge_steps=10, hz=10))
    interp = interpolate_blocks(split)
    path = tmp_path / "scaler.json"
    payload = fit_train_scaler(interp, path)
    assert payload["fit_split"] == "train"
    assert payload["fit_rows"] == int((interp.split == "train").sum())
    assert payload["fit_run_ids"] == sorted(interp[interp.split == "train"].run_id.unique())


def test_run_boundary_cannot_be_hidden_by_windowing():
    rows, _ = detect_runs(sample_rows(), "synthetic.csv")
    split, _ = split_raw(rows, config=SplitConfig(purge_steps=10, hz=10))
    interp = interpolate_blocks(split)
    windows = make_windows(interp, input_steps=5, output_steps=3, stride=1)
    assert windows.groupby("window_id").run_id.nunique().max() == 1
    assert windows.groupby("window_id").split.nunique().max() == 1


def test_legacy_csv_embedded_headers_are_audited_without_becoming_telemetry():
    root = Path(__file__).parents[1]
    rao = inspect_csv_layout(root / "BTP-1" / "Data.csv")
    assert rao["table_type"] == "rao_blocks"
    assert rao["block_count"] == 10
    assert rao["header_rows_zero_based"] == [1, 94, 187, 280, 373, 466, 559, 652, 745, 838]
    assert rao["block_labels"][0] == "2.5 kn; 0 deg"
    assert rao["block_labels"][-1] == "8 kn; 180 deg"

    metadata = inspect_csv_layout(root / "BTP-1" / "Ship-Parameters-data.csv")
    assert metadata["table_type"] == "metadata_tables"
    assert metadata["side_by_side_tables"] is True
