from __future__ import annotations

import hashlib
import json
import re
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MOTION_COLUMNS = ["surge_m", "sway_m", "heave_m", "roll_deg", "pitch_deg", "yaw_deg"]
CANONICAL_COLUMNS = ["run_id", "source_file", "time_s", "wave_m", "ship_speed_kn", *MOTION_COLUMNS]
INPUT_WINDOW = 100
OUTPUT_WINDOW = 50
DEFAULT_STRIDE = 10
DEFAULT_HZ = 10.0

ALIASES = {
    "time series": "time_s", "time[s]": "time_s", "time": "time_s",
    "wave[m]": "wave_m", "wave": "wave_m", "ship_speed_kn": "ship_speed_kn",
    "surge[m]": "surge_m", "sway[m]": "sway_m", "heave[m]": "heave_m",
    "roll[deg]": "roll_deg", "pitch[deg]": "pitch_deg", "yaw[deg]": "yaw_deg",
}


def inspect_csv_layout(path: Path) -> dict:
    """Inspect all CSV rows for embedded headers and side-by-side tables.

    ``pandas.read_csv`` assumes row zero is the header.  Several legacy exports
    instead put a condition label first, repeat an RAO header for each block, or
    place two metadata tables beside each other.  This inspection is descriptive
    only: it never promotes an RAO/metadata table to motion telemetry.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    def values(row):
        return {str(cell).strip().lower() for cell in row if str(cell).strip()}

    motion_header_aliases = {"time series", "time[s]", "time"} | {"wave[m]", "wave"} | {
        "surge[m]", "sway[m]", "heave[m]", "roll[deg]", "pitch[deg]", "yaw[deg]"
    }
    header_rows = []
    rao_rows = []
    metadata_rows = []
    for index, row in enumerate(rows):
        cells = values(row)
        if {"encounter freq. rad/s", "wave freq. rad/s", "wavelength m"}.issubset(cells):
            rao_rows.append(index)
            header_rows.append(index)
        if {"hydrostatic coefficients", "input parameters for data"} & cells or {"parameter", "values to use", "count"}.issubset(cells):
            metadata_rows.append(index)
            header_rows.append(index)
        if motion_header_aliases.intersection(cells) and (
            (cells & {"time series", "time[s]", "time"})
            and (cells & {"wave[m]", "wave"})
            and {"surge[m]", "sway[m]", "heave[m]", "roll[deg]", "pitch[deg]", "yaw[deg]"}.issubset(cells)
        ):
            header_rows.append(index)

    if rao_rows:
        table_type = "rao_blocks"
        details = {
            "block_count": len(rao_rows),
            "header_rows_zero_based": rao_rows,
            "block_labels": [rows[i - 1][0].strip() if i else "" for i in rao_rows],
        }
    elif metadata_rows:
        table_type = "metadata_tables"
        details = {
            "header_rows_zero_based": sorted(set(metadata_rows)),
            "side_by_side_tables": bool(rows and len(rows[0]) >= 7),
        }
    elif header_rows:
        table_type = "motion_telemetry"
        details = {"header_rows_zero_based": sorted(set(header_rows))}
    else:
        table_type = "unknown"
        details = {"header_rows_zero_based": sorted(set(header_rows))}
    return {"table_type": table_type, "row_count": len(rows), "max_columns": max((len(r) for r in rows), default=0), **details}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in frame.columns}
    out = frame.rename(columns=renamed).copy()
    missing = {"time_s", "wave_m", *MOTION_COLUMNS} - set(out.columns)
    if missing:
        raise ValueError(f"not a canonical motion dataset; missing {sorted(missing)}")
    if "ship_speed_kn" not in out:
        out["ship_speed_kn"] = np.nan
    for col in ["time_s", "wave_m", "ship_speed_kn", *MOTION_COLUMNS]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time_s", "wave_m", *MOTION_COLUMNS]).reset_index(drop=True)
    return out[[c for c in CANONICAL_COLUMNS if c not in {"run_id", "source_file"}]]


def load_canonical(path: Path) -> pd.DataFrame:
    return canonicalize(pd.read_csv(path))


def detect_runs(frame: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect runs only at within-file non-increasing timestamps.

    A speed change alone is not a boundary. It is metadata; a reset/non-increase
    is required, preventing accidental joins of simulations with reused clocks.
    """
    df = frame.copy()
    boundary = df["time_s"].diff().fillna(1.0).le(0)
    df["_run_number"] = boundary.cumsum().astype(int)
    stem = _slug(Path(source_file).stem)
    rows = []
    run_records = []
    for number, block in df.groupby("_run_number", sort=True):
        run_id = f"{stem}__run{number + 1:02d}"
        block = block.copy()
        block["run_id"] = run_id
        block["source_file"] = source_file
        block["source_row_index"] = block.index.astype(int)
        block["raw_timestep_id"] = block.apply(lambda r: f"{run_id}:{int(r.source_row_index):06d}", axis=1)
        if block["time_s"].diff().dropna().le(0).any():
            raise AssertionError(f"non-increasing time remains in {run_id}")
        speeds = block["ship_speed_kn"].dropna().unique().tolist()
        run_records.append({
            "run_id": run_id, "source_file": source_file, "run_number": number + 1,
            "raw_rows": len(block), "time_start_s": float(block.time_s.iloc[0]),
            "time_end_s": float(block.time_s.iloc[-1]),
            "vessel_speed_kn": float(speeds[0]) if len(speeds) == 1 else None,
            "speed_values_kn": json.dumps([float(x) for x in speeds]),
            "time_reset_at_start": bool(number > 0),
            "raw_row_start": int(block.source_row_index.iloc[0]),
            "raw_row_end": int(block.source_row_index.iloc[-1]),
        })
        rows.append(block.drop(columns="_run_number"))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(run_records)


def audit_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    layout = inspect_csv_layout(path)
    try:
        canonical = canonicalize(raw)
    except ValueError:
        return pd.DataFrame(), pd.DataFrame(), {"source_file": str(path), "eligible": False,
            "raw_columns": list(raw.columns), "layout": layout}
    rows, runs = detect_runs(canonical, str(path))
    dt = rows.groupby("run_id")["time_s"].diff().dropna()
    report = {
        "source_file": str(path), "eligible": True, "raw_rows": int(len(raw)),
        "canonical_rows": int(len(rows)), "columns": list(canonical.columns),
        "null_count": int(canonical[["time_s", "wave_m", *MOTION_COLUMNS]].isna().sum().sum()), "runs": int(len(runs)),
        "duplicate_run_time": int(rows.duplicated(["run_id", "time_s"]).sum()),
        "nonpositive_within_run_deltas": int((dt <= 0).sum()),
        "median_dt_s": float(dt.median()) if len(dt) else None,
        "speed_values_kn": sorted(rows["ship_speed_kn"].dropna().unique().tolist()),
        "layout": layout,
    }
    return rows, runs, report


def discover(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    all_rows, all_runs, reports = [], [], []
    for path in sorted(root.rglob("*.csv")):
        rows, runs, report = audit_file(path)
        reports.append(report)
        if report["eligible"]:
            all_rows.append(rows); all_runs.append(runs)
    return (pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(),
            pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame(), reports)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _json_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    purge_steps: int = 150
    hz: float = DEFAULT_HZ


def split_raw(rows: pd.DataFrame, strategy: str = "purged_temporal", config: SplitConfig = SplitConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, summary = [], []
    for run_id, block in rows.groupby("run_id", sort=True):
        block = block.sort_values("source_row_index").copy()
        if strategy == "run_heldout":
            raise ValueError("run_heldout is assigned by assign_run_splits")
        t0, t1 = float(block.time_s.iloc[0]), float(block.time_s.iloc[-1])
        train_end = t0 + (t1 - t0) * config.train_fraction
        val_end = t0 + (t1 - t0) * (config.train_fraction + config.validation_fraction)
        purge_s = config.purge_steps / config.hz
        labels = np.full(len(block), "excluded_purge", dtype=object)
        labels[block.time_s.to_numpy() <= train_end] = "train"
        labels[(block.time_s.to_numpy() > train_end + purge_s) & (block.time_s.to_numpy() <= val_end)] = "validation"
        labels[block.time_s.to_numpy() > val_end + purge_s] = "test"
        block["split"] = labels
        records.append(block)
        summary.append({"run_id": run_id, "strategy": strategy, "raw_before": len(block),
                        "train_raw": int((labels == "train").sum()),
                        "validation_raw": int((labels == "validation").sum()),
                        "test_raw": int((labels == "test").sum()),
                        "purged_raw": int((labels == "excluded_purge").sum()),
                        "purge_steps": config.purge_steps})
    return pd.concat(records, ignore_index=True), pd.DataFrame(summary)


def assign_run_splits(rows: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = runs.sort_values(["source_file", "run_number", "run_id"])["run_id"].tolist()
    if len(ordered) < 3:
        raise ValueError("run-heldout requires at least three independent runs")
    labels = {run: ("train" if i < len(ordered) - 2 else "validation" if i == len(ordered) - 2 else "test") for i, run in enumerate(ordered)}
    out = rows.copy(); out["split"] = out.run_id.map(labels)
    summary = out.groupby(["run_id", "split"], as_index=False).size().rename(columns={"size": "raw_after_purge"})
    summary["raw_before"] = summary["raw_after_purge"]; summary["purged_raw"] = 0; summary["strategy"] = "run_heldout"
    return out, summary


def interpolate_blocks(rows: pd.DataFrame, hz: float = DEFAULT_HZ) -> pd.DataFrame:
    """Interpolate each (run_id, split) block; never interpolate across a boundary."""
    dt = 1.0 / hz; outputs = []
    for (run_id, split), block in rows[rows.split != "excluded_purge"].groupby(["run_id", "split"], sort=False):
        block = block.sort_values("time_s").reset_index(drop=True)
        grid = np.arange(float(block.time_s.iloc[0]), float(block.time_s.iloc[-1]) + dt * 0.5, dt)
        out = pd.DataFrame({"run_id": run_id, "split": split, "time_s": grid})
        for col in ["wave_m", *MOTION_COLUMNS]: out[col] = np.interp(grid, block.time_s, block[col])
        out["ship_speed_kn"] = block.ship_speed_kn.iloc[0] if block.ship_speed_kn.notna().any() else np.nan
        pos = np.searchsorted(block.time_s.to_numpy(), grid, side="right").clip(1, len(block)) - 1
        out["raw_row_left"] = block.source_row_index.to_numpy()[pos]
        out["raw_row_right"] = block.source_row_index.to_numpy()[np.minimum(pos + 1, len(block) - 1)]
        out["source_file"] = block.source_file.iloc[0]
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def fit_train_scaler(interpolated: pd.DataFrame, path: Path) -> dict:
    train = interpolated[interpolated.split == "train"]
    features = ["wave_m", *MOTION_COLUMNS]
    if train["ship_speed_kn"].notna().any():
        features.insert(1, "ship_speed_kn")
    means = train[features].mean(); std = train[features].std(ddof=0).replace(0, 1.0).fillna(1.0)
    payload = {"features": features, "mean": means.to_dict(), "scale": std.to_dict(),
               "fit_split": "train", "fit_rows": int(len(train)),
               "fit_run_ids": sorted(train.run_id.unique().tolist()),
               "fit_data_hash": _json_hash(train[["run_id", "time_s", *features]].to_dict("records"))}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def make_windows(interpolated: pd.DataFrame, input_steps: int = INPUT_WINDOW, output_steps: int = OUTPUT_WINDOW, stride: int = DEFAULT_STRIDE) -> pd.DataFrame:
    records = []; width = input_steps + output_steps
    for (run_id, split), block in interpolated.groupby(["run_id", "split"], sort=False):
        block = block.sort_values("time_s").reset_index(drop=True)
        for start in range(0, max(0, len(block) - width + 1), stride):
            inp = block.iloc[start:start + input_steps]; target = block.iloc[start + input_steps:start + width]
            records.append({"window_id": f"{run_id}__{split}__{start:06d}", "run_id": run_id, "split": split,
                            "input_start_time_s": float(inp.time_s.iloc[0]), "input_end_time_s": float(inp.time_s.iloc[-1]),
                            "target_start_time_s": float(target.time_s.iloc[0]), "target_end_time_s": float(target.time_s.iloc[-1]),
                            "raw_input_start": int(inp.raw_row_left.min()), "raw_input_end": int(inp.raw_row_right.max()),
                            "raw_target_start": int(target.raw_row_left.min()), "raw_target_end": int(target.raw_row_right.max()),
                            "raw_coverage_start": int(min(inp.raw_row_left.min(), target.raw_row_left.min())),
                            "raw_coverage_end": int(max(inp.raw_row_right.max(), target.raw_row_right.max())),
                            "input_steps": input_steps, "output_steps": output_steps, "stride": stride})
    return pd.DataFrame(records)


def assert_no_leakage(windows: pd.DataFrame) -> None:
    for run_id, block in windows.groupby("run_id"):
        groups = {s: block[block.split == s] for s in block.split.unique()}
        labels = sorted(groups)
        for i, a in enumerate(labels):
            left = groups[a].sort_values("raw_coverage_start")
            for b in labels[i + 1:]:
                right = groups[b].sort_values("raw_coverage_start")
                if a >= b: continue
                if left.empty or right.empty: continue
                j = 0
                right_records = list(right[["raw_coverage_start", "raw_coverage_end"]].itertuples(index=False, name=None))
                for x in left[["raw_coverage_start", "raw_coverage_end"]].itertuples(index=False, name=None):
                    while j < len(right_records) and right_records[j][1] < x[0]:
                        j += 1
                    if j < len(right_records) and right_records[j][0] <= x[1]:
                        raise AssertionError(f"raw coverage overlap: {run_id} {a}/{b}")


def build_dataset_artifacts(root: Path, out: Path) -> dict:
    rows, runs, reports = discover(root)
    manifest = out / "manifests"; manifest.mkdir(parents=True, exist_ok=True)
    _write_table(runs, manifest / "run_manifest.parquet"); _write_table(rows, manifest / "raw_row_manifest.parquet")
    (manifest / "data_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    lines = ["# Data audit", "", "Generated from the repository CSVs; legacy result metrics are not included.", "", "## Eligible files", ""]
    for report in reports:
        if report["eligible"]:
            lines.append(f"- `{report['source_file']}`: {report['canonical_rows']} rows, {report['runs']} runs, columns `{', '.join(report['columns'])}`, speeds `{report['speed_values_kn']}`")
    lines += ["", "## Run manifest", "", "| run_id | source | rows | time start | time end | speed | reset |", "|---|---|---:|---:|---:|---:|---|"]
    for _, r in runs.iterrows():
        lines.append(f"| `{r.run_id}` | `{r.source_file}` | {r.raw_rows} | {r.time_start_s} | {r.time_end_s} | {r.vessel_speed_kn} | {r.time_reset_at_start} |")
    (manifest / "data_audit.md").write_text("\n".join(lines) + "\n")
    return {"rows": rows, "runs": runs, "reports": reports}
