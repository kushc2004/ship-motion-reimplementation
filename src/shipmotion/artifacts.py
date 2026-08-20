"""Validation for reusable, immutable benchmark artifacts.

The cache is deliberately accepted only when the required output files exist
and the M0--M2 hashes recorded by the primary benchmark still match.  This
prevents a stale or partially written Kaggle output from being mistaken for a
finished benchmark.
"""
from __future__ import annotations

from pathlib import Path

from shipmotion.phase2 import verify_frozen_inputs


REQUIRED = {
    "primary": (
        "primary_benchmark_m37",
        (
            "manifest_hashes.json",
            "primary_comparison.csv",
            "metrics_by_dof.csv",
            "metrics_by_horizon.csv",
            "predictions.parquet",
            "transformer_comparison.json",
        ),
    ),
    "phase2": (
        "m8_m12",
        (
            "frozen_input_verification_before.json",
            "frozen_input_verification_after.json",
            "m8_feature_ablations.csv",
            "m9_speed_ood.csv",
            "m10_multitask_summary.json",
            "m11_phase_lag.csv",
        ),
    ),
}


def cached_stage_status(root: Path, output_root: Path, stage: str) -> dict:
    """Return a machine-readable, integrity-checked cache status."""
    if stage not in REQUIRED:
        raise ValueError(f"unknown stage: {stage}")
    directory_name, required = REQUIRED[stage]
    path = output_root / directory_name
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return {"complete": False, "stage": stage, "path": str(path), "reason": f"missing: {', '.join(missing)}"}
    try:
        verified = verify_frozen_inputs(root, output_root)
    except Exception as exc:
        return {"complete": False, "stage": stage, "path": str(path), "reason": f"immutable-input verification failed: {exc}"}
    return {"complete": True, "stage": stage, "path": str(path), "frozen_inputs": verified}
