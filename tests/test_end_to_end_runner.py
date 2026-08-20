import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_end_to_end as e2e


def args(tmp_path, **overrides):
    values = dict(
        through_stage="report", output_root=tmp_path, python=sys.executable,
        rebuild_m02=False, force_primary=False, force_phase2=False,
        skip_tests=False, require_accelerator="none", dry_run=True,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dry_run_plans_every_stage_without_mutating_benchmark_outputs(tmp_path):
    runner = e2e.Runner(args(tmp_path))
    runner.execute()
    planned = [step["name"] for step in runner.state["steps"]]
    assert runner.state["status"] == "complete"
    assert "build_m02" not in planned
    assert "run_primary" in planned
    assert "run_phase2" in planned
    assert "export_final_report" in planned
    assert not (tmp_path / "primary_benchmark_m37").exists()


def test_m02_rebuild_is_explicit_only(tmp_path):
    runner = e2e.Runner(args(tmp_path, through_stage="m02", rebuild_m02=True))
    runner.execute()
    assert "build_m02" in [step["name"] for step in runner.state["steps"]]


def test_atomic_status_is_valid_json(tmp_path):
    path = tmp_path / "status.json"
    e2e.atomic_json(path, {"status": "complete", "value": 7})
    assert json.loads(path.read_text()) == {"status": "complete", "value": 7}
    assert not path.with_suffix(".json.tmp").exists()
