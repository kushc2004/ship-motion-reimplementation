#!/usr/bin/env python3
"""Resumable, integrity-gated ship-motion pipeline (M0--M12).

This is an orchestrator, not a second implementation of the benchmark.  It
calls the existing milestone entry points, enforces their ordering, verifies
immutable artifacts between stages, and records enough state to resume after a
Kaggle timeout or native-process failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


M02_REQUIRED = (
    "data/manifests/data_audit.json",
    "data/manifests/data_audit.md",
    "data/manifests/raw_row_manifest.parquet",
    "data/manifests/run_manifest.parquet",
    *(f"data/splits/{dataset}_{name}" for dataset in ("BTP-1", "BTP-2") for name in (
        "raw.parquet", "summary.parquet", "interpolated.parquet",
        "train_scaler.json", "windows.parquet", "metadata.json",
    )),
)

REPORT_REQUIRED = (
    "reports/FINAL_BENCHMARK_REPORT.md",
    "cv/CV_CLAIMS.md",
    "figures/m11_accuracy_latency.png",
    "figures/m9_speed_ood.png",
    "figures/m11_phase_lag.png",
)

STAGE_ORDER = {"m02": 0, "primary": 1, "phase2": 2, "report": 3}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def accelerator(python_executable: str) -> dict:
    probe = """
import json, platform
try:
    import torch
except Exception as exc:
    print(json.dumps({'kind': 'unavailable', 'error': str(exc)}))
else:
    if torch.cuda.is_available():
        value = {'kind': 'cuda', 'name': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    elif torch.backends.mps.is_available():
        value = {'kind': 'mps', 'name': 'Apple Metal Performance Shaders', 'torch': torch.__version__}
    else:
        value = {'kind': 'cpu', 'name': platform.processor() or platform.machine(), 'torch': torch.__version__}
    print(json.dumps(value))
"""
    result = subprocess.run(
        [python_executable, "-c", probe], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        value = {"kind": "unavailable", "error": result.stderr.strip() or "accelerator probe failed"}
    return value


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class PipelineError(RuntimeError):
    pass


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_root = args.output_root.resolve()
        self.state_dir = self.output_root / "orchestration"
        self.log_dir = self.state_dir / "logs"
        self.status_path = self.state_dir / "end_to_end_status.json"
        self.environment = os.environ.copy()
        self.environment.update({
            "PYTHONPATH": str(SRC) + os.pathsep + self.environment.get("PYTHONPATH", ""),
            "SHIPMOTION_OUTPUT_ROOT": str(self.output_root),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        })
        self.state = {
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "repository": str(ROOT),
            "git_commit": git_commit(),
            "output_root": str(self.output_root),
            "python": sys.version,
            "accelerator": accelerator(args.python),
            "arguments": vars(args),
            "steps": [],
        }

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        if not self.args.dry_run:
            atomic_json(self.status_path, self.state)

    def record(self, name: str, status: str, **extra: object) -> None:
        entry = {"name": name, "status": status, "at": utc_now(), **extra}
        self.state["steps"].append(entry)
        self.save()

    def run_command(self, name: str, command: list[str]) -> None:
        rendered = " ".join(command)
        if self.args.dry_run:
            print(f"[dry-run] {name}: {rendered}", flush=True)
            self.record(name, "planned", command=command)
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{len(self.state['steps']) + 1:02d}_{name}.log"
        print(f"\n==> {name}\n$ {rendered}\nlog: {log_path}", flush=True)
        started = time.monotonic()
        self.record(name, "started", command=command, log=str(log_path))
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=self.environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            returncode = process.wait()
        elapsed = time.monotonic() - started
        if returncode:
            self.record(name, "failed", command=command, returncode=returncode,
                        elapsed_s=elapsed, log=str(log_path))
            raise PipelineError(f"{name} failed with exit code {returncode}; see {log_path}")
        self.record(name, "complete", command=command, returncode=0,
                    elapsed_s=elapsed, log=str(log_path))

    def require_files(self, name: str, base: Path, relative_paths: Iterable[str]) -> None:
        if self.args.dry_run:
            self.record(name, "planned", base=str(base), files=list(relative_paths))
            return
        missing = [str(base / item) for item in relative_paths if not (base / item).is_file()]
        if missing:
            raise PipelineError(f"{name} is incomplete; missing: " + ", ".join(missing))
        self.record(name, "verified", files=len(tuple(relative_paths)))

    def test(self, name: str, test_file: str) -> None:
        if self.args.skip_tests:
            self.record(name, "skipped", reason="--skip-tests")
            return
        self.run_command(name, [self.args.python, "-m", "pytest", "-q", test_file])

    def stage_status(self, stage: str) -> dict:
        result = subprocess.run(
            [self.args.python, "scripts/check_cached_stage.py", stage],
            cwd=ROOT, env=self.environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise PipelineError(f"could not inspect {stage} cache: {detail}") from exc

    def verify_stage(self, stage: str) -> None:
        if self.args.dry_run:
            self.record(f"verify_{stage}", "planned")
            return
        status = self.stage_status(stage)
        if not status["complete"]:
            raise PipelineError(f"{stage} artifact verification failed: {status['reason']}")
        self.record(f"verify_{stage}", "verified", path=status["path"])

    def preflight(self) -> None:
        if not (ROOT / "SHIP_MOTION_REIMPLEMENTATION_SPEC.md").is_file():
            raise PipelineError("SHIP_MOTION_REIMPLEMENTATION_SPEC.md is missing")
        found = self.state["accelerator"]["kind"]
        required = self.args.require_accelerator
        if STAGE_ORDER[self.args.through_stage] >= STAGE_ORDER["primary"] and found == "unavailable":
            raise PipelineError(
                "PyTorch is unavailable in this Python environment; install "
                "requirements-kaggle.txt or pass --python for an environment containing the benchmark dependencies"
            )
        if required != "none" and found != required:
            raise PipelineError(f"required accelerator {required!r}, but detected {found!r}")
        self.record("preflight", "verified", accelerator=self.state["accelerator"])

    def m02(self) -> None:
        if self.args.rebuild_m02:
            self.run_command("build_m02", [self.args.python, "scripts/build_m02_artifacts.py"])
        self.require_files("verify_m02_files", ROOT, M02_REQUIRED)
        self.test("test_m02_leakage_gates", "tests/test_m02.py")

    def primary(self) -> None:
        status = self.stage_status("primary") if not self.args.dry_run else {"complete": False}
        if status["complete"] and not self.args.force_primary:
            print(f"Reusing verified primary artifact: {status['path']}", flush=True)
            self.record("run_primary", "reused", path=status["path"])
        else:
            command = [self.args.python, "scripts/run_primary_benchmark.py"]
            if self.args.force_primary:
                command.append("--force")
            self.run_command("run_primary", command)
        self.verify_stage("primary")
        self.test("test_m37_contract", "tests/test_m37.py")

    def phase2(self) -> None:
        self.verify_stage("primary")
        status = self.stage_status("phase2") if not self.args.dry_run else {"complete": False}
        if status["complete"] and not self.args.force_phase2:
            print(f"Reusing verified phase-2 artifact: {status['path']}", flush=True)
            self.record("run_phase2", "reused", path=status["path"])
        else:
            command = [self.args.python, "scripts/run_m8_m12.py"]
            if self.args.force_phase2:
                command.append("--force")
            self.run_command("run_phase2", command)
        self.verify_stage("phase2")
        self.test("test_m812_contract", "tests/test_m812.py")

    def report(self) -> None:
        self.verify_stage("phase2")
        self.run_command("export_final_report", [self.args.python, "scripts/export_final_report.py"])
        self.run_command("export_cv_claims", [self.args.python, "scripts/export_cv_claims.py"])
        self.require_files("verify_report_files", self.output_root, REPORT_REQUIRED)

    def execute(self) -> None:
        self.preflight()
        actions = (("m02", self.m02), ("primary", self.primary),
                   ("phase2", self.phase2), ("report", self.report))
        for name, action in actions:
            if STAGE_ORDER[name] <= STAGE_ORDER[self.args.through_stage]:
                action()
        self.state["status"] = "complete"
        self.state["completed_at"] = utc_now()
        self.save()
        label = "Dry-run plan complete" if self.args.dry_run else "End-to-end pipeline complete"
        print(f"\n{label} through {self.args.through_stage}.", flush=True)
        if not self.args.dry_run:
            print(f"Status: {self.status_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-stage", choices=tuple(STAGE_ORDER), default="report",
                        help="last milestone group to execute (default: report/M12)")
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("SHIPMOTION_OUTPUT_ROOT", ROOT / "outputs")))
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child stages")
    parser.add_argument("--rebuild-m02", action="store_true",
                        help="explicitly regenerate M0--M2; never done implicitly")
    parser.add_argument("--force-primary", action="store_true", help="ignore verified M3--M7 cache")
    parser.add_argument("--force-phase2", action="store_true", help="ignore verified M8--M11 cache")
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest gates (not recommended)")
    parser.add_argument("--require-accelerator", choices=("none", "cuda", "mps"), default="none")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without executing them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        runner = Runner(args)
        runner.execute()
        return
    state_dir = args.output_root.resolve() / "orchestration"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "end_to_end.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another end-to-end run holds {lock_path}") from exc
        runner = Runner(args)
        runner.save()
        try:
            runner.execute()
        except BaseException as exc:
            runner.state["status"] = "failed"
            runner.state["failed_at"] = utc_now()
            runner.state["error"] = f"{type(exc).__name__}: {exc}"
            runner.save()
            raise


if __name__ == "__main__":
    main()
