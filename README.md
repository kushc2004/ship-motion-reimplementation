# Ship-motion reimplementation

This repository is a reproducible reimplementation workspace for the ship-motion project. It includes the source datasets, legacy artifacts inspected during the audit, the M0--M2 leakage-control pipeline, immutable split/window manifests, and benchmark code.

## Run on Kaggle

Open [`notebooks/ship_motion_kaggle.ipynb`](notebooks/ship_motion_kaggle.ipynb) in a Kaggle notebook with a GPU accelerator enabled. The notebook clones this public repository, validates the immutable M0--M2 artifacts, runs the primary M3--M7 benchmark, and provides an opt-in cell for the later M8--M12 experiments.

The notebook uses CUDA on Kaggle, Apple MPS on a compatible Mac, and CPU otherwise. Its generated outputs remain outside version control.

For a single resumable M0--M12 command, use the end-to-end orchestrator. It
reuses only integrity-checked artifacts, writes per-stage logs/status, and never
rebuilds immutable M0--M2 inputs unless explicitly requested:

```bash
python scripts/run_end_to_end.py --require-accelerator cuda
```

On a later Kaggle session, mount/copy the previous artifact dataset to a
persistent output directory and pass it with `--output-root`. The same command
will skip verified completed stages. Use `--dry-run` to inspect the execution
plan, or `--through-stage primary` for M0--M7 only.

## Reproduce locally

```bash
python -m pip install -r requirements-kaggle.txt
python scripts/build_m02_artifacts.py
pytest -q
python scripts/run_primary_benchmark.py
```

The M0--M2 build is deterministic and writes raw-timeline split/window manifests under `data/`. Do not regenerate those artifacts when evaluating a frozen benchmark unless you intentionally begin a new audited protocol.

## Repository contents

- `BTP-1/`, `BTP-2/`: supplied source data and legacy source artifacts.
- `data/`: canonical data audit, run manifest, immutable split/window manifests, and train-only scalers.
- `docs/LEGACY_AUDIT.md`: verified audit of the legacy notebooks and source artifacts.
- `src/shipmotion/`: run-aware preprocessing, common evaluation, primary models, and later-phase utilities.
- `scripts/`: reproducible pipeline and benchmark entry points.
- `tests/`: leakage, manifest, and later-phase guard tests.

Generated model files, predictions, plots, and benchmark outputs are deliberately ignored. This README does not state benchmark conclusions; those require complete frozen-test artifacts.
