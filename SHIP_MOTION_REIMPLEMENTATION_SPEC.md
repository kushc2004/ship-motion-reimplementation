# ShipMotionBench
## Rigorous Reimplementation and Evaluation of Multi-Horizon 6-DoF Ship Motion Forecasting and Propeller Operability Prediction

**Document type:** Codex-ready implementation specification  
**Project origin:** B.Tech Project — Ship Motion Prediction / Multi-Task Temporal Transformer  
**Primary objective:** Rebuild the existing project into one reproducible, leakage-free benchmark whose results are technically defensible on a Data Science / ML CV  
**Core task:** 10-second multivariate history → 5-second future 6-DoF vessel-motion trajectory  
**Secondary task:** Propeller-operability proxy prediction through continuous risk regression and rare-event classification  
**Existing model families:** Persistence / simple statistical baselines, Random Forest, XGBoost, LSTM, Transformer, Multi-Task Transformer  
**Primary source artifacts to preserve:**  
- `BTP-PPT.pptx`
- `BTP_2_Report_22NA3AI39_Kush_Chaudhari (2).pdf`
- `ShipMotionTransformer_Colab (1) (3).ipynb`
- `XGBoost (1).ipynb`
- `LSTMMX (2).ipynb`
- `random_forest_ship_motion (2).ipynb`
- `BTP2_ShipMotion_Propeller_Transformer.ipynb`

---

# 0. Executive Summary

The existing B.Tech project has a strong problem statement and a useful breadth of models, but its current headline metrics should not be reused until the complete experiment is rerun under a common, leakage-free protocol.

The reimplementation must preserve the original scientific intent:

```text
Maxsurf / MOSES hydrodynamic simulation
        ↓
irregular JONSWAP-wave vessel telemetry
        ↓
10 Hz preprocessing
        ↓
10 s historical multivariate window
        ↓
forecast 5 s of:
surge / sway / heave / roll / pitch / yaw
```

and the Phase-2 extension:

```text
shared temporal representation
        ↓
6-DoF motion forecasting
+ continuous propeller-operability risk
+ binary operability-risk event
```

However, all new reported results must come from a **single canonical pipeline** that fixes the following known issues in the legacy notebooks:

1. Sliding windows were created before temporal splitting, causing heavily overlapping train/validation/test examples.
2. Some scalers were fit on the complete dataset before splitting.
3. The existing LSTM notebook trains on all sequences before defining its test subset.
4. Different model notebooks use different input feature sets.
5. Metrics are not consistently evaluated in the same physical/normalized space.
6. Some notebooks use stride 1 while others use stride 10.
7. Existing code can concatenate or interpolate across independent simulation runs.
8. The Phase-1 Transformer performs cubic interpolation of ship speed, although vessel speed is a discrete/run-level operating condition.
9. The Phase-2 loader can be unsafe if several simulation runs share timestamps because it sorts by time and drops duplicate timestamps without a persistent run identifier.
10. The realistic propeller-depth experiment contains no positive emergence events; the reported classification result uses an intentionally synthetic 0.2 m threshold.
11. The current emergence target is based on a heave-derived rolling proxy, not independently simulated propeller ventilation.
12. No strong persistence baseline is currently used to verify that the models beat trivial continuation of a smooth trajectory.
13. There is no uncertainty/confidence-interval analysis or run-level statistical comparison.
14. Test-set evaluation is not centralized, making accidental tuning against the final test set difficult to rule out.

The goal is **not** to reproduce the old 0.0095 RMSE or 43% improvement claim.

The goal is to produce a new benchmark where every number can be traced to:

```text
raw simulation run
→ immutable split manifest
→ training-only preprocessing
→ exact model/config/seed
→ saved checkpoint
→ physical-unit prediction file
→ common evaluator
→ final metrics report
```

The final CV claim must be generated only from these new artifacts.

---

# 1. Source-Derived Project Definition

The existing report and presentation establish the following intended project.

## 1.1 Vessel and simulation

The project uses a large container-vessel model with approximately:

```text
Length overall: ~350 m
Draft: 15 m
Operating speeds: 2, 5, 8 knots
Wave spectrum: JONSWAP
Wave regime: irregular/random seas
Simulation software: Maxsurf Motions / MOSES Strip Theory
```

The project forecasts the six rigid-body motions:

```text
surge [m]
sway [m]
heave [m]
roll [deg]
pitch [deg]
yaw [deg]
```

The source presentation states that simulation outputs are resampled to a 10 Hz grid.

## 1.2 Forecast formulation

Canonical horizon:

```text
sampling rate = 10 Hz

input window  = 100 timesteps = 10 seconds
output window =  50 timesteps =  5 seconds
```

This multi-horizon formulation must remain the primary task.

The model must predict the entire future trajectory:

```text
Y ∈ R^(50 × 6)
```

rather than only the final future state.

## 1.3 Existing Phase-1 model comparison

The original project compares:

```text
Random Forest
XGBoost
LSTM
Transformer
```

The presentation currently contains headline performance numbers for these models.

**Those old numbers are historical project artifacts only.**

They must not be copied into the new README, report, or CV unless the new benchmark independently reproduces them.

## 1.4 Existing Phase-2 model

The Phase-2 report defines a Multi-Task Temporal Transformer with:

```text
input:
  wave + 6-DoF history

shared Transformer encoder:
  d_model = 128
  n_heads = 8
  encoder layers = 4
  feed-forward dim = 512
  dropout = 0.1
  activation = GELU

task heads:
  6-DoF trajectory regression
  emergence-score regression
  emergence-flag classification

legacy loss weights:
  motion = 1.0
  emergence score = 0.5
  emergence flag = 0.3
```

The new project should reproduce this architecture as a **legacy-compatible baseline**, then evaluate it rigorously.

---

# 2. Mandatory Principles for Codex

Codex must follow these rules throughout the implementation.

### Rule 1 — Do not overwrite legacy evidence

Existing notebooks, PPTs, reports, data exports, and generated figures must remain untouched.

Place them under an explicit legacy/reference location if repository restructuring is required.

### Rule 2 — Never reuse a legacy metric as a new result

Every new result must be generated by the canonical evaluator.

### Rule 3 — Split before learning any distributional statistic

This includes:

- StandardScaler
- MinMaxScaler
- target normalization
- any learned imputation
- feature selection based on outcomes
- class weights
- threshold tuning
- hyperparameter selection

### Rule 4 — Simulation runs are independent experimental units

Never create one continuous artificial time series by stitching separate Maxsurf runs together.

### Rule 5 — No window may cross a run boundary

### Rule 6 — No raw timestep may belong to windows from two different splits

This must be unit-tested.

### Rule 7 — All models in the primary benchmark use the same data split, history, horizon, and input feature set

### Rule 8 — Evaluate all models in original physical units

Training losses may use standardized targets; reported ship-motion RMSE/MAE must be in meters/degrees.

### Rule 9 — Validation selects models; test only reports them

### Rule 10 — A trivial baseline is mandatory

A sophisticated model is not useful if it does not beat continuation/persistence.

### Rule 11 — Propeller classification claims must distinguish realistic and synthetic-threshold experiments

### Rule 12 — No claim that attention "learns RAOs" unless an actual quantitative attention/physics analysis is implemented

### Rule 13 — No claim of real-world onboard safety or deployment without corresponding deployment evidence

### Rule 14 — Every CV number must be traceable to an immutable result artifact

---

# 3. Known Problems in the Legacy Implementation

Codex should create `docs/LEGACY_AUDIT.md` and record these findings, verifying them against the actual notebooks before editing new code.

## 3.1 Sliding-window leakage

Legacy XGBoost and Random Forest create 100-step input + 50-step target windows using stride 1 and then split the windows chronologically.

Adjacent windows share almost all their underlying timesteps.

Example:

```text
Window A:
input  [0 ... 99]
target [100 ... 149]

Window B:
input  [1 ... 100]
target [101 ... 150]
```

If the train/test boundary occurs between neighboring windows, train and test samples can share approximately 149 of 150 covered raw timesteps.

The Transformer uses stride 10, which reduces but does not remove overlap.

The Phase-2 pipeline similarly creates windows before performing its train/validation/test split.

### Required fix

Split raw run timelines first.

Create windows separately inside each split.

Never split an already-windowed array.

---

## 3.2 Full-dataset scaler leakage

The Transformer preprocessing currently fits `StandardScaler` on the full interpolated series before creating/splitting windows.

Phase 2 similarly fits:

```text
feature scaler
emergence-score mean
emergence-score standard deviation
```

before the sequence split.

### Required fix

Fit all normalization parameters using the training partition only.

Persist:

```text
feature_scaler.pkl
target_scaler.pkl or target statistics
emergence_score_stats.json
```

with the training split hash.

---

## 3.3 Existing LSTM test contamination

The legacy LSTM notebook creates all sequences and trains:

```text
model.fit(X_lstm, y_lstm, ...)
```

before later defining an 80/20 subset for reporting.

Therefore the nominal final 20% has already been used for optimization.

### Required fix

Implement the LSTM from scratch in the canonical framework.

Do not reuse its old evaluation result.

---

## 3.4 Inconsistent feature sets

The current notebooks are not performing the same prediction task.

Examples from the legacy implementation:

```text
XGBoost:
primarily flattened historical 6-DoF

Random Forest:
flattened historical 6-DoF

Phase-1 Transformer:
ship speed + 6-DoF
(no wave channel in the current ALL_INPUT_FEATURES)

Phase-2 Transformer:
wave + 6-DoF
```

The presentation describes wave-driven forecasting, so direct model comparison is not valid if different models see different information.

### Required fix

Define one primary input schema and use it for every benchmark model.

Recommended schema when available:

```text
wave[m]
ship_speed_kn
surge[m]
sway[m]
heave[m]
roll[deg]
pitch[deg]
yaw[deg]
```

That is 8 features.

If `ship_speed_kn` is unavailable in the actual Phase-2 source file, use:

```text
wave + 6-DoF = 7 features
```

for the primary benchmark and run ship-speed conditioning only where valid.

Do not invent missing columns.

---

## 3.5 Run concatenation and interpolation problems

The Phase-1 Transformer contains logic that makes timestamps from separate runs globally increasing, then interpolates over the combined timeline.

This can produce artificial transition samples between independent simulation runs.

It also cubic-interpolates ship speed, even though speed is a run-level/discrete operating setting.

### Required fix

Process every simulation run independently.

Within a run:

```text
continuous signals:
  linear / cubic / PCHIP interpolation

ship speed:
  constant metadata or nearest-neighbor
```

Never interpolate from the end of one run into the beginning of another.

---

## 3.6 Phase-2 duplicate-time handling

The current Phase-2 loader sorts by time and executes a duplicate-time removal.

If one CSV contains several independent runs that each restart from `t = 0`, this can remove valid observations from other runs.

This is a conditional risk that must be verified against the real file.

### Required fix

Construct `run_id` before any duplicate handling.

Duplicate timestamps are only duplicates **within the same run**.

---

## 3.7 Metric-space inconsistency

Some legacy model outputs are evaluated in raw physical units while Transformer training/evaluation code also reports normalized-space metrics.

This makes old cross-model RMSE numbers non-comparable.

### Required fix

All benchmark result tables must come from one evaluator operating on denormalized:

```text
[N_windows, 50, 6]
```

arrays.

---

## 3.8 Propeller-emergence interpretation

The Phase-2 report states that using a realistic static depth of approximately `0.5 m` produced no emergence events in the test set.

The `0.2 m` experiment was intentionally introduced to produce positive cases for classification analysis.

The current target is a mathematical proxy derived from heave amplitude.

### Required fix

Represent the two experiments separately:

```text
REALISTIC_PROXY:
  H_static = 0.5 m
  report prevalence
  evaluate continuous score
  do not report F1 if only one class exists

SYNTHETIC_STRESS_TEST:
  H_static = 0.2 m
  clearly label threshold as synthetic
  classification metrics allowed
```

Never write:

```text
"99% F1 detecting real propeller ventilation"
```

unless genuine independent ventilation labels are available.

---

# 4. Target Repository Structure

Create a clean package alongside the legacy artifacts.

```text
ship-motion-benchmark/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── Makefile
├── configs/
│   ├── data/
│   │   ├── default.yaml
│   │   └── local.yaml
│   ├── split/
│   │   ├── purged_temporal.yaml
│   │   ├── run_holdout.yaml
│   │   └── leave_one_speed_out.yaml
│   ├── model/
│   │   ├── persistence.yaml
│   │   ├── linear_trend.yaml
│   │   ├── ridge.yaml
│   │   ├── random_forest.yaml
│   │   ├── xgboost.yaml
│   │   ├── lstm.yaml
│   │   ├── transformer.yaml
│   │   └── multitask_transformer.yaml
│   └── experiment/
│       ├── motion_primary.yaml
│       ├── feature_ablation.yaml
│       ├── speed_ood.yaml
│       ├── multitask_realistic.yaml
│       └── multitask_stress.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── splits/
├── legacy/
│   ├── notebooks/
│   ├── reports/
│   └── presentations/
├── src/
│   └── shipmotion/
│       ├── __init__.py
│       ├── data/
│       │   ├── schemas.py
│       │   ├── audit.py
│       │   ├── run_detection.py
│       │   ├── interpolation.py
│       │   ├── splitting.py
│       │   ├── scaling.py
│       │   ├── windowing.py
│       │   ├── dataset.py
│       │   └── manifest.py
│       ├── models/
│       │   ├── base.py
│       │   ├── persistence.py
│       │   ├── linear_trend.py
│       │   ├── ridge.py
│       │   ├── random_forest.py
│       │   ├── xgboost_model.py
│       │   ├── lstm.py
│       │   ├── transformer.py
│       │   └── multitask_transformer.py
│       ├── training/
│       │   ├── classical.py
│       │   ├── neural.py
│       │   ├── losses.py
│       │   ├── checkpointing.py
│       │   └── reproducibility.py
│       ├── evaluation/
│       │   ├── motion_metrics.py
│       │   ├── horizon_metrics.py
│       │   ├── phase_metrics.py
│       │   ├── operability_metrics.py
│       │   ├── bootstrap.py
│       │   ├── latency.py
│       │   └── report.py
│       ├── operability/
│       │   ├── legacy_proxy.py
│       │   ├── geometry_proxy.py
│       │   └── events.py
│       ├── reporting/
│       │   ├── figures.py
│       │   ├── tables.py
│       │   ├── cv_claims.py
│       │   └── markdown.py
│       └── utils/
│           ├── io.py
│           ├── hashing.py
│           ├── seed.py
│           └── logging.py
├── scripts/
│   ├── audit_data.py
│   ├── audit_legacy.py
│   ├── build_manifest.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_motion_benchmark.py
│   ├── run_multitask_benchmark.py
│   ├── run_speed_ood.py
│   ├── benchmark_latency.py
│   ├── export_report.py
│   └── export_cv_claims.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_motion_benchmark.ipynb
│   ├── 03_horizon_error_analysis.ipynb
│   ├── 04_speed_generalization.ipynb
│   └── 05_operability_analysis.ipynb
├── tests/
│   ├── test_run_detection.py
│   ├── test_interpolation.py
│   ├── test_splits.py
│   ├── test_window_leakage.py
│   ├── test_scaling.py
│   ├── test_model_shapes.py
│   ├── test_motion_metrics.py
│   ├── test_operability.py
│   └── test_reproducibility.py
└── outputs/
    ├── runs/
    ├── predictions/
    ├── metrics/
    ├── figures/
    ├── reports/
    └── cv/
```

---

# 5. Environment

Recommended Python:

```text
Python 3.11
```

Core dependencies:

```text
numpy
pandas
scipy
scikit-learn
pyarrow
matplotlib
plotly
pydantic
hydra-core
omegaconf
joblib
tqdm
rich
pytest
```

Models:

```text
torch
xgboost
```

Optional:

```text
mlflow
wandb
statsmodels
```

Do not copy dependency versions from old Colab notebooks.

Build a working environment first, then lock exact versions.

Support:

```text
CUDA — full neural benchmark
MPS — development/smoke test where supported
CPU — all unit tests and classical baselines
```

---

# 6. Canonical Data Schema

Create:

```python
@dataclass
class SimulationRun:
    run_id: str
    source_file: str
    vessel_speed_kn: float | None
    sea_state_id: str | None
    wave_seed: str | None
    metadata: dict
    dataframe: pd.DataFrame
```

Canonical columns:

```text
run_id
time_s
wave_m
ship_speed_kn        # optional but recommended
surge_m
sway_m
heave_m
roll_deg
pitch_deg
yaw_deg
```

Do not rename source columns destructively.

Keep a source-to-canonical mapping in the manifest.

---

# 7. Data Audit — Required First Step

Implement:

```bash
python scripts/audit_data.py data=<path>
```

Output:

```text
outputs/reports/data_audit.md
outputs/figures/run_lengths.png
outputs/figures/sampling_intervals.png
outputs/figures/signal_ranges.png
outputs/figures/speed_distribution.png
outputs/figures/correlation_matrix.png
```

The audit must report:

- input files,
- rows per file,
- inferred simulation runs,
- time resets,
- duplicate `(run_id, time)` values,
- sampling interval distribution,
- missing values,
- speed values,
- runs per speed,
- signal min/max/mean/std,
- wave range,
- all 6-DoF ranges,
- whether multiple runs share timestamp ranges,
- whether a `ship_speed_kn` column exists,
- whether propeller coordinates/diameter exist anywhere in metadata,
- whether sea-state identifiers/random seeds exist.

If run boundaries cannot be determined reliably, stop and ask for manual mapping rather than silently concatenating the data.

---

# 8. Run Detection

Preferred run identity order:

```text
1. explicit simulation/run ID
2. source filename
3. explicit simulation seed / sea-state metadata
4. combination of vessel speed + detected time reset
5. time-reset segmentation
```

Never use `time` alone as global identity.

Example manifest:

```text
run_id          source_file     speed_kn    start_s   end_s    n_raw
speed2_run01    data_2kn.csv    2           0.0       231.0    ...
speed5_run01    data_5kn.csv    5           0.0       231.0    ...
speed8_run01    data_8kn.csv    8           0.0       231.0    ...
```

If several runs use the same speed, assign unique run IDs.

---

# 9. Interpolation

## 9.1 Never interpolate across runs

Each run is resampled independently.

## 9.2 Target grid

Default:

```text
10 Hz
Δt = 0.1 s
```

## 9.3 Continuous channels

For:

```text
wave
surge
sway
heave
roll
pitch
yaw
```

support configurable:

```text
linear
cubic
PCHIP
```

Use the legacy cubic interpolation as one reproducibility option.

Recommended default for the rigorous benchmark:

```text
PCHIP or linear
```

because they avoid some spline overshoot risk.

The choice must be reported.

## 9.4 Ship speed

If speed is constant per run:

```text
ship_speed_kn[t] = run.speed
```

Do not cubic-interpolate speed.

---

# 10. Split Before Windowing

This is the most important methodological change.

## 10.1 Preferred split hierarchy

Use the strongest feasible protocol supported by the actual data.

### Protocol A — Run-held-out

Use when there are enough independent simulation runs.

```text
train runs
validation runs
test runs
```

No run appears in more than one split.

This is the preferred primary benchmark.

### Protocol B — Purged temporal

Use when the number of independent runs is too small.

For each run:

```text
early block  → train
middle block → validation
late block   → test
```

Add an embargo/purge between blocks.

Default:

```text
purge_steps = INPUT_WINDOW + OUTPUT_WINDOW = 150 timesteps
             = 15 seconds at 10 Hz
```

The exact purge can be configurable.

### Protocol C — Leave-one-speed-out

Use as an explicit out-of-distribution experiment.

For each of:

```text
2 kn
5 kn
8 kn
```

hold one speed entirely out for testing.

Train on the remaining speeds.

Use a training-only temporal or run-level validation subset for early stopping.

This tests generalization across operating conditions and should not replace the in-domain benchmark.

---

# 11. Strict Interpolation/Split Order

To prevent interpolation itself from peeking across a split boundary, the rigorous pipeline should use:

```text
raw run
↓
determine split time boundaries
↓
apply purge/embargo
↓
interpolate each train/val/test block independently
↓
fit scaler on train block only
↓
transform val/test using train scaler
↓
window each block independently
```

Do not:

```text
interpolate full timeline
→ normalize full timeline
→ create all windows
→ split windows
```

If independent cubic interpolation produces unstable boundary behavior, use linear/PCHIP and document the reason.

---

# 12. Window Generation

Canonical configuration:

```yaml
sampling_rate_hz: 10
input_steps: 100
output_steps: 50
stride_steps: 10
```

Primary benchmark uses **one common stride** across all models.

Recommended:

```text
stride = 10 = 1 second
```

because stride 1 produces highly redundant samples and excessive pseudo-replication.

Optional ablation:

```text
stride = 1
stride = 5
stride = 10
stride = 25
```

but never compare models trained with different strides as if architecture alone caused the difference.

Each window must retain metadata:

```text
window_id
run_id
split
input_start_time
input_end_time
target_start_time
target_end_time
raw/interpolated index coverage
speed
```

Save:

```text
data/manifests/windows.parquet
```

---

# 13. Leakage Assertions

Implement explicit assertions.

## 13.1 Run leakage

```python
assert train_run_ids.isdisjoint(test_run_ids)
```

for run-held-out protocols.

## 13.2 Window raw-time overlap

For any pair of splits:

```text
coverage(train) ∩ coverage(validation) = ∅
coverage(train) ∩ coverage(test) = ∅
coverage(validation) ∩ coverage(test) = ∅
```

This checks both input and output ranges.

## 13.3 Scaler fitting

Store the list/hash of training run IDs and time ranges used to fit each scaler.

## 13.4 Hyperparameter isolation

Test labels must never enter model selection.

---

# 14. Feature Sets

Create explicit named feature-set experiments.

## F0 — Motion only

```text
surge
sway
heave
roll
pitch
yaw
```

## F1 — Wave + motion

```text
wave
+ 6-DoF
```

## F2 — Wave + motion + speed

```text
wave
ship_speed
+ 6-DoF
```

### Primary comparison

Use F2 if the dataset supports ship speed correctly.

Otherwise use F1.

Every primary model must receive the identical feature set.

### Why feature ablation matters

This answers:

```text
Does explicit wave excitation help?
Does explicit vessel speed help?
How much can the model infer from motion autoregression alone?
```

This is much more defensible than silently changing inputs between model families.

---

# 15. Target Scaling

Six motion channels have very different magnitudes and units.

Do not optimize raw MSE across mixed meters/degrees because high-variance channels can dominate.

Fit training-only target normalization:

```text
z_d = (y_d - μ_train,d) / σ_train,d
```

Train neural models on standardized targets.

Classical tree models may operate on physical targets, but evaluation must still be identical.

For fairness, the evaluator always receives denormalized physical predictions.

---

# 16. Mandatory Baselines

## B0 — Persistence / last-state baseline

For each DoF:

```text
ŷ(t+h) = y(t_last)
```

for all `h = 1...50`.

This is mandatory.

## B1 — Linear-trend extrapolation

Fit a line over the last configurable portion of the input window per DoF and extrapolate 50 future steps.

Example:

```text
trend window = last 20 timesteps
```

This is a strong cheap baseline for smooth motion.

## B2 — Ridge multi-output regression

Flatten:

```text
100 × F → 100F
```

predict:

```text
50 × 6 → 300 outputs
```

This provides a simple linear high-dimensional baseline.

---

# 17. Random Forest Baseline

Input:

```text
[N, 100, F] → flatten → [N, 100F]
```

Target:

```text
[N, 50, 6] → flatten → [N, 300]
```

Starting configuration based on the original project:

```yaml
n_estimators: 200
max_depth: 15
min_samples_split: 5
min_samples_leaf: 2
bootstrap: true
n_jobs: -1
random_state: 42
```

Do not tune on test.

A small validation-only search is allowed.

Record:

```text
fit time
artifact size
CPU inference latency
```

---

# 18. XGBoost Baseline

Use the same flattened input and 300-output target.

Starting configuration:

```yaml
n_estimators: 200
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
objective: reg:squarederror
random_state: 42
```

Codex must inspect the installed XGBoost version and use a supported multi-output strategy.

Do not silently rely on version-specific native behavior.

Possible implementation choices:

```text
native multi-output if verified
or
documented wrapper strategy
```

Persist exact XGBoost version.

---

# 19. LSTM Baseline

Implement a clean PyTorch LSTM; do not reuse the contaminated Keras result.

Recommended starting architecture:

```text
input projection
2 stacked LSTM layers
hidden size = 128
dropout = 0.2
final hidden representation
MLP
reshape → [B, 50, 6]
```

Training:

```yaml
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-5
batch_size: 32
max_epochs: 100
early_stopping_patience: 10
gradient_clip: 1.0
```

Validation loss only determines early stopping.

Save best validation checkpoint.

---

# 20. Transformer Baseline

Reimplement the existing Phase-1 architecture under the same data protocol.

Starting architecture:

```text
input features → Linear(d_model=128)
→ positional encoding
→ 3 Transformer encoder layers
   8 heads
   FFN=512
   GELU
   dropout=0.1
→ temporal pooling
→ MLP
→ 50 × 6 trajectory
```

Training:

```yaml
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-5
batch_size: 32
max_epochs: 100
early_stopping_patience: 10
gradient_clip: 1.0
scheduler: ReduceLROnPlateau
```

The model must use the same F1/F2 inputs as all other primary models.

---

# 21. Optional Stronger Transformer V2

Only after the legacy-compatible Transformer benchmark works.

The current architecture mean-pools the entire history into one vector and generates all 50 future steps from that single representation.

A stronger direct multi-horizon variant may use learned future queries:

```text
historical encoder tokens
        ↓
50 learned future query tokens
        ↓
cross-attention to history
        ↓
one 6-DoF prediction per future timestep
```

This preserves explicit future-step structure.

Call it:

```text
TransformerV2
```

Do not replace the legacy-compatible Transformer; compare against it.

This is optional and should not delay the rigorous baseline rerun.

---

# 22. Primary Motion Loss

For standardized targets:

```text
L_motion = mean_d,t [(ŷ_z - y_z)^2]
```

Because each DoF is standardized, channels contribute comparably.

Optional robust ablation:

```text
Huber loss
```

Do not tune a custom weighted loss using test-set performance.

---

# 23. Multi-Task Phase-2 Reimplementation

Reproduce the existing structure:

```text
shared temporal Transformer
        ↓
shared future representation
        ├── motion head → [50,6]
        ├── score head  → [50,1]
        └── flag head   → [50,1]
```

Legacy-compatible configuration:

```yaml
d_model: 128
heads: 8
layers: 4
ff_dim: 512
dropout: 0.1

loss_weights:
  motion: 1.0
  score: 0.5
  flag: 0.3
```

Use `BCEWithLogitsLoss`, not sigmoid followed by `BCELoss`, for improved numerical stability.

---

# 24. Propeller Operability Target — Two-Tier Design

## 24.1 Legacy proxy

Reproduce the original heave-derived score for comparability.

Keep its implementation clearly named:

```text
legacy_heave_proxy
```

Do not call it measured ventilation.

## 24.2 Geometry-aware proxy — preferred if source geometry exists

The report states that a propeller reference disc was introduced and propeller geometry was recorded.

Codex must search available data/report assets for:

```text
propeller center coordinates
propeller diameter/radius
static vertical position relative to waterline
```

If sufficient geometry is genuinely available, implement a more physically interpretable vertical-location estimate using ship heave and pitch.

Conceptually:

```text
z_propeller(t) =
heave(t)
+ rigid-body rotational contribution from pitch
(+ roll contribution if lateral offset is nonzero)
```

Then derive propeller-top clearance relative to the instantaneous/static waterline.

Do not invent geometry values.

If exact geometry is missing, do not implement fake geometry.

---

# 25. Fix the Rolling-Proxy Boundary Problem

The existing score uses a centered rolling heave amplitude.

A centered window for a target timestep can depend on points after that timestep and, near the end of a 5-second forecast, potentially on points outside the forecast horizon.

For the rigorous prediction task, use one of:

### Preferred

Instantaneous geometry-based clearance/risk.

### Fallback

A trailing/causal rolling risk statistic:

```text
risk(t) = function of values ≤ t
```

### Legacy-only evaluation

Centered rolling score may be reproduced in a separate legacy experiment, but it must not be the only primary operability target.

---

# 26. Operability Evaluation Modes

## O1 — Realistic threshold

Use the source project's realistic static-depth setting:

```text
H_static = 0.5 m
```

First report:

```text
n_positive
n_negative
positive prevalence
```

If the test split contains no positives:

```text
F1 = N/A
PR-AUC = N/A
ROC-AUC = N/A
```

Do not convert a single-class test set into a positive classification claim.

Still evaluate:

```text
continuous risk RMSE
MAE
correlation
```

## O2 — Synthetic stress test

Use:

```text
H_static = 0.2 m
```

only as a clearly labeled stress-test scenario.

Report:

```text
positive-class precision
positive-class recall
positive-class F1
macro F1
balanced accuracy
PR-AUC
ROC-AUC
confusion matrix
```

The report and CV helper must label this:

```text
synthetic-threshold operability proxy
```

---

# 27. Direct Classification vs Derived Classification

This is a required Phase-2 ablation.

Compare:

## Method A — explicit classifier head

```text
Transformer → flag head
```

## Method B — derive flag from predicted motion

```text
predicted heave/pitch trajectory
→ operability proxy formula
→ predicted event flag
```

If Method B performs similarly to the dedicated flag head, state that the event can be obtained by physically interpretable post-processing.

If the direct head is materially better, quantify the gain.

This determines whether multi-task classification genuinely adds information.

---

# 28. Rare-Event Handling

If positive events exist in the training set:

- compute class frequency on training only,
- use `pos_weight` in `BCEWithLogitsLoss` if needed,
- report the unweighted natural test distribution,
- do not oversample test data.

For highly correlated contiguous positive timesteps, also construct event intervals.

Evaluate event detection separately from timestep classification.

Possible event metric:

```text
event detected if any predicted positive overlaps a true positive interval
```

Report:

```text
event recall
false alarms per minute
mean detection lead/lag if meaningful
```

Only do this if there are enough events.

---

# 29. Common Motion Metrics

For each model, report all of the following on the frozen test split.

## 29.1 Per-DoF physical RMSE

```text
RMSE_surge [m]
RMSE_sway [m]
RMSE_heave [m]
RMSE_roll [deg]
RMSE_pitch [deg]
RMSE_yaw [deg]
```

## 29.2 Per-DoF MAE

Same units.

## 29.3 Per-DoF R²

Do not average raw multi-output R² in an opaque way.

## 29.4 Normalized RMSE

Define:

```text
NRMSE_d =
RMSE_d / σ_train,d
```

Use **training** target standard deviation in the denominator.

Then report:

```text
Macro NRMSE = mean over 6 DoFs
```

This allows fair aggregation across meters and degrees.

## 29.5 Overall standardized RMSE

Evaluate all six standardized channels jointly as a secondary summary.

Do not call raw mixed-unit aggregate RMSE the primary metric.

---

# 30. Skill Relative to Persistence

For each DoF and model compute:

```text
MSE Skill =
1 - MSE_model / MSE_persistence
```

Interpretation:

```text
> 0  better than persistence
= 0  equal to persistence
< 0  worse than persistence
```

Also report macro skill across DoFs.

This is one of the most useful additions for the CV/interview because smooth physical trajectories can make naive forecasting deceptively strong.

---

# 31. Horizon-Dependent Error

A 5-second aggregate score hides degradation with forecast distance.

Evaluate at:

```text
0.5 s
1.0 s
2.0 s
3.0 s
4.0 s
5.0 s
```

or every future timestep.

Generate:

```text
RMSE vs forecast horizon
MAE vs forecast horizon
skill vs persistence vs forecast horizon
```

Required figure:

```text
outputs/figures/horizon_error_curves.png
```

A good model should remain meaningfully superior at longer horizons, not only the first few future samples.

---

# 32. Phase / Timing Accuracy

The original presentation qualitatively claims that Transformer forecasts track phase better.

Quantify this rather than using visual language alone.

For oscillatory DoFs such as:

```text
heave
roll
pitch
```

calculate one or more:

### Cross-correlation lag

Estimate lag maximizing normalized cross-correlation between predicted and true future trajectory.

Report lag in:

```text
timesteps
seconds
```

### Peak timing error

Detect significant local peaks/troughs and compare predicted vs true timing.

### Optional spectral/coherence analysis

Only if enough long contiguous predictions are reconstructed without duplicated overlap.

Do not claim phase superiority unless these metrics support it.

---

# 33. Prediction Reconstruction for Time-Series Plots

Overlapping forecast windows produce multiple predictions for one physical timestep.

Do not create misleading long timeline plots by simply concatenating window outputs.

For reconstructed time-series visualization:

1. map each forecast to absolute `run_id + timestamp`,
2. gather all predictions for each timestamp,
3. aggregate using:
   - mean,
   - median,
   - or forecast with shortest lead time,
4. report which strategy is used.

Keep window-level metric calculation separate.

---

# 34. Statistical Uncertainty

## 34.1 Multiple seeds

For neural models, recommended:

```text
3 seeds minimum
```

if compute allows.

Example:

```text
42
123
2026
```

Report:

```text
mean ± standard deviation
```

for primary metrics.

## 34.2 Confidence intervals

Do not bootstrap individual timesteps as independent samples.

Preferred hierarchy:

```text
run-level bootstrap
```

if enough runs exist.

Otherwise:

```text
block bootstrap
```

over contiguous blocks substantially longer than the motion correlation time.

Report 95% CIs for:

```text
Macro NRMSE
Persistence skill
selected per-DoF RMSEs
difference between best two models
```

---

# 35. Model Comparison Statistics

For two models evaluated on the same windows/runs:

```text
Δmetric =
metric_model_A - metric_model_B
```

Use paired run/block bootstrap.

Report:

```text
mean paired difference
95% CI
```

Do not claim one model is meaningfully superior based only on a tiny decimal difference.

---

# 36. Inference Latency

Benchmark every model under the same conditions.

## 36.1 Batch-1 CPU latency

For deployment relevance:

- batch size 1,
- warmup first,
- at least 500 timed runs,
- single process where possible,
- record hardware.

Report:

```text
p50 latency
p95 latency
mean latency
```

## 36.2 GPU latency

Optional.

Synchronize CUDA before/after timing.

## 36.3 Model complexity

Report:

```text
parameter count
artifact size
fit time
inference latency
```

Do not reuse PPT timing values unless reproduced.

---

# 37. Hyperparameter Selection

Do not perform an enormous search on this small simulated dataset.

Use:

```text
legacy/default configuration
+
small validation-only search
```

Examples:

### RF

```text
max_depth ∈ {10, 15, None}
min_samples_leaf ∈ {1, 2, 5}
```

### XGBoost

```text
max_depth ∈ {4, 6, 8}
learning_rate ∈ {0.03, 0.05, 0.1}
```

### LSTM

```text
hidden ∈ {64, 128}
layers ∈ {1, 2}
```

### Transformer

```text
layers ∈ {2, 3, 4}
d_model ∈ {64, 128}
```

Select by validation Macro NRMSE.

Never use test RMSE to choose architecture.

---

# 38. Primary Experiment Matrix

At minimum run:

| ID | Model | Feature Set | Split | Purpose |
|---|---|---|---|---|
| E0 | Persistence | primary | primary | trivial baseline |
| E1 | Linear Trend | primary | primary | smooth-signal baseline |
| E2 | Ridge | primary | primary | linear multi-output baseline |
| E3 | Random Forest | primary | primary | classical nonlinear |
| E4 | XGBoost | primary | primary | boosted-tree baseline |
| E5 | LSTM | primary | primary | recurrent deep model |
| E6 | Transformer | primary | primary | attention baseline |

`primary` split means:

```text
run-held-out if enough independent runs exist
else purged-temporal
```

All E0–E6 use identical windows.

---

# 39. Feature Ablation Matrix

Using the strongest two or three models:

| ID | Features | Question |
|---|---|---|
| A0 | 6-DoF only | autoregressive information |
| A1 | wave + 6-DoF | value of wave excitation |
| A2 | wave + speed + 6-DoF | value of operating condition |

Only run A2 when speed metadata is valid.

This turns a feature inconsistency in the legacy project into a useful scientific analysis.

---

# 40. Speed-Generalization Experiment

If 2, 5, and 8 knot data are genuinely independent runs:

```text
Fold S2:
train 5+8
test 2

Fold S5:
train 2+8
test 5

Fold S8:
train 2+5
test 8
```

Validation must be carved only from training speeds.

Report:

```text
per-speed Macro NRMSE
per-speed persistence skill
per-DoF RMSE
```

Interpret carefully:

```text
5 kn may be interpolation between observed speeds
2 / 8 kn can represent edge/extrapolation conditions
```

Do not claim broad speed generalization from only three speeds.

---

# 41. Multi-Task Experiment Matrix

| ID | Motion | Score | Flag | Threshold | Purpose |
|---|---|---|---|---|---|
| M0 | yes | no | no | — | single-task Transformer |
| M1 | yes | yes | no | realistic | continuous auxiliary task |
| M2 | yes | yes | yes | 0.5 m | realistic proxy |
| M3 | yes | yes | yes | 0.2 m | synthetic stress test |
| M4 | motion-derived | derived | derived | corresponding | explicit-head comparison |

Questions:

```text
Does multi-task learning improve motion forecasting?
Does the score auxiliary task help?
Does explicit flag prediction outperform deriving the flag from motion?
```

---

# 42. Negative Transfer Analysis

The old report states that the shared encoder does not suffer negative transfer.

This must be tested, not assumed.

Compare:

```text
single-task motion Transformer
vs
multi-task Transformer
```

under the identical split and seed set.

Calculate:

```text
Δ Macro NRMSE
Δ per-DoF RMSE
```

If motion worsens, multi-task training caused negative transfer under that configuration.

Report the result honestly.

---

# 43. Data Expansion — Optional but High Value

If Maxsurf simulation access remains available, the strongest improvement to experimental credibility is more independent runs.

Rather than producing only more overlapping windows, generate independent hydrodynamic conditions.

Vary where possible:

```text
vessel speed
JONSWAP random phase/seed
significant wave height
peak period
wave heading
sea state
```

Each simulation must receive a unique `run_id`.

Do not generate new configurations unless the simulation setup can be reproduced correctly.

This is an optional later milestone because Codex may not be able to drive Maxsurf automatically.

---

# 44. Required Saved Artifacts per Run

Every training run must save:

```text
config.yaml
data_manifest_hash.txt
split_manifest.parquet
split_hash.txt
feature_schema.json
feature_scaler.pkl
target_scaler.pkl
seed.txt
environment.txt
git_commit.txt
checkpoint/model artifact
training_history.json
predictions.parquet or npz
metrics.json
metrics_by_dof.csv
metrics_by_horizon.csv
latency.json
```

Prediction artifact must contain enough metadata to recompute all metrics without retraining.

---

# 45. Experiment ID

Create deterministic IDs from:

```text
dataset version
split hash
feature set
model
model config hash
seed
```

Example:

```text
shipmotion__runholdout_v1__F2__transformer__seed42__8a31c2
```

---

# 46. Canonical Prediction Artifact

Preferred Parquet schema, one row per:

```text
window × future timestep × DoF
```

Columns:

```text
run_id
window_id
split
forecast_origin_time_s
target_time_s
horizon_s
dof
y_true
y_pred
model
seed
experiment_id
```

Alternatively save compressed NumPy arrays plus a metadata Parquet.

Evaluation code must not depend on in-memory training objects.

---

# 47. Common Evaluator

All model comparison tables must be created by:

```bash
python scripts/evaluate.py run_id=<RUN_ID>
```

or the benchmark aggregation command.

No model notebook should contain its own private metric definition.

Evaluator outputs:

```text
overall metrics
per-DoF metrics
per-horizon metrics
persistence skill
phase metrics
confidence intervals
```

---

# 48. Required Figures

Generate:

1. data/run timeline and split visualization,
2. training/validation curves for neural models,
3. per-model Macro NRMSE,
4. per-DoF RMSE comparison,
5. persistence skill by DoF,
6. RMSE vs forecast horizon,
7. representative 6-DoF prediction,
8. worst-case/high-error example,
9. residual distributions,
10. cross-correlation phase lag plot,
11. speed-held-out performance,
12. operability score prediction,
13. operability confusion matrix where both classes exist,
14. model accuracy vs inference latency.

Never cherry-pick only the best-looking sample.

For qualitative examples automatically select:

```text
median-error sample
90th-percentile-error sample
```

and optionally best sample separately.

---

# 49. Data Leakage Report

Implement:

```bash
python scripts/audit_legacy.py
```

and:

```bash
python scripts/make_splits.py ...
```

The new split script should produce:

```text
outputs/reports/leakage_check.md
```

Required fields:

```text
train raw coverage count
validation raw coverage count
test raw coverage count

train ∩ val = 0
train ∩ test = 0
val ∩ test = 0

run overlap
window overlap
scaler fit scope
```

This is an important project artifact.

---

# 50. CLI Contract

## Audit

```bash
python scripts/audit_data.py data.path=<PATH>
```

## Build manifest

```bash
python scripts/build_manifest.py data.path=<PATH>
```

## Purged split

```bash
python scripts/make_splits.py \
  split=purged_temporal \
  data.input_steps=100 \
  data.output_steps=50 \
  split.purge_steps=150
```

## Run-held-out split

```bash
python scripts/make_splits.py split=run_holdout
```

## Train one model

```bash
python scripts/train.py \
  model=transformer \
  experiment=motion_primary \
  seed=42
```

## Full motion benchmark

```bash
python scripts/run_motion_benchmark.py \
  experiment=motion_primary
```

## Feature ablation

```bash
python scripts/run_motion_benchmark.py \
  experiment=feature_ablation
```

## Speed OOD

```bash
python scripts/run_speed_ood.py
```

## Multi-task benchmark

```bash
python scripts/run_multitask_benchmark.py \
  experiment=multitask_realistic
```

## Synthetic threshold stress test

```bash
python scripts/run_multitask_benchmark.py \
  experiment=multitask_stress
```

## Export final report

```bash
python scripts/export_report.py
```

## Export CV-safe claims

```bash
python scripts/export_cv_claims.py
```

---

# 51. Configuration Example

```yaml
data:
  sampling_rate_hz: 10
  input_steps: 100
  output_steps: 50
  stride_steps: 10

  feature_set: wave_motion_speed

  motion_columns:
    - surge_m
    - sway_m
    - heave_m
    - roll_deg
    - pitch_deg
    - yaw_deg

interpolation:
  continuous: pchip
  speed: constant

split:
  strategy: run_holdout
  purge_steps: 150

normalization:
  features: standard
  targets: standard
  fit_scope: train_only

evaluation:
  horizons_seconds: [0.5, 1, 2, 3, 4, 5]
  bootstrap_samples: 1000
  confidence_level: 0.95
```

---

# 52. Transformer Config

```yaml
name: transformer

d_model: 128
n_heads: 8
n_layers: 3
ff_dim: 512
dropout: 0.1
activation: gelu

training:
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.00001
  batch_size: 32
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip: 1.0
```

---

# 53. Multi-Task Config

```yaml
name: multitask_transformer

encoder:
  d_model: 128
  n_heads: 8
  n_layers: 4
  ff_dim: 512
  dropout: 0.1

heads:
  motion: true
  emergence_score: true
  emergence_flag: true

loss:
  motion_weight: 1.0
  score_weight: 0.5
  flag_weight: 0.3

operability:
  score_mode: geometry_if_available_else_causal_heave_proxy
  static_depth_m: 0.5
  stress_test_static_depth_m: 0.2
```

---

# 54. Unit Tests

## Run detection

- timestamps restarting at zero produce separate runs,
- speed changes produce separate run identity only when metadata implies independent runs,
- duplicate timestamps in different runs are preserved.

## Interpolation

- output frequency is 10 Hz,
- no interpolation crosses run boundary,
- constant vessel speed remains constant,
- interpolation does not create NaNs.

## Splits

- deterministic for fixed seed/config,
- correct run assignment,
- purge respected.

## Window leakage

Construct a toy sequence and verify that raw timestep coverage from train/val/test windows is disjoint.

This test is mandatory.

## Scaling

- scaler is fit only on training data,
- an extreme test-only value does not influence `scaler.mean_` or `scale_`.

## Model shapes

All models output:

```text
[B, 50, 6]
```

Motion multi-task head also outputs this exact shape.

## Metrics

Validate RMSE/MAE/R²/skill on hand-computed arrays.

## Operability

- realistic single-class case returns classification metrics as N/A rather than fake 1.0,
- stress threshold produces correct flags,
- derived-from-motion flag matches formula.

---

# 55. Integration Smoke Test

Create synthetic toy runs.

Execute:

```text
load
→ detect runs
→ split raw timelines
→ interpolate blocks
→ fit training scaler
→ window
→ train Ridge or tiny neural model
→ predict
→ denormalize
→ evaluate
→ save artifacts
```

This must run in CI without real Maxsurf data.

---

# 56. Tiny-Set Neural Overfit Test

Before full LSTM/Transformer runs:

- select a few training windows,
- train until near-zero training loss,
- verify architecture can overfit.

If it cannot, debug model/data pipeline before a long run.

---

# 57. Reproducibility

Set and log:

```text
Python random seed
NumPy seed
PyTorch seed
CUDA deterministic settings where practical
```

Keep in mind exact GPU reproducibility may not always be guaranteed.

Log CUDA/cuDNN versions.

---

# 58. Report Structure

Generate:

```text
outputs/reports/FINAL_BENCHMARK_REPORT.md
```

Sections:

1. Project problem and simulation source
2. Legacy implementation audit
3. Data/run structure
4. Leakage-safe benchmark design
5. Model architectures
6. Baselines
7. Primary results
8. Per-DoF performance
9. Horizon-dependent performance
10. Persistence skill
11. Phase/timing analysis
12. Speed generalization
13. Multi-task learning
14. Propeller-operability proxy analysis
15. Statistical uncertainty
16. Accuracy/latency trade-off
17. Limitations
18. CV-safe conclusions

---

# 59. Final Comparison Table

The report should contain a table similar to:

| Model | Macro NRMSE ↓ | Persistence Skill ↑ | Heave RMSE | Pitch RMSE | p50 CPU Latency | Params |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | ... | 0.000 | ... | ... | ... | 0 |
| Linear Trend | ... | ... | ... | ... | ... | 0 |
| Ridge | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... | ... |
| LSTM | ... | ... | ... | ... | ... | ... |
| Transformer | ... | ... | ... | ... | ... | ... |

Do not populate with placeholder fake numbers.

---

# 60. CV Claim Generator

`export_cv_claims.py` must read only final saved benchmark metrics.

It should output:

```text
outputs/cv/CV_CLAIMS.md
```

with:

## Project title

Recommended:

**Multi-Task Temporal Transformer for Ship Motion & Propeller Operability — B.Tech Project**

## Safe architecture bullet

Always allowed after implementation:

> Developed multi-task Temporal Transformer forecasting 5-second 6-DoF vessel trajectories while estimating propeller-operability risk from wave-motion telemetry.

## Comparison bullet

Only if actual benchmark supports it:

> Benchmarked Transformer, LSTM, XGBoost, Random Forest and persistence baselines under leakage-free temporal/run-held-out evaluation, reducing Macro NRMSE by XX%.

The code must determine:

```text
best non-Transformer baseline
relative improvement
confidence interval if available
```

Do not automatically state "43%" because it appeared in the old presentation.

## Data/preprocessing bullet

Possible:

> Built leakage-free 10 Hz forecasting pipeline across Maxsurf/MOSES irregular-wave simulations with run-aware interpolation, train-only scaling and multi-horizon evaluation.

## Multi-task result bullet

Only if defensible:

> Extended shared temporal encoder to joint motion and operability-risk prediction, achieving XX while preserving/improving motion-forecast skill.

The generated file should include both:

```text
SAFE NOW
SAFE ONLY IF METRIC CONDITION PASSES
DO NOT CLAIM
```

---

# 61. Claim Validation Rules

Implement simple guardrails.

### Improvement claim

Only generate if:

```text
best Transformer Macro NRMSE
<
best non-Transformer Macro NRMSE
```

and ideally paired CI favors Transformer.

### Real-time claim

Only generate if measured p95 batch-1 latency is below the declared control-cycle budget on named hardware.

### Operability F1 claim

Only generate when:

```text
test positives > 0
test negatives > 0
```

and the output explicitly identifies the target as a proxy/stress-test where applicable.

### Cross-speed claim

Only generate if leave-one-speed-out evaluation was actually run.

---

# 62. Claims to Remove from Current Materials Unless Revalidated

Do not propagate these automatically:

```text
"Transformer RMSE = 0.0095"
"43% reduction over XGBoost"
"Transformer creates a new benchmark"
"99% F1 detecting propeller ventilation"
"attention inherently learns the RAO"
"operationally safe for real-world integration"
"near-perfectly emulates wave-hull interactions"
```

Some may become defensible after the new benchmark, but they must be earned by new evidence.

---

# 63. Better Interview Framing

The strongest technical story after reimplementation should be:

> We initially built several forecasting models, but overlapping time windows and inconsistent preprocessing made cross-model comparison overly optimistic. I rebuilt the evaluation at the raw simulation-run level, fit preprocessing only on training data, added persistence and linear baselines, evaluated all models in physical units across the same 5-second horizon, and separately tested out-of-distribution generalization across vessel speeds. I then extended the best temporal architecture to multi-task operability-risk prediction and tested whether that auxiliary objective improved or hurt motion forecasting.

This is a significantly stronger Data Science story than simply saying:

> "Transformer had the lowest RMSE."

---

# 64. Milestone Plan

## M0 — Preserve and audit legacy project

Deliver:

- legacy artifacts copied/linked without modification,
- `LEGACY_AUDIT.md`,
- source-file inventory,
- list of old metrics marked historical.

Done when:

```text
no legacy source was overwritten
audit identifies each known leakage/comparison issue
```

## M1 — Canonical data/run pipeline

Deliver:

- run detection,
- canonical schema,
- audit report,
- independent interpolation,
- manifests.

Done when:

```text
every row belongs to exactly one run
no interpolation crosses runs
```

## M2 — Leakage-free splits/windowing

Deliver:

- run-held-out and purged-temporal strategies,
- training-only scalers,
- window manifest,
- leakage tests.

Done when:

```text
all cross-split raw coverage intersections are zero
```

## M3 — Simple baselines

Deliver:

- persistence,
- linear trend,
- Ridge,
- common evaluator.

Done when:

```text
physical-unit metrics and horizon curves are generated
```

## M4 — Classical ML

Deliver:

- RF,
- XGBoost,
- fit/latency artifacts.

## M5 — LSTM

Deliver:

- clean PyTorch LSTM,
- validation-only early stopping,
- common evaluation.

## M6 — Transformer

Deliver:

- legacy-compatible Transformer,
- same exact split/features/evaluator.

## M7 — Primary benchmark

Deliver:

- all-model table,
- per-DoF metrics,
- persistence skill,
- horizon metrics,
- CIs where feasible.

This milestone is the minimum required before updating the CV with performance numbers.

## M8 — Feature ablations

Deliver:

- motion only,
- +wave,
- +speed where available.

## M9 — Speed OOD

Deliver:

- leave-one-speed-out evaluation.

## M10 — Phase-2 multi-task

Deliver:

- continuous proxy,
- realistic threshold,
- synthetic stress-test threshold,
- explicit head vs motion-derived event ablation.

## M11 — Phase/timing and latency analysis

Deliver:

- cross-correlation timing metric,
- p50/p95 latency,
- accuracy-latency plot.

## M12 — Final report and CV export

Deliver:

- `FINAL_BENCHMARK_REPORT.md`,
- figures,
- `CV_CLAIMS.md`,
- reproducible commands.

---

# 65. Minimum CV-Worthy Definition of Done

The project is ready for placements when:

- [ ] raw simulation runs are explicitly identified,
- [ ] no window crosses a run or split boundary,
- [ ] scalers are train-only,
- [ ] persistence baseline exists,
- [ ] RF/XGBoost/LSTM/Transformer use the same feature set,
- [ ] all models use the same input/output horizon,
- [ ] all primary metrics are in physical units plus Macro NRMSE,
- [ ] horizon-dependent errors are reported,
- [ ] final test data were not used for tuning,
- [ ] at least one robust generalization protocol is used,
- [ ] Phase-2 realistic vs synthetic threshold is clearly separated,
- [ ] results are saved to reproducible artifacts,
- [ ] CV claims are generated from new metrics only.

---

# 66. Stronger Research-Level Definition of Done

For an even stronger project:

- [ ] several independent simulation runs per condition,
- [ ] run-held-out test rather than only temporal holdout,
- [ ] leave-one-speed-out analysis,
- [ ] 3 neural seeds,
- [ ] paired block/run bootstrap CIs,
- [ ] phase-lag quantification,
- [ ] geometry-aware propeller proxy if true geometry exists,
- [ ] explicit negative-transfer test for multi-task learning,
- [ ] optional TransformerV2 future-query architecture.

---

# 67. README Positioning

Do not title the repository:

```text
Ship Motion Prediction Using ML
```

Preferred:

# Multi-Task Temporal Forecasting for 6-DoF Ship Motion and Propeller Operability

Subtitle:

> Leakage-free benchmarking of classical ML, recurrent networks and Transformers for 5-second vessel-motion forecasting under irregular-wave simulations.

README opening should explain:

```text
Problem
Dataset/simulation
Forecasting task
Why temporal leakage matters
Models
Evaluation protocols
Primary results
Phase-2 operability extension
Limitations
Reproduce
```

---

# 68. Limitations to State Explicitly

1. Training data are simulation-derived rather than measured full-scale vessel telemetry.
2. Hydrodynamic solver assumptions constrain what the learned surrogate can reproduce.
3. Available speeds/sea states may cover a narrow operating envelope.
4. Overlapping windows create correlated examples even when split leakage is removed.
5. The realistic propeller threshold may contain no positive events.
6. The legacy emergence score is a proxy based on hull motion, not direct propeller ventilation simulation.
7. Generalization to a different hull geometry is not established.
8. Real-time latency depends on hardware and deployment stack.
9. Attention weights alone do not prove physical causal interpretation.
10. More independent simulation runs are more valuable than simply increasing overlapping-window count.

---

# 69. First Prompt to Give Codex

After giving Codex this specification, use:

```text
Read SHIP_MOTION_REIMPLEMENTATION_SPEC.md completely before changing code.

Do not start by retraining the Transformer.

First implement only Milestones M0–M2.

1. Inspect every legacy notebook and source artifact listed in the specification.
2. Create docs/LEGACY_AUDIT.md and verify the identified issues against the real code.
3. Inspect the actual ship-motion datasets available in the repository. Determine true simulation-run boundaries, vessel-speed metadata, time resets, and whether multiple runs share timestamps.
4. Build the canonical run manifest and data audit.
5. Implement run-aware interpolation without joining separate simulations.
6. Implement raw-timeline train/validation/test splitting BEFORE scaling and window generation.
7. Fit preprocessing statistics on train only.
8. Create windows independently inside each split.
9. Persist window/run manifests.
10. Add hard tests proving zero cross-split raw-timestep coverage and no run-boundary crossing.

Do not implement RF, XGBoost, LSTM, Transformer, Phase-2 operability, or CV claims until these tests pass.

Do not copy any legacy result number into new output files.

At completion report:
- repository structure created,
- datasets and runs discovered,
- verified input columns,
- exact run-boundary logic,
- split sizes before and after purge,
- scaler fitting scope,
- test results,
- any assumptions that could not be verified from the available files.
```

---

# 70. Second Prompt to Give Codex After M0–M2 Pass

```text
Continue with Milestones M3–M7.

Use the exact immutable split and window manifests produced previously.

Implement:
1. persistence,
2. linear-trend baseline,
3. Ridge,
4. Random Forest,
5. XGBoost,
6. clean PyTorch LSTM,
7. legacy-compatible Transformer.

All models must use the same primary feature set, 100-step history, 50-step target, stride, train/validation/test manifests, and common evaluator.

Hyperparameters may be selected using validation data only.

Evaluate only through the central evaluator and save:
- per-DoF RMSE/MAE/R²,
- Macro NRMSE normalized by training-set target std,
- persistence skill,
- horizon-dependent error,
- predictions,
- fit time,
- CPU batch-1 p50/p95 latency.

Do not update README conclusions until the full benchmark is complete.

At the end, generate the primary comparison table and identify whether the Transformer actually outperforms the strongest non-Transformer baseline under the leakage-free protocol.
```

---

# 71. Third Prompt for Phase 2

```text
Continue with Milestones M8–M12 only after the primary motion benchmark is frozen.

Implement:
1. feature ablations,
2. leave-one-speed-out evaluation where data supports it,
3. legacy-compatible Multi-Task Transformer,
4. realistic 0.5 m operability-proxy experiment,
5. separately labeled synthetic 0.2 m stress-test experiment,
6. dedicated flag head vs flag derived from predicted motion,
7. single-task vs multi-task negative-transfer comparison,
8. phase/timing analysis,
9. final report,
10. CV claim exporter.

Inspect available source files for genuine propeller geometry before implementing any geometry-aware proxy. Never invent coordinates or dimensions.

If the realistic test split has only one event class, report classification metrics as N/A and explain why.

Generate CV claims only from the new frozen-test artifacts.
```

---

# 72. Final Desired Technical Story

The rebuilt project should ultimately support a statement like:

> **Multi-Task Temporal Transformer for Ship Motion & Propeller Operability** — Built a leakage-free multi-horizon forecasting benchmark on Maxsurf/MOSES irregular-wave simulations, comparing persistence, classical ML, LSTM and Transformer models for 5-second 6-DoF prediction; evaluated horizon-dependent error, cross-speed generalization, inference latency, and multi-task operability-risk forecasting.

Then add actual quantified bullets only when the new benchmark supports them.

The project's strongest signal should become:

```text
domain simulation
+ careful time-series experimental design
+ classical/deep model benchmarking
+ multi-horizon forecasting
+ OOD/generalization analysis
+ multi-task learning
+ honest proxy-label treatment
+ reproducible statistical evaluation
```

That is substantially stronger for a Data Science / ML CV than merely quoting the lowest RMSE from the original notebooks.
