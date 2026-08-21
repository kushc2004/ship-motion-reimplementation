# Multi-Horizon 6-DoF Ship-Motion Forecasting

## Detailed technical implementation, experimental protocol, and verified results

## 1. Project objective

This project develops and evaluates a leakage-free machine-learning system for
forecasting vessel motion in six degrees of freedom (6-DoF): surge, sway,
heave, roll, pitch, and yaw. The system observes the recent wave elevation and
ship-motion history, then predicts the complete future trajectory of all six
motion channels over a fixed multi-step horizon.

The central research question is not simply whether a neural network can fit a
ship-motion time series. It is whether different forecasting models can be
compared fairly when the data contains multiple independent simulation runs,
repeated timestamps, different vessel speeds, and strongly overlapping
sliding windows.

The implementation therefore treats experimental validity as part of the
modeling problem. It establishes simulation-run identity before interpolation,
splits the raw timeline before scaling or window generation, fits all
preprocessing statistics using training data only, and verifies with automated
tests that no raw timestep is shared across train, validation, and test
partitions.

The completed primary study compares seven forecasting approaches under one
common protocol:

1. persistence;
2. local linear-trend extrapolation;
3. Ridge regression;
4. Random Forest regression;
5. XGBoost regression;
6. a sequence-to-vector PyTorch LSTM;
7. a legacy-compatible Transformer encoder.

The project also implements feature ablations, leave-one-speed-out testing,
multi-task motion-and-operability comparisons, phase/timing analysis, and a
frozen-artifact report/CV-claim exporter. All M0–M12 stages are complete under
the same immutable split and window protocol.

## 2. Forecasting problem formulation

At each forecast origin, the model receives a 100-timestep history sampled at
10 Hz. This represents 10 seconds of observed behavior. Each timestep contains
the primary feature vector

\[
x_t = [w_t, s_t, y_t, h_t, r_t, p_t, \psi_t],
\]

where:

- \(w_t\) is wave elevation in metres;
- \(s_t\) is surge in metres;
- \(y_t\) is sway in metres;
- \(h_t\) is heave in metres;
- \(r_t\) is roll in degrees;
- \(p_t\) is pitch in degrees;
- \(\psi_t\) is yaw in degrees.

The input tensor for one example has shape `100 × 7`. The required output is a
50-timestep trajectory for the six motion channels:

\[
\hat{Y}_{t+1:t+50} \in \mathbb{R}^{50 \times 6}.
\]

At 10 Hz, the model therefore predicts the next 5 seconds of vessel motion.
Windows advance with a stride of 10 samples, equivalent to a new forecast
origin every second.

This is direct multi-horizon forecasting: the learned models produce all 50
future steps in one forward pass. They do not recursively feed predictions
back into the input. Direct prediction avoids recursive error accumulation and
makes the inference time of every learned model independent of the forecast
horizon length.

Vessel speed is deliberately excluded from the primary seven-feature protocol
because the BTP-2 source contains no verifiable speed field. Speed is evaluated
separately in the BTP-1 feature ablation, where it is genuinely available.

## 3. Source-data investigation

### 3.1 Why source inspection was necessary

The repository contains CSV files with different purposes and layouts. Some
have column names below introductory rows, some contain repeated embedded
headers, and some contain ship parameters rather than time-series samples.
Selecting a file merely because it contains marine terminology would mix
incompatible data into the forecasting pipeline.

Every relevant notebook and source artifact was inspected to determine:

- which files contain real sequential motion observations;
- where their actual headers begin;
- whether timestamps reset;
- whether equal timestamps belong to different simulations;
- whether vessel speed is a measured column, run-level metadata, or absent;
- whether a source is a time series, an RAO table, or a parameter table;
- how the legacy notebooks performed interpolation, scaling, splitting, and
  window generation.

### 3.2 Eligible motion datasets

Two files contain eligible motion telemetry:

| Dataset | Eligible source | Raw rows | Independent runs | Verified speed metadata |
|---|---|---:|---:|---|
| BTP-1 | `BTP-1/BTP - Time-Series Data.csv` | 16,407 | 3 | 2, 5, and 8 kn |
| BTP-2 | `BTP-2/BTP_2_data - Sheet1.csv` | 6,733 | 1 | unavailable |

The verified canonical columns are:

| Canonical field | Meaning | Unit |
|---|---|---|
| `time_s` | simulation time | seconds |
| `wave_m` | incident/observed wave elevation | metres |
| `ship_speed_kn` | run-level vessel speed, where present | knots |
| `surge_m` | longitudinal translation | metres |
| `sway_m` | lateral translation | metres |
| `heave_m` | vertical translation | metres |
| `roll_deg` | rotation about longitudinal axis | degrees |
| `pitch_deg` | rotation about transverse axis | degrees |
| `yaw_deg` | rotation about vertical axis | degrees |

### 3.3 Files deliberately excluded from model training

`BTP-1/Data.csv` is not a telemetry sequence. It contains ten repeated
response-amplitude-operator blocks: two speed conditions, 2.5 and 8 kn, at
headings 0°, 45°, 90°, 135°, and 180°. The internal headings recur at rows 1,
94, and subsequent block boundaries. The file has no simulation-time column,
so it cannot provide chronological forecast windows.

`BTP-1/Ship-Parameters-data.csv` contains side-by-side parameter and
hydrostatic input tables. It describes the simulated ship but does not contain
a sampled motion trajectory. It was not treated as training data.

This distinction is important: RAO curves and vessel parameters may be useful
for a future physics-informed model, but they cannot be silently concatenated
with time-domain telemetry.

### 3.4 Discovered simulation runs

The complete run inventory is:

| Run | Source-row range | Raw rows | Time range | Speed | Reset at run start |
|---|---:|---:|---:|---:|---|
| BTP-1 run 1 | 0–1,271 | 1,272 | 0–232.864 s | 2 kn | no |
| BTP-1 run 2 | 1,272–2,895 | 1,624 | 0–268.684 s | 5 kn | yes |
| BTP-1 run 3 | 2,896–16,406 | 13,511 | 0–1,811.994 s | 8 kn | yes |
| BTP-2 run 1 | 0–6,732 | 6,733 | 0–539.513 s | unavailable | no |

BTP-1 contains multiple simulations whose local clocks all start at zero.
Consequently, equal timestamp values can occur in different simulations. This
is valid and expected. Within a run, however, `(run_id, time_s)` is unique.

The exact boundary rule is:

\[
\text{new run at row } i \iff time_i - time_{i-1} \leq 0.
\]

A vessel-speed change is retained as metadata but is not independently used to
invent a boundary. In the available BTP-1 data, the timestamp resets and speed
conditions align.

## 4. Legacy implementation audit

The legacy notebooks were valuable for recovering the intended variables,
forecast horizon, and model family, but their evaluation logic could not be
used unchanged. The code audit verified the following methodological risks.

### 4.1 Windowing before splitting

Several workflows created overlapping input/target windows from the complete
series and then split the windows. Because consecutive windows share most of
their raw samples, a timestamp can appear in training input, validation target,
or test input simultaneously. Randomly separating window IDs does not remove
this overlap.

### 4.2 Scaling on the complete dataset

Legacy workflows fitted `StandardScaler` or `MinMaxScaler` before creating the
data split. Even without target labels, this leaks validation/test distribution
statistics into model training and makes strict held-out evaluation invalid.

### 4.3 Joining independent simulations

Some Transformer preparation code added time offsets to separate runs,
concatenated them, removed run identity, and interpolated the combined table.
This creates synthetic continuity between the end of one simulation and the
beginning of another. A sliding window near that seam can contain samples from
two physically independent trajectories.

### 4.4 Inconsistent model inputs

Different legacy model notebooks used different combinations of wave, vessel
speed, and motion variables. Metrics from models with different information
sets do not establish an architecture-level comparison.

### 4.5 Non-experimental result values

Some visualizations contained hard-coded or illustrative values. No metric in
the new benchmark was copied from a notebook, report, presentation, or manually
entered result table. Every reported value in this document is derived from
the new prediction artifacts.

## 5. Leakage-free data pipeline

### 5.1 Required operation order

The data operations are intentionally ordered as follows:

1. parse and canonicalize an eligible source file;
2. retain source file and original source-row identity;
3. detect independent simulation runs from local timestamp resets;
4. assign the raw rows to train, validation, test, or purge;
5. interpolate each `(run_id, split)` block independently to 10 Hz;
6. estimate feature and target statistics from training rows only;
7. create input/target windows independently inside each run and split;
8. verify the raw-row coverage of every generated window.

The crucial design choice is that splitting precedes interpolation, scaling,
and windowing. Once a raw row is assigned to validation or test, it cannot
affect a learned preprocessing statistic or a training window.

### 5.2 Run-aware interpolation

The raw samples are resampled to a uniform 0.1-second grid. Continuous motion
and wave channels use linear interpolation. Vessel speed is constant run-level
metadata and is held constant rather than numerically blended.

Interpolation is performed separately for every `(run_id, split)` block. It
never fills values:

- across a timestamp reset;
- from the end of one simulation into another simulation;
- across a train/validation boundary;
- across a validation/test boundary;
- through a purged boundary region.

This prevents interpolation itself from becoming a hidden leakage path.

### 5.3 Split design

#### BTP-1: run-held-out and speed-shift evaluation

The three complete simulations define the partitions:

| Partition | Run | Speed | Raw rows | Windows |
|---|---|---:|---:|---:|
| train | run 1 | 2 kn | 1,272 | 219 |
| validation | run 2 | 5 kn | 1,624 | 254 |
| test | run 3 | 8 kn | 13,511 | 1,798 |

No purge is necessary because each split is a different simulation. This is a
strict out-of-run test and also a cross-speed distribution shift: the model is
trained at 2 kn, selected at 5 kn, and tested at 8 kn.

#### BTP-2: purged chronological evaluation

BTP-2 has one continuous run and no speed metadata, so it cannot use a
run-held-out partition. The raw timeline is assigned chronologically using a
60/20/20 structure. A 150-sample purge, equal to the full 100-step history plus
50-step target span, is placed at each split boundary.

| Partition/state | Raw rows | Windows |
|---|---:|---:|
| train | 4,016 | 309 |
| validation | 1,169 | 78 |
| test | 1,170 | 79 |
| purged | 378 | 0 |

The purge ensures that even the raw temporal support of windows on opposite
sides of a boundary is disjoint.

### 5.4 Train-only scaling

For each feature \(j\), standardization uses

\[
z_{t,j} = \frac{x_{t,j}-\mu_j^{train}}{\sigma_j^{train}}.
\]

The mean and standard deviation are computed only after interpolation and only
from the training partition:

- BTP-1 scaler: 2,330 interpolated rows, all from the 2 kn training run;
- BTP-2 scaler: 3,238 interpolated rows, all from the chronological train
  block.

Target means and target standard deviations for neural training and NRMSE are
also estimated from training data only. A zero standard deviation is replaced
with one solely to prevent division by zero; this fallback is not needed for
the six observed target channels in the reported benchmark.

### 5.5 Window provenance

Each window records its dataset, run ID, split, input start/end time, target
start/end time, and source coverage. Windows are generated only inside one
sorted `(run_id, split)` block.

For a start index \(i\), one example contains:

- input indices `i ... i+99`;
- target indices `i+100 ... i+149`;
- next start index `i+10`.

No padding or cross-run borrowing is used. If a block has fewer than 150
samples remaining, no window is emitted.

### 5.6 Hard integrity tests

The M0–M2 tests passed with **6 passed**. They establish that:

- timestamp resets produce independent run IDs;
- interpolation cannot join two runs;
- interpolation cannot bridge two data splits;
- train, validation, and test windows have zero cross-split raw-timestep
  coverage;
- no window crosses a run boundary;
- scaling statistics are fit only from training rows;
- embedded non-telemetry headers are not accepted as motion samples.

The model/evaluator contract tests passed with **3 passed** and one benign
PyTorch nested-tensor warning. They verify neural output shapes and the required
benchmark artifact schema.

## 6. Model implementations

Every primary model uses exactly the same 100-step history, 50-step target,
stride, F1 feature set, split manifests, and test evaluator.

### 6.1 Persistence baseline

Persistence repeats the final observed 6-DoF state for every future step:

\[
\hat{Y}_{t+h} = Y_t, \qquad h=1,\ldots,50.
\]

It has no fitted parameters. This is the mandatory reference for determining
whether a learned model improves on the assumption that motion remains at its
latest observed value.

### 6.2 Linear-trend baseline

For each DoF independently, ordinary least squares fits a line to the final 20
history samples and extrapolates it for the next 50 samples. This baseline
tests whether short-term velocity/trend information is sufficient without any
learned dataset-level model.

The fit is local to each forecast window and does not use test targets.

### 6.3 Ridge regression

The 100 × 7 history is flattened into 700 predictors. The 50 × 6 target is
flattened into 300 outputs. A multi-output Ridge model with `alpha=1.0` learns
one regularized linear mapping:

\[
\hat{W}=\arg\min_W \|XW-Y\|_2^2 + \alpha\|W\|_2^2.
\]

Ridge provides a strong low-complexity test of whether the trajectory is
approximately a linear function of the recent history.

### 6.4 Random Forest

The Random Forest uses the same 700-dimensional flattened input and predicts
the complete 300-dimensional future trajectory. Its configuration is:

- 200 trees;
- maximum depth 15;
- minimum samples to split 5;
- minimum samples per leaf 2;
- bootstrap sampling enabled;
- fixed random seed 42.

The model captures nonlinear interactions without requiring sequence-specific
architecture assumptions.

### 6.5 XGBoost

XGBoost uses histogram-based gradient-boosted trees with:

- 200 boosting estimators;
- maximum depth 6;
- learning rate 0.05;
- row subsampling 0.8;
- column subsampling 0.8;
- squared-error objective;
- one CPU worker per estimator;
- seed 42.

The 700 input features and 300 target coordinates are identical to Ridge and
Random Forest. The implementation trains one scalar XGBoost regressor per
future coordinate through a serial multi-output wrapper. This preserves the
forecasting task while avoiding an unstable native multi-output runtime on the
development machine.

### 6.6 PyTorch LSTM

The LSTM model is a clean sequence encoder rather than a flattened regressor:

1. a linear layer projects each 7-feature timestep to 128 dimensions;
2. a two-layer LSTM processes the 100-step sequence with hidden width 128;
3. dropout 0.2 is applied between recurrent layers;
4. the final hidden output is passed through a `128 → 256 → 300` decoder;
5. GELU provides the decoder nonlinearity;
6. the 300 outputs are reshaped to `50 × 6`.

This architecture tests whether recurrent temporal state improves direct
multi-horizon forecasting.

### 6.7 Legacy-compatible Transformer

The primary Transformer preserves the intended encoder-style legacy model
family while using clean PyTorch components:

1. linear projection from 7 input features to `d_model=128`;
2. fixed sinusoidal positional encoding for all 100 history positions;
3. three Transformer encoder layers;
4. eight self-attention heads per layer;
5. 512-dimensional feed-forward sublayers;
6. GELU activation, dropout 0.1, and pre-layer normalization;
7. mean pooling over the encoded history;
8. a `128 → 256 → 300` decoder followed by reshape to `50 × 6`.

It predicts the entire future trajectory directly. No causal mask is required
inside the encoder because all 100 input samples are historical and available
at forecast time.

## 7. Training and model-selection protocol

Classical hyperparameters are fixed in the implementation. Neural models are
trained in standardized target space using mean-squared error:

\[
\mathcal{L}_{motion} = \frac{1}{50\cdot6}
\sum_{h=1}^{50}\sum_{d=1}^{6}
(\hat{z}_{h,d}-z_{h,d})^2.
\]

The common neural configuration is:

- AdamW optimizer;
- learning rate `1e-3`;
- weight decay `1e-5`;
- batch size 32;
- maximum 100 epochs;
- gradient norm clipping at 1.0;
- early stopping after 10 epochs without validation improvement;
- checkpoint selected by validation MSE only;
- seed 42.

The Transformer additionally uses `ReduceLROnPlateau` with factor 0.5 and
patience 5. Test results are computed only after model fitting and checkpoint
selection. Test data is never used for early stopping or hyperparameter
selection.

Predictions are converted back to metres/degrees before the central evaluator
computes accuracy metrics.

## 8. Central evaluation methodology

For each model, dataset, DoF, horizon step, and test window, the evaluator
stores forecast origin, target time, true value, predicted value, run ID, split,
and seed. All summary metrics are derived from this common prediction table.

### 8.1 Per-DoF error metrics

For \(N\) target values:

\[
RMSE = \sqrt{\frac{1}{N}\sum_i(\hat y_i-y_i)^2},
\]

\[
MAE = \frac{1}{N}\sum_i|\hat y_i-y_i|,
\]

\[
R^2 = 1-\frac{\sum_i(\hat y_i-y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

Translations are evaluated in metres and rotations in degrees.

### 8.2 Train-normalized NRMSE

To compare channels with different units, each DoF's RMSE is divided by the
standard deviation of that target in the training partition:

\[
NRMSE_d = \frac{RMSE_d}{\sigma^{train}_d}.
\]

Macro NRMSE is the unweighted mean over the six DoFs and, in the primary
comparison table, over the two datasets. Lower is better. The normalization
never uses validation or test statistics.

### 8.3 Persistence skill

Improvement over persistence is measured as:

\[
Skill = 1-\frac{MSE_{model}}{MSE_{persistence}}.
\]

- `1.0` is perfect prediction;
- a positive value improves on persistence;
- `0.0` equals persistence;
- a negative value performs worse than persistence.

### 8.4 Horizon-dependent error

RMSE, MAE, and persistence skill are computed separately for each horizon step
from 0.1 to 5.0 seconds. This identifies whether a model is accurate only near
the forecast origin or remains stable across the full target horizon.

### 8.5 Runtime metrics

The experiment records wall-clock fit time and CPU batch-size-one inference
latency. Latency uses 20 warm-up predictions followed by 500 timed predictions,
reported as p50 and p95 milliseconds. Neural models are explicitly moved to CPU
for this latency comparison.

## 9. Primary benchmark results

### 9.1 Overall comparison

The following values are generated from the frozen leakage-free test
predictions. Macro NRMSE is an unweighted aggregate across datasets and DoFs.

| Model | Macro NRMSE ↓ | Persistence skill ↑ | Fit time (s) | CPU p50 / p95 (ms) | Mean heave RMSE (m) | Mean pitch RMSE (deg) |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 884.652396 | 0.000000 | 0.000017 | 0.003833 / 0.016299 | 1.273763 | 1.743999 |
| Linear Trend | 1772.067466 | -1.583167 | 0.000022 | 0.064521 / 0.198365 | 1.614170 | 2.722341 |
| Ridge | 1286.477101 | -0.502350 | 0.264128 | 0.287917 / 1.389214 | 0.844110 | 1.469053 |
| **Random Forest** | **593.585413** | **0.776001** | 61.257133 | 27.594458 / 48.572404 | **0.207268** | **0.609990** |
| XGBoost | 593.601510 | 0.775952 | 4409.122048 | 93.449177 / 164.137631 | 0.211493 | 0.612180 |
| LSTM | 593.681777 | 0.754660 | 37.288983 | 7.924427 / 27.654078 | 0.412206 | 0.769464 |
| Transformer | 97200.976415 | -19099.730191 | 63.938995 | 4.415864 / 11.456852 | 99.640428 | 244.236470 |

Random Forest is the strongest model in this aggregate and therefore the
strongest non-Transformer baseline. XGBoost is numerically very close but has
much higher fit and inference cost in the implemented multi-output form. The
LSTM also reaches a similar aggregate NRMSE with lower CPU latency than the
tree ensembles.

The Transformer does **not** outperform the strongest non-Transformer model.
Its failure is especially severe on the BTP-1 cross-speed holdout. The project
therefore does not claim Transformer superiority.

### 9.2 Dataset-specific results

The aggregate table hides two very different generalization regimes:

| Dataset | Model | Macro NRMSE | Mean RMSE | Mean MAE | Mean R² | Persistence skill |
|---|---|---:|---:|---:|---:|---:|
| BTP-1 | Persistence | 1768.351787 | 0.982624 | 0.725661 | -1.233583 | 0.000000 |
| BTP-1 | Linear Trend | 3543.152625 | 2.015311 | 1.392272 | -8.176562 | -3.103891 |
| BTP-1 | Ridge | 2572.954009 | 1.528379 | 1.154215 | -5.672407 | -2.004699 |
| BTP-1 | Random Forest | **1187.166628** | 0.653197 | 0.524733 | 0.000001 | **0.552021** |
| BTP-1 | XGBoost | 1187.195884 | 0.653276 | 0.524799 | -0.000135 | 0.551960 |
| BTP-1 | LSTM | 1187.166989 | **0.653193** | **0.524728** | 0.000001 | 0.552020 |
| BTP-1 | Transformer | 194401.721797 | 178.062559 | 117.141999 | -85105.48 | -38200.400321 |
| BTP-2 | Persistence | 0.953006 | 1.057331 | 0.838215 | 0.090844 | 0.000000 |
| BTP-2 | Linear Trend | 0.982307 | 1.088982 | 0.800969 | 0.034078 | -0.062444 |
| BTP-2 | Ridge | **0.000193** | **0.000221** | **0.000031** | 1.000000 | **1.000000** |
| BTP-2 | Random Forest | 0.004198 | 0.004667 | 0.003419 | 0.999982 | 0.999981 |
| BTP-2 | XGBoost | 0.007136 | 0.008037 | 0.005500 | 0.999948 | 0.999943 |
| BTP-2 | LSTM | 0.196566 | 0.216120 | 0.171400 | 0.961159 | 0.957300 |
| BTP-2 | Transformer | 0.231034 | 0.267379 | 0.218163 | 0.945457 | 0.939939 |

#### Interpretation of BTP-1

BTP-1 tests a major operating-condition shift: 2 kn training, 5 kn validation,
and 8 kn testing. The training run has extremely small target variation:

| DoF | Training target standard deviation |
|---|---:|
| surge | 0.000098 m |
| sway | 0.000215 m |
| heave | 0.001069 m |
| roll | 0.001260° |
| pitch | 0.002610° |
| yaw | 0.000368° |

These train-only denominators make normalized errors very large when applied to
the more energetic 8 kn run. Random Forest, XGBoost, and LSTM obtain positive
persistence skill of approximately 0.552, but their mean R² is approximately
zero. They improve over a weak persistence baseline without explaining the
test-run variation well. This is evidence of limited cross-speed
generalization, not deployment-grade prediction.

#### Interpretation of BTP-2

BTP-2 is a chronological holdout within one simulation. Its trajectory is
highly predictable from the immediately preceding history. Ridge reaches Macro
NRMSE 0.000193 and near-perfect R², showing that this particular within-run
mapping is almost linear. Tree ensembles also perform extremely well. The
neural models are accurate but unnecessary for this regime.

The contrast between BTP-1 and BTP-2 is a major research finding: random
within-run or contiguous evaluation can produce excellent metrics that do not
translate to a separate simulation at a new speed.

### 9.3 Error growth over the forecast horizon

The mean RMSE below averages datasets and DoFs at selected future steps:

| Model | 0.1 s | 1.0 s | 2.0 s | 3.0 s | 4.0 s | 5.0 s |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.043921 | 0.432073 | 0.822921 | 1.138454 | 1.356436 | 1.468785 |
| Linear Trend | 0.095142 | 0.411745 | 0.927007 | 1.539260 | 2.167976 | 2.747175 |
| Ridge | 0.341778 | 0.646796 | 0.780698 | 0.742500 | 0.741463 | 1.016688 |
| Random Forest | 0.328979 | 0.329014 | 0.328989 | 0.328902 | 0.328836 | 0.329165 |
| XGBoost | 0.329775 | 0.331116 | 0.330475 | 0.330840 | 0.329909 | 0.330155 |
| LSTM | 0.426163 | 0.429673 | 0.432639 | 0.437450 | 0.437591 | 0.435628 |
| Transformer | 117.750518 | 64.759463 | 67.079268 | 70.513031 | 76.735800 | 82.108290 |

Persistence is strongest at the first forecast step but deteriorates as the
horizon increases. Linear extrapolation eventually diverges. Random Forest,
XGBoost, and LSTM maintain nearly flat aggregate errors because they predict
the complete horizon directly. The flat error should not be interpreted alone
as accurate phase tracking; on BTP-1 it is also consistent with conservative
predictions near a learned mean.

## 10. Feature-ablation study

Feature ablations use a fixed 50-tree Random Forest so the experiment measures
the effect of the input information rather than re-tuning a model for each
feature set.

The tested feature groups are:

- F0: six historical motion channels only;
- F1: wave elevation plus six historical motion channels;
- F2: wave elevation, verified vessel speed, and six motion channels.

F2 is evaluated only for BTP-1 because BTP-2 has no verified speed metadata.

| Dataset | Feature set | Validation Macro NRMSE | Test Macro NRMSE |
|---|---|---:|---:|
| BTP-1 | F0 motion only | 1005.442519 | 1187.169242 |
| BTP-1 | F1 wave + motion | 1005.440948 | 1187.167815 |
| BTP-1 | F2 wave + speed + motion | 1005.439211 | 1187.166257 |
| BTP-2 | F0 motion only | **0.005581** | 0.006453 |
| BTP-2 | F1 wave + motion | 0.008873 | **0.006153** |

Adding wave and speed information produces only negligible change on the
BTP-1 run-held-out task. Because speed is constant inside each complete run,
the training partition contains only the value 2 kn; the model cannot learn a
general speed-response function from one training speed. On BTP-2, adding wave
improves the frozen test value slightly but worsens validation. Test performance
must not be used to retroactively select the feature set.

## 11. Leave-one-speed-out evaluation

The BTP-1 runs enable a limited leave-one-speed-out (LOSO) experiment. Each
speed/run is held out as the complete test simulation. The remaining runs are
split locally into chronological train and validation blocks with a 150-step
purge. Ridge is held fixed across all folds.

| Held-out speed | Train windows | Validation windows | Test windows | Macro NRMSE |
|---:|---:|---:|---:|---:|
| 2 kn | 879 | 244 | 113 | 0.092745 |
| 5 kn | 858 | 241 | 148 | 0.500261 |
| 8 kn | 145 | 3 | 1,337 | 1.326371 |

Generalization becomes progressively harder for the higher-speed held-out run.
The 8 kn fold is especially data-limited because only the short 2 kn and 5 kn
runs remain for training; after preserving temporal separation, just three
validation windows remain.

This experiment supports a speed-shift sensitivity analysis, but it is not a
statistically broad speed-generalization study: the repository contains only
one simulation run per speed.

## 12. Multi-task operability extension

### 12.1 Motivation

The later project phase extends motion forecasting toward an operational
decision question: can the shared temporal representation predict both future
motion and a threshold-based heave operability signal?

The available source files do not provide genuine numerical propeller
coordinates, propeller diameter, stern geometry, draft-relative position, or a
validated emergence equation. Therefore, no physical propeller-emergence label
is invented. The implementation uses an explicitly labelled heave-based proxy.

### 12.2 Causal operability proxy

For each forecast timestep, a trailing five-sample heave half-range is
calculated:

\[
A_t = \frac{\max(H_{t-4:t})-\min(H_{t-4:t})}{2}.
\]

The normalized operability score for threshold/depth \(D\) is

\[
q_t = \frac{A_t}{D},
\]

and the event flag is

\[
f_t = \mathbb{1}[q_t \geq 1].
\]

The first four target steps carry the final four historical heave samples so
the trailing statistic remains causal at the forecast boundary. No centered
window or future target sample is used to label an earlier timestep.

Two thresholds are defined:

- 0.5 m: realistic operability-proxy experiment;
- 0.2 m: separately labelled synthetic stress-test experiment.

Neither threshold is described as measured propeller emergence.

### 12.3 Multi-task Transformer architecture

The multi-task network uses:

- 7-feature, 100-step input;
- linear projection to `d_model=128`;
- sinusoidal positional encoding;
- four Transformer encoder layers;
- eight attention heads;
- feed-forward width 512;
- GELU and dropout 0.1;
- mean-pooled shared sequence representation.

Three output heads can be enabled independently:

1. motion head: `128 → 256 → 300`, reshaped to `50 × 6`;
2. continuous score head: `128 → 50`;
3. direct event-logit head: `128 → 50`.

The joint objective is

\[
\mathcal{L} = \mathcal{L}_{motion}
+0.5\mathcal{L}_{score}
+0.3\mathcal{L}_{flag},
\]

where score loss is MSE and flag loss is class-weighted binary cross-entropy.
Positive-class weight is computed from training labels. Validation loss selects
the checkpoint.

This design supports two separate event-generation paths:

- dedicated flag head: classify the event directly from the shared encoder;
- motion-derived flag: forecast heave first, calculate the causal score from
  predicted motion, then threshold it.

Comparing these paths tests whether direct classification adds information or
merely duplicates the motion forecast. Comparing the single-task and
multi-task motion errors measures negative transfer from the auxiliary tasks.

### 12.4 Frozen operability and negative-transfer results

The four M10 configurations were trained and evaluated on the frozen test
windows with the same four-layer Transformer design: single-task motion at the
0.5 m proxy threshold, score-only auxiliary learning at 0.5 m, direct
multi-task score/flag learning at 0.5 m, and a separately labelled 0.2 m
synthetic stress test. The final run used Kaggle Tesla T4 CUDA hardware; CPU
batch-one latency was measured separately.

| Experiment | Meaning | Motion Macro NRMSE | Fit time (s) | CPU p50/p95 (ms) |
|---|---|---:|---:|---:|
| single-task 0.5 m | motion head only | 0.228648 | 2.830122 | 3.091499 / 3.416445 |
| score-only 0.5 m | motion plus continuous score | 0.292179 | 2.103644 | 3.216238 / 3.600661 |
| multi-task 0.5 m | motion, score, and direct flag head | 0.294067 | 2.130804 | 3.307235 / 3.694236 |
| multi-task 0.2 m | synthetic stress-test threshold | 0.303660 | 2.125572 | 3.470285 / 5.764311 |

At 0.5 m, multi-task minus single-task motion Macro NRMSE is **+0.065420**,
so the auxiliary tasks produced negative transfer in this frozen experiment.
The realistic test split contains **0 positive and 3,950 negative event
labels**. Direct-head and motion-derived classification metrics are therefore
**N/A**, not zero or perfect, because one-class data cannot measure event
discrimination. The 0.2 m outcome is explicitly synthetic and must not be
described as a realistic operability result.

### 12.5 Phase/timing and frozen reporting

M11 estimates horizon-wise lag by cross-correlation. Random Forest has median
lags of -0.3 s for heave, -1.7 s for pitch, and 0.0 s for roll; the primary
Transformer has -0.1 s, +0.1 s, and -0.1 s respectively, but its large motion
error means this apparent timing alignment is not a superiority claim. M12
verified the immutable inputs before and after M8–M11, then generated the final
benchmark report and conservative CV-claim file only from new frozen-test
artifacts.

## 13. Main technical findings

1. **Simulation identity materially changes evaluation.** BTP-2's contiguous
   within-run holdout is nearly solved by Ridge, while BTP-1's independent
   8 kn simulation remains difficult after training at 2 kn.

2. **Random Forest is the strongest primary aggregate model.** It achieves
   Macro NRMSE 593.585413 and persistence skill 0.776001, narrowly ahead of
   XGBoost and LSTM under the frozen common protocol.

3. **Architecture complexity did not guarantee better generalization.** The
   Transformer failed catastrophically on the speed-shifted BTP-1 test run and
   did not outperform Random Forest.

4. **Baseline choice affects the conclusion.** Persistence performs well at
   the first forecast step but degrades sharply over five seconds. Positive
   persistence skill on BTP-1 does not imply high R² or complete distribution
   generalization.

5. **Feature availability limits speed conditioning.** Adding a constant speed
   value cannot teach speed dependence when the training split contains only
   one speed. Genuine speed-conditioned learning requires multiple independent
   training runs across speeds.

6. **Leakage control is essential with overlapping windows.** Splitting raw
   timelines and runs before window generation prevents thousands of highly
   overlapping samples from creating falsely optimistic results.

7. **Operability labels must match available physics.** Without verified
   propeller geometry, the project uses a causal heave proxy and explicitly
   avoids claiming real emergence detection.

8. **Auxiliary operability heads caused negative transfer in this experiment.**
   The 0.5 m multi-task model increased frozen-test motion Macro NRMSE by
   0.065420 relative to the motion-only model.

## 14. Core engineering and research contributions

The substantive implementation work in this project includes:

- a schema-aware parser for heterogeneous marine simulation CSV layouts;
- source-row provenance and timestamp-reset-based run discovery;
- run-aware, split-aware 10 Hz interpolation;
- run-held-out and purged chronological split strategies;
- train-only feature and target normalization;
- independent split/run window generation for 100-to-50 forecasting;
- automated raw-timestep intersection and run-boundary tests;
- a shared tensor-loading and prediction schema for all model families;
- seven directly comparable forecasting implementations;
- validation-only checkpoint selection and early stopping;
- one central physical-unit, normalized, skill, horizon, and latency evaluator;
- complete per-window prediction persistence for metric traceability;
- feature-ablation and leave-one-speed-out experiment implementations;
- a multi-head Transformer for motion, continuous operability score, and event
  classification;
- a causal threshold-label construction that avoids using future information;
- explicit handling of one-class classification tests as N/A.

Primary technologies used are Python, pandas, NumPy, Parquet, scikit-learn,
XGBoost, PyTorch, pytest, and Apple Metal/CUDA-compatible neural execution.

## 15. Limitations and claims that must remain bounded

1. BTP-1 has one run at each of 2, 5, and 8 kn. The LOSO study therefore has
   one test simulation per operating speed and cannot estimate broad
   uncertainty across repeated simulations.

2. BTP-2 has no verified vessel-speed metadata. No speed value was inferred
   from reports or inserted into its primary inputs.

3. The BTP-1 primary split deliberately combines run holdout with a speed
   shift. It measures a difficult operational change but cannot isolate whether
   degradation comes from speed, wave realization, or other run differences.

4. Primary results use one random seed. They are deterministic benchmark
   results, not a multi-seed confidence interval.

5. BTP-1's very small 2 kn training target standard deviations inflate NRMSE
   on the more energetic 8 kn test run. Physical-unit metrics and R² must be
   interpreted alongside NRMSE.

6. No genuine numeric propeller geometry was found. The 0.5 m and 0.2 m
   experiments are operability proxies, not validated propeller-emergence
   physics.

7. The 0.5 m realistic test currently has one event class. Classification
   metrics are N/A, not zero and not perfect.

8. The primary Transformer result is negative. A CV, paper, or interview must
   not claim that it outperformed the strongest baseline.

9. M8–M12 are frozen and reproducible from the stored Kaggle artifact cache;
   the supported claims remain limited by one run per speed and one-class
   realistic-threshold test labels.

## 16. Verified fact bank for CV or interview writing

The following facts can be safely used to construct concise CV bullets:

- Developed a multi-horizon forecasting system that maps 10 seconds of wave
  and 6-DoF motion history to a 5-second, 50-step trajectory for surge, sway,
  heave, roll, pitch, and yaw.
- Audited 23,140 eligible raw telemetry rows across BTP-1 and BTP-2 and
  identified four independent simulation runs, including 2, 5, and 8 kn BTP-1
  operating conditions.
- Designed raw-timeline splitting, 15-second boundary purges, train-only
  normalization, and run-aware interpolation/window generation to prevent
  cross-split and cross-simulation leakage.
- Created 2,737 immutable forecast windows across both datasets: 528 training,
  332 validation, and 1,877 test windows.
- Implemented and compared persistence, linear trend, Ridge, Random Forest,
  XGBoost, LSTM, and Transformer models using identical inputs, targets, and
  partitions.
- Built a central evaluator for per-DoF RMSE/MAE/R², train-normalized Macro
  NRMSE, persistence skill, 50-step horizon error, fit time, and CPU p50/p95
  latency.
- Found Random Forest to be the strongest aggregate model with Macro NRMSE
  593.585413 and persistence skill 0.776001; it narrowly exceeded XGBoost and
  LSTM, while the Transformer did not outperform it.
- Demonstrated a major evaluation gap between near-linear within-run BTP-2
  forecasting and independent cross-speed BTP-1 generalization.
- Implemented feature ablations and three leave-one-speed-out folds, with Ridge
  Macro NRMSE increasing from 0.092745 at the held-out 2 kn run to 1.326371 at
  the held-out 8 kn run.
- Designed a multi-task Transformer with motion, continuous score, and direct
  event heads plus a causal heave-based operability proxy; the frozen 0.5 m
  comparison found negative transfer rather than a motion improvement.
- Added hard automated tests proving zero cross-split raw-timestep coverage,
  train-only scaler fitting, and zero run-boundary crossing.

CV wording should emphasize leakage-free experimental design, multi-model
forecasting, cross-condition validation, and honest model comparison. It should
not emphasize notebook execution infrastructure, claim real-vessel deployment,
claim physical propeller-emergence detection, or state that the Transformer was
the best model.

## 17. Evidence used for this document

The detailed claims above can be traced to:

- `docs/LEGACY_AUDIT.md` for the source-code and notebook audit;
- `data/manifests/data_audit.md` and `run_manifest.parquet` for datasets and
  run boundaries;
- `data/splits/BTP-1_metadata.json` and `BTP-2_metadata.json` for split,
  scaler, and window counts;
- `outputs/primary_benchmark_m37/metrics_by_dof.csv` for physical and
  normalized test metrics;
- `outputs/primary_benchmark_m37/metrics_by_horizon.csv` for 50-step error
  profiles;
- `outputs/primary_benchmark_m37/primary_comparison.csv` for the primary model
  table;
- `outputs/m8_m12/m8_feature_ablations.csv` for feature experiments;
- `outputs/m8_m12/m9_speed_ood.csv` for leave-one-speed-out results;
- `outputs/m8_m12/m10_multitask_summary.json` for the complete single-task,
  score-only, direct-head, and synthetic stress-test comparison;
- `outputs/m8_m12/m11_phase_summary.json` and `m11_phase_lag.csv` for timing;
- `outputs/reports/FINAL_BENCHMARK_REPORT.md` and `outputs/cv/CV_CLAIMS.md`
  for frozen-test-only report and CV wording;
- `src/shipmotion/benchmark.py`, `src/shipmotion/data/pipeline.py`, and
  `src/shipmotion/phase2.py` for the implementation details.

No numerical result from the legacy notebooks or reports is included.
