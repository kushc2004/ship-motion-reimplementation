# Legacy audit (M0)

This audit is based on the actual files in this repository. It deliberately does
not carry any legacy metric into the new M0-M2 artifacts.

## Artifact inventory and verified findings

| Artifact | Verified behavior / issue |
|---|---|
| `BTP-1/random_forest_ship_motion (2).ipynb` | Creates all 100+50-step windows first, then takes the first 80% of windows. It sorts only by time, uses `np.interp` on a single table, and has no run identity. |
| `BTP-1/XGBoost (1).ipynb` | Same pre-split window construction, flattened six-motion targets, and an 80/20 window split. It omits wave and speed from the model input. |
| `BTP-1/LSTMMX (2).ipynb` | Reads a file not present here, uses hand-defined columns, interpolates at 0.2 s, fits `MinMaxScaler` to the full series, and fits the LSTM on all sequences before reporting a later nominal test slice. |
| `BTP-1/ShipMotionTransformer_Colab (1).ipynb` | Detects resets/speed changes but adds cumulative time offsets and drops segment identity, joining simulations. It cubic-interpolates the joined table, fits `StandardScaler` globally, then splits windows. |
| `BTP-2/ShipMotionTransformer_Colab.ipynb` | Sorts one global clock, drops duplicate time, cubic-interpolates globally, fits scaling/statistics globally, and windows before splitting. |
| `BTP-2/BTP2_ShipMotion_Propeller_Transformer.ipynb` | Phase-2 operability code; global interpolation/scaling and pre-split windows. Explicitly out of scope for M0-M2. |
| `BTP-2/generate_plots.py` | Contains synthetic plotting comments/data, including “matching” old metrics; not used by the new pipeline. |
| `BTP-2/generate_extra_plots.py` | Generates synthetic samples and hard-coded illustrative scores; not used by the new pipeline. |
| `BTP-1/BTP-PPT.pptx` | Presents model comparison numbers and a 10 Hz/cubic/global-scaling pipeline; these are legacy claims, not new outputs. |
| `BTP-2/Latex_Report/*.tex` | States 2/5/8 kn, cubic interpolation, standard scaling, and old model/operability results. Those claims are not copied into new data or result files. |
| `BTP-2/BTP-2-updates.pdf` | Existing update artifact; retained as historical evidence only. |
| `BTP-2/BTP_2_Report_22NA3AI39_Kush_Chaudhari (2).pdf` | Listed by the specification but absent from this checkout. The available report source is under `BTP-2/Latex_Report/`. |

## Dataset audit

The eligible motion CSVs are `BTP-1/BTP - Time-Series Data.csv` and
`BTP-2/BTP_2_data - Sheet1.csv`. The two additional BTP-1 files were inspected
row-by-row, including rows below the first CSV row:

* `BTP-1/Data.csv` has ten repeated RAO headers at zero-based rows
  `1, 94, 187, 280, 373, 466, 559, 652, 745, 838`. Their preceding condition
  labels are 2.5 kn at headings 0/45/90/135/180 degrees and 8 kn at the same
  five headings. The columns are encounter/wave frequency, wavelength, RAO,
  and phase values; there is no time column and no six-DoF telemetry schema.
* `BTP-1/Ship-Parameters-data.csv` is a side-by-side metadata export. Its first
  row identifies `Hydrostatic Coefficients` and `INPUT PARAMETERS FOR DATA`;
  the right-hand table uses `Parameter`, `Values to Use`, and `Count`. It has
  no motion time series.

The notebook loaders do not make these files eligible: the Excel branches in
the Random Forest and XGBoost notebooks look for a `Time-Series Data` sheet,
while their CSV branches assume the header is row zero. The LSTM notebook has a
special text search for a `time series` marker, but its referenced source file
is absent. The Transformer notebooks also read a normal row-zero CSV header.
The new audit records the embedded headers and keeps both files excluded from
telemetry discovery.

Verified canonical input columns are `time_s`, `wave_m`, optional
`ship_speed_kn`, and `surge_m`, `sway_m`, `heave_m`, `roll_deg`, `pitch_deg`,
`yaw_deg`. The source spellings are preserved in the raw files and normalized
only in the manifest.

`BTP-1/BTP - Time-Series Data.csv` has 16,407 rows and three runs:

| run | rows | time range (s) | speed (kn) | boundary |
|---|---:|---:|---:|---|
| 1 | 1,272 | 0 to 232.864 | 2 | file start |
| 2 | 1,624 | 0 to 268.684 | 5 | time reset at source row 1,272 |
| 3 | 13,511 | 0 to 1,811.994 | 8 | time reset at source row 2,896 |

There are 32 repeated clock values globally because the clock restarts; there
are no repeated `(run_id,time_s)` values after boundary assignment. Samples are
irregular rather than an already-uniform 10 Hz grid.

`BTP-2/BTP_2_data - Sheet1.csv` has 6,733 rows, one strictly increasing run,
time 0 to 539.513 s, no speed column, no nulls, and no duplicate timestamps.
The report's speed claims cannot be verified from this CSV and are not invented
in the manifest.

## Boundary, split, and preprocessing contract

Run identity is scoped by source file and starts a new run at every non-increasing
timestamp. A speed change alone is not a boundary. Each run is interpolated only
inside its assigned split block, with linear interpolation at 10 Hz; speed is
carried as a constant metadata value and is never cubic-interpolated.

The BTP-1 artifact uses run-held-out splitting because three independent runs are
available (train/validation/test respectively). BTP-2 uses purged temporal
splitting because only one run is available: 60%/20%/20% chronological raw
timeline with a 150-step (15-second at 10 Hz) purge between partitions. Splits
occur before interpolation, scaling, or windows. The train scaler is fit only on
interpolated training rows. Windows are made independently per `(run_id,split)`
with input=100, output=50, stride=10.

The manifests include raw row coverage for every window. Tests assert that
coverage intervals from different splits do not intersect and that no window
crosses a run or split boundary.
