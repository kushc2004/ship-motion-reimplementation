# Data audit

Generated from the repository CSVs; legacy result metrics are not included.

## Eligible files

- `/Volumes/Extreme SSD/projects/btp/BTP-1/BTP - Time-Series Data.csv`: 16407 rows, 3 runs, columns `time_s, wave_m, ship_speed_kn, surge_m, sway_m, heave_m, roll_deg, pitch_deg, yaw_deg`, speeds `[2, 5, 8]`
- `/Volumes/Extreme SSD/projects/btp/BTP-2/BTP_2_data - Sheet1.csv`: 6733 rows, 1 runs, columns `time_s, wave_m, ship_speed_kn, surge_m, sway_m, heave_m, roll_deg, pitch_deg, yaw_deg`, speeds `[]`

## Run manifest

| run_id | source | rows | time start | time end | speed | reset |
|---|---|---:|---:|---:|---:|---|
| `btp-time-series-data__run01` | `/Volumes/Extreme SSD/projects/btp/BTP-1/BTP - Time-Series Data.csv` | 1272 | 0.0 | 232.864 | 2.0 | False |
| `btp-time-series-data__run02` | `/Volumes/Extreme SSD/projects/btp/BTP-1/BTP - Time-Series Data.csv` | 1624 | 0.0 | 268.684 | 5.0 | True |
| `btp-time-series-data__run03` | `/Volumes/Extreme SSD/projects/btp/BTP-1/BTP - Time-Series Data.csv` | 13511 | 0.0 | 1811.994 | 8.0 | True |
| `btp-2-data-sheet1__run01` | `/Volumes/Extreme SSD/projects/btp/BTP-2/BTP_2_data - Sheet1.csv` | 6733 | 0.0 | 539.513 | None | False |
