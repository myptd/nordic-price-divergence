# Price co-movement and price information in Nordic electricity zones

Code and results for the paper *Price co-movement and price information diverge across Nordic electricity zones after
the energy crisis* (Phan, Truong, Ha and Nguyen).

> **Note on data and reproducibility.** Most input data cannot be redistributed here because of their licences:
> ENTSO-E (prices, load, wind and solar forecasts, reservoir stored energy, generation), NVE (reservoir statistics) and
> the commodity prices (Yahoo Finance, EU ETS allowance price). This repository therefore contains the code, the weather
> and map data, the model forecasts **without the observed prices**, and all result tables and figures. Before any
> script can be run, you must download the data yourself (free ENTSO-E API key needed) with
> `bash sh_runs/01_collect_data.sh`, build the cleaned data with `sh_runs/02_preprocess.sh` and add the observed prices
> to the forecast files with `python bin/restore_actuals.py` (see "Reproducing the results"). The commodity prices are
> optional and only needed for pooled LightGBM, the neural networks of the screening, the transfer matrix and the error
> diagnostics (see "Data"). All numbers, tables and figures of the paper are already in `results/` and
> `paper_outputs/`, so they can be checked without running anything.

The paper measures how closely the day-ahead prices of the Norwegian and Swedish bidding zones move with
Germany-Luxembourg (DE-LU) from 2018 to 2025. It then forecasts the 2025 prices of each zone at 11:00 on the previous
day and tests which inputs (neighbouring prices, wind and solar forecasts, reservoir levels) improve accuracy.

## Contents

```
sh_runs/        pipeline steps 00-12 (Slurm batch scripts, numbered in run order) and run_all.sh
bin/            zone definitions, download scripts, preprocessing, data checks, restore_actuals.py
models/         information set (features_day_ahead.py), LEAR, LightGBM, networks, 2024 screening,
                rolling test (rolling.py), transfer matrix
analysis/       01-19 results, 20-30 paper figures and tables; see analysis/README.md
data/           Open-Meteo weather (data/weather/) and Natural Earth borders (data/naturalearth/)
model_results/  forecasts (y_pred) of the 2024 screening, the 2025 rolling test and the 2024 check; transfer matrix
results/        result tables and text summaries quoted in the paper
paper_outputs/  paper and supplement figures (PDF/JPG) and LaTeX tables
```

## Installation

```bash
conda env create -f env.yml      # creates the "elec" environment (Python 3.13, LightGBM, PyTorch)
conda activate elec
```

PyTorch is used only by the neural networks of the 2024 screening (`sh_runs/04_screening_deep.sh`).

## Data

| Source | Content | Included | How to get it |
|---|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | day-ahead prices (13 zones), load and wind/solar forecasts, weekly reservoir stored energy, 2024 generation by type | no | `bin/collect_entsoe.py`, `bin/collect_generation_entsoe.py` (free API key) |
| [NVE](https://www.nve.no) Magasinstatistikk API | Norwegian reservoir fill share and its 20-year median | no | `bin/collect_reservoir_nve.py` |
| [Open-Meteo](https://open-meteo.com) historical archive (ECMWF IFS, ERA5, ERA5-Land) | hourly weather, 1-6 points per zone | yes, `data/weather/` (CC BY 4.0, weather data by Open-Meteo.com) | `bin/collect_weather_openmeteo.py` |
| [Natural Earth](https://www.naturalearthdata.com) | 1:50m country borders (Fig. 1) | yes, `data/naturalearth/` (public domain) | `sh_runs/01_collect_data.sh` |
| Yahoo Finance; ECB | daily closes of TTF gas, API2 coal, Brent, Henry Hub, EUR/NOK; USD/EUR rate | no | optional: `bin/collect_commodities_yfinance.py`, `bin/collect_fx_ecb.py` |
| EU ETS allowance price | front-December future, EUR/t, daily | no | optional: supplied by the user (see below) |

Only data that may be redistributed are included. The observed prices are therefore removed from the forecast files in
`model_results/` (column `y_true`); `bin/restore_actuals.py` adds them back after the data have been downloaded.

**Commodity prices (optional).** Pooled LightGBM (2024 screening, benchmark), the neural networks, the transfer matrix and
the error diagnostics (`analysis/10_crossborder_gas.py`, `analysis/19_error_diagnostics.py`) also use daily commodity
prices. To include them, run the two optional download scripts above and place a daily EU ETS allowance price
(front-December future, EUR/t) in `data/commodities/carbon_eua_ember.csv`, with the columns `date` (YYYY-MM-DD, trading
days) and `Close`. The paper used the series from Ember's European electricity prices and costs tool (data from Montel).
Without these files, `data/cleaned/` is built without the `com_*` columns; the steps that need them stop with a message,
and `run_all.sh`, `05_benchmark.sh` and `12_analysis.sh` skip them.

## Reproducing the results

All steps run from the repository root. The jobs in `sh_runs/` are Slurm batch scripts (written for the Saga cluster);
set your project account once with `source sh_runs/00_set_slurm_account.sh`. Array jobs run one task per zone or
setting; to run a single task without Slurm, set the task index first, for example
`SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=4 bash sh_runs/05_benchmark.sh`.

**1. Data** (required for every step):

```bash
export ENTSOE_API_KEY="..."            # free key from the ENTSO-E Transparency Platform
bash sh_runs/01_collect_data.sh        # downloads (existing files are skipped) and raw-data checks
sbatch sh_runs/02_preprocess.sh        # data/cleaned/ and checks
python bin/restore_actuals.py          # observed prices into the included forecast files
```

**2. Figures and tables from the included forecasts** (no model training):

```bash
sbatch sh_runs/12_analysis.sh          # all result files, figures and tables -> results/, paper_outputs/
```

**3. Models.** `bash sh_runs/run_all.sh` submits every job with dependencies (`SKIP_PREPROCESS=1` starts from the
existing `data/cleaned/`). The jobs one by one:

| Job | Content | Paper |
|---|---|---|
| `03_screening_classic`, `04_screening_deep` | screening on 2024: LEAR, LightGBM grid, LSTM/TCN/Transformer; no 2025 data | Table S1, Section S2 |
| `05_benchmark` | 2025 benchmark: LEAR, pooled and per-hour LightGBM | Table 3, Table S2, Fig. S5 |
| `06_crossborder` | per-hour LightGBM with previous-day neighbouring prices | Fig. 4, Table S5 |
| `07_training_windows`, `11_transfer_matrix` | training windows and the raw target; train on one year, test on another | Fig. 3, Fig. S4 |
| `08_tso_forecasts` | without the load, wind and solar forecasts | Fig. 5, Fig. S8, Tables S6-S7, Fig. S9 |
| `09_reservoir_inputs` | with the reservoir inputs | Section 4.4 |
| `10_tso_2024` | base and no wind/solar forecasts on 2024 | Tables S8-S9 |
| `12_analysis` | every result file, figure and table (`analysis/01-30`) | Figs. 1-6, Tables 2-3, Figs. S1-S9, Tables S1-S9 |

`model_results/screening/` holds the screening that fixed the model choices (Table S1). It was run with an earlier
carbon price series, so the current data reproduce `model_results/screening_rerun/` instead (Supplementary Section S2).

## Design notes

- Forecast task: the 24 hourly prices of day D at 11:00 on D-1, using only values whose measurement is complete
  before that time (`models/features_day_ahead.py`). Publication delays are not modelled: wind and solar forecasts
  and the weekly reservoir values are assumed available at 11:00 on D-1.
- Model choices were fixed on 2024 data; 2025 is used only for testing, with monthly recalibration and an expanding
  window from 2018-01-08 (`models/rolling.py`).
- Accuracy is reported as rMAE (MAE relative to the D-7 naive forecast) and compared with multivariate
  Diebold-Mariano tests with Holm adjustment.
- ENTSO-E prices are quarter-hourly from 2025-10-01 and are averaged to hours. The DE-LU price before 2018-10-01 is the
  DE-AT-LU price. The NO5 wind and solar forecast is zero in every year.

## Licence

The code is released under the MIT licence (`LICENSE`). The included data remain under the terms of their sources
(see "Data").
