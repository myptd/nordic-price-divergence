# analysis

One script per step, numbered in run order; `sh_runs/12_analysis.sh` runs them all. Each header lists its inputs and
outputs. Run from the repository root (`python analysis/NN_name.py`), on Saga as a Slurm job.

| Scripts | Content | Main outputs |
|---|---|---|
| `01`-`04` | Co-movement with DE-LU by zone and year at three time scales, clustering, cluster membership | `results/comovement/`, Figs. S1-S3 |
| `05` | Market types: 2024 generation shares, coupling, 2025 mean price | `results/market_types_2024.csv`, `tab_market_types` |
| `06` | Screening on 2024 (both screening runs) | `results/screening_*`, `results/screening_rerun/` |
| `07` | 2025 benchmark, multivariate DM tests | `results/benchmark_2025*` |
| `08`-`11` | Neighbouring prices: H1-H2, day bootstrap, H3 (gas), alternative coupling measures and precision | `results/crossborder_*`, `results/h3_2025*`, `results/coupling_alternatives_2024.csv`, `tab_s_coupling_alt` |
| `12`, `13` | Transfer matrix; training windows | `results/transfer/`, `results/windows_2025*` |
| `14`-`17` | Value of the TSO forecasts: without them, against the generation mix, against the reservoir level, 2024 and EUR/MWh | `results/no_forecasts_2025*`, `results/vre_mix.txt`, `results/tso_gain_*`, Figs. S8-S9, `tab_s_tso_gain_*` |
| `18` | Reservoir inputs in per-hour LightGBM | `results/reservoir_2025*` |
| `19` | Error diagnostics (double machine learning) | `results/diagnostics_2025*` |
| `20`-`25` | Figs. 1-6 | `paper_outputs/figures/` |
| `26`-`28` | Benchmark, hypotheses and correlation-by-year tables | `paper_outputs/tables/` |
| `29` | Screening table and Figs. S4-S7 | `paper_outputs/tables/`, `paper_outputs/figures/` |
| `30` | Graphical abstract | `paper_outputs/figures/` |

`paper_style.py` holds the zone order (2024 coupling), the market types and the colour palette of every paper figure;
`dml_tools.py` the market conditions and the double machine learning used by `10` and `19`.
