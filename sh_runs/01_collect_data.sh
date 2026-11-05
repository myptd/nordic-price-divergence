#!/bin/bash
# =============================================================================
# 01_collect_data.sh - download all raw data and audit it
#
# Runs on a machine with internet access (a laptop or the Saga login node; compute nodes have no internet),
# not as a Slurm job: the scripts only download files. Every collector skips files it already has, so rerun after an
# interruption. ENTSO-E needs a free API key.
# Scripts: bin/collect_reservoir_nve.py, collect_entsoe.py,
# collect_generation_entsoe.py, collect_weather_openmeteo.py (about 2 hours), bin/audit_raw_data.py.
# Outputs: data/reservoir/, data/entsoe/, data/entsoe/generation/, data/naturalearth/, data/weather/.
# Run from the repository root:
#   export ENTSOE_API_KEY="your-key"
#   bash sh_runs/01_collect_data.sh
# Next: sbatch sh_runs/02_preprocess.sh
# =============================================================================
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate elec
python -c "import sys, pandas, numpy; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__)"

python bin/collect_reservoir_nve.py            # NVE reservoir statistics
# Optional commodity prices (see README): python bin/collect_commodities_yfinance.py; python bin/collect_fx_ecb.py;
# data/commodities/carbon_eua_ember.csv is supplied by the user.
python bin/collect_entsoe.py                   # prices, load and wind/solar forecasts, reservoir (needs ENTSOE_API_KEY)
python bin/collect_generation_entsoe.py        # actual generation by type, 2024 (market-type table only)

mkdir -p data/naturalearth                     # Natural Earth 1:50m countries (public domain), base map of Fig. 1
[ -f data/naturalearth/ne_50m_admin_0_countries.geojson ] || curl -L -o data/naturalearth/ne_50m_admin_0_countries.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson

python bin/collect_weather_openmeteo.py        # Open-Meteo weather (about 2 hours, hourly API limit)
python bin/audit_raw_data.py                   # prints ALL CHECKS PASSED or the failing checks
