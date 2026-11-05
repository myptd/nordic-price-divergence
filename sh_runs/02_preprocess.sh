#!/bin/bash
#SBATCH --job-name=02_preprocess
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/%x_%j.out

# Merge the raw data into one hourly file per zone and audit the result.
# bin/data_preprocess.py runs its data quality guard first (runs of identical values, forward-fill lengths, zero volumes)
# and stops on failure; bin/audit_clean_data.py checks the cleaned files; bin/audit_price_processing.py re-derives the
# hourly prices independently (Supplementary S1).
# Inputs: data/entsoe/, data/weather/, data/reservoir/, data/commodities/.
# Outputs: data/cleaned/{ZONE}_hourly.parquet, data/cleaned/panel_hourly.parquet, data/cleaned/coverage.csv.
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/02_preprocess.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
python -c "import sys, pandas, numpy; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__)"

python bin/data_preprocess.py
python bin/audit_clean_data.py
python bin/audit_price_processing.py
