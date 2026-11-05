#!/bin/bash
#SBATCH --job-name=10_tso_2024
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

# Per-hour LightGBM on the test year 2024 (the screening year), with all inputs (base) and without the wind and solar
# forecasts (no_vre): frozen settings, monthly recalibration, expanding window (models/rolling.py --year 2024).
# Inputs: data/cleaned/. Outputs: model_results/rolling/{zone}_lgbm_hourly_{base,no_vre}_expanding_2024.parquet
# and _params.csv.
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/10_tso_2024.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import sys, pandas, numpy, sklearn, lightgbm; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__)"

## One array task per zone: 0-9 = NO1..NO5, SE1..SE4, DE_LU
ZONES=(NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4 DE_LU)
zone=${ZONES[$SLURM_ARRAY_TASK_ID]}
echo "zone=$zone"
for i in base no_vre; do
    python models/rolling.py --zones "$zone" --models lgbm_hourly --inputs "$i" --window expanding --year 2024
done
