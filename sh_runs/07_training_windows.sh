#!/bin/bash
#SBATCH --job-name=07_training_windows
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

# Training windows, 2025: per-hour LightGBM with the level-and-scale target on the last 365, 730 and 1,095 days and on
# the expanding window without 2021-2022, and with the raw target on the expanding and 365-day windows
# (models/rolling.py). The expanding level-and-scale forecasts come from 05_benchmark.sh.
# Inputs: data/cleaned/. Outputs: model_results/rolling/{zone}_lgbm_hourly_base_{365,730,1095,expanding_no_crisis,
# expanding_none,365_none}.parquet and _params.csv.
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/07_training_windows.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import sys, pandas, numpy, sklearn, lightgbm; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__)"

## One array task per zone: 0-9 = NO1..NO5, SE1..SE4, DE_LU
ZONES=(NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4 DE_LU)
zone=${ZONES[$SLURM_ARRAY_TASK_ID]}
echo "zone=$zone"
for w in 365 730 1095 expanding_no_crisis; do
    python models/rolling.py --zones "$zone" --models lgbm_hourly --window "$w"
done
for w in expanding 365; do
    python models/rolling.py --zones "$zone" --models lgbm_hourly --window "$w" --target none
done
