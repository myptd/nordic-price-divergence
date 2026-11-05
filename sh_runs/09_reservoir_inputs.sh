#!/bin/bash
#SBATCH --job-name=09_reservoir_inputs
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out

# Per-hour LightGBM with the reservoir inputs (input set reservoir), 2025, frozen settings, monthly recalibration,
# expanding window (models/rolling.py).
# Inputs: data/cleaned/. Outputs: model_results/rolling/{zone}_lgbm_hourly_reservoir_expanding.parquet and _params.csv.
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/09_reservoir_inputs.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import sys, pandas, numpy, sklearn, lightgbm; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__)"

## One array task per Norwegian or Swedish zone: 0-8 = NO1..NO5, SE1..SE4
ZONES=(NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4)
zone=${ZONES[$SLURM_ARRAY_TASK_ID]}
echo "zone=$zone"
python models/rolling.py --zones "$zone" --models lgbm_hourly --inputs reservoir --window expanding
