#!/bin/bash
#SBATCH --job-name=03_screening_classic
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

# Screening on 2024 (no 2025 data): LEAR (AIC) and the LightGBM grid (pooled and per hour, three targets); fit
# 2018-2022, early stopping on 2023, refit 2018-2023, evaluate 2024. Runs models/screening.py, one array task per zone.
# Inputs: data/cleaned/. Outputs: $SCREENING_OUT/{zone}_classic.parquet and {zone}_trees.csv.
# SCREENING_OUT defaults to model_results/screening_rerun, the screening with the current commodity inputs.
# model_results/screening holds the run that fixed the frozen choices; it was made with the earlier carbon input and
# cannot be reproduced from the current data/cleaned.
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/03_screening_classic.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SCREENING_OUT=${SCREENING_OUT:-model_results/screening_rerun}
python -c "import sys, pandas, numpy, sklearn, lightgbm; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__)"

## One array task per zone: 0-9 = NO1..NO5, SE1..SE4, DE_LU
ZONES=(NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4 DE_LU)
zone=${ZONES[$SLURM_ARRAY_TASK_ID]}
echo "zone=$zone SCREENING_OUT=$SCREENING_OUT"
python models/screening.py --zones "$zone"
