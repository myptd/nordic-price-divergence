#!/bin/bash
#SBATCH --job-name=04_screening_deep
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-26
#SBATCH --output=logs/%x_%A_%a.out

# Screening on 2024 (no 2025 data) of LSTM, TCN and Transformer, NO1-NO5 and SE1-SE4: fit 2018-2022, early stopping on
# 2023, refit 2018-2023, evaluate 2024, three seeds, CPU. Runs models/screening_deep.py, one array task per model and zone.
# Inputs: data/cleaned/. Outputs: $SCREENING_OUT/{zone}_{model}.parquet and {zone}_{model}_epochs.csv
# (SCREENING_OUT as in 03_screening_classic.sh).
# Submit from the repository root (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/04_screening_deep.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SCREENING_OUT=${SCREENING_OUT:-model_results/screening_rerun}
python -c "import sys, pandas, numpy, torch; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"

## One array task per (model, zone): 0-8 = lstm NO1..SE4, 9-17 = tcn, 18-26 = transformer
MODELS=(lstm tcn transformer)
ZONES=(NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4)
model=${MODELS[$((SLURM_ARRAY_TASK_ID / 9))]}
zone=${ZONES[$((SLURM_ARRAY_TASK_ID % 9))]}
echo "model=$model zone=$zone SCREENING_OUT=$SCREENING_OUT"
python models/screening_deep.py --model "$model" --zones "$zone"
