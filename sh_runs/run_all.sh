#!/bin/bash
# =============================================================================
# run_all.sh - submit the whole pipeline after the data download (01_collect_data.sh), with Slurm dependencies:
#   02 preprocessing -> 03-11 models (in parallel) -> 12 analysis.
# Runs on the login node and only calls sbatch. Wall time about 2-3 hours when the queue is free.
# Run from the repository root:
#   source sh_runs/00_set_slurm_account.sh
#   bash sh_runs/run_all.sh                 # SKIP_PREPROCESS=1 to start from the existing data/cleaned/
# =============================================================================
set -eo pipefail
mkdir -p logs
sub() { sbatch --parsable "$@"; }

dep=""
if [ -z "$SKIP_PREPROCESS" ]; then
    pre=$(sub sh_runs/02_preprocess.sh); echo "02_preprocess $pre"
    dep="--dependency=afterok:$pre"
fi
ids=()
JOBS="05_benchmark 06_crossborder 07_training_windows 08_tso_forecasts 09_reservoir_inputs 10_tso_2024"
if ls data/commodities/*_daily.csv data/commodities/carbon_eua_ember.csv >/dev/null 2>&1; then
    JOBS="03_screening_classic 04_screening_deep $JOBS 11_transfer_matrix"
else
    echo "No commodity files in data/commodities/: screening (03, 04) and transfer matrix (11) not submitted (see README)"
fi
for s in $JOBS; do
    id=$(sub $dep sh_runs/$s.sh); echo "$s $id"; ids+=("$id")
done
ana=$(sub --dependency=afterok:$(IFS=:; echo "${ids[*]}") sh_runs/12_analysis.sh); echo "12_analysis $ana"
