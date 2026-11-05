#!/bin/bash
#SBATCH --job-name=12_analysis
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out

# All result files, figures and tables of the paper: runs analysis/01-30 in numeric order (each script lists its inputs
# and outputs in its header). No model is trained.
# Inputs: data/cleaned/, data/entsoe/generation/, data/naturalearth/, model_results/ (jobs 03-11).
# Outputs: results/ (csv, txt), paper_outputs/figures/ (pdf, jpg), paper_outputs/tables/ (LaTeX).
# Submit from the repository root after jobs 03-11 (account from SBATCH_ACCOUNT, see 00_set_slurm_account.sh):
#   sbatch sh_runs/12_analysis.sh
# A subset:  sbatch --export=ALL,SCRIPTS="21_fig2_divergence 25_fig6_diagnostics" sh_runs/12_analysis.sh
set -eo pipefail

eval "$(conda shell.bash hook)"   # batch shells do not read ~/.bashrc
conda activate elec
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import sys, pandas, numpy, scipy, statsmodels, sklearn, lightgbm, matplotlib; print('python', sys.version.split()[0], 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'statsmodels', statsmodels.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__, 'matplotlib', matplotlib.__version__)"

DEFAULT=$(cd analysis && ls [0-9][0-9]_*.py | sed 's/\.py$//')
# 10_crossborder_gas and 19_error_diagnostics use the commodity prices; without them in data/cleaned/ they are skipped
# and the existing results/h3_2025* and results/diagnostics_2025* are kept.
if [ -z "$SCRIPTS" ] && ! python -c "import sys, pyarrow.parquet as pq; sys.exit(0 if any(c.startswith('com_') for c in pq.read_schema('data/cleaned/NO1_hourly.parquet').names) else 1)"; then
  echo "No commodity inputs in data/cleaned/: skipping 10_crossborder_gas and 19_error_diagnostics"
  DEFAULT=$(echo $DEFAULT | sed 's/10_crossborder_gas//; s/19_error_diagnostics//')
fi
for s in ${SCRIPTS:-$DEFAULT}; do
  echo "=== $s"; python analysis/$s.py
done
