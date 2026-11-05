# =============================================================================
# 00_set_slurm_account.sh - set the Saga project account for all sbatch jobs
#
# sbatch reads SBATCH_ACCOUNT as --account, so no job script contains an account.
# Edit the account below, then load it into your shell (source, not bash):
#   source sh_runs/00_set_slurm_account.sh
# =============================================================================
export SBATCH_ACCOUNT=nnXXXXk      # your Saga project account, e.g. nn1234k
echo "SBATCH_ACCOUNT=$SBATCH_ACCOUNT"
