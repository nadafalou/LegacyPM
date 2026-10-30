#!/bin/bash
#SBATCH -A desi -q regular -C cpu -N 1 -t 03:00:00
#SBATCH -J notebooks
#SBATCH -o /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/notebooks-%j.out
# Execute figure notebooks of run/ on a CPU node (they load many bricks; don't run them on a login node).
# Each notebook is executed in place (outputs saved into it); figures go where the notebook saves them (writeup/plots/
# for paper figures). A notebook stops at its first error and is reported as FAILED; the next one still runs.
#   cd slurm && sbatch run_notebooks.sbatch err-plots dcr-plots pm-error-model
cd "$SLURM_SUBMIT_DIR/../run"
export OMP_NUM_THREADS=8
for nb in "$@"; do
  echo "### $nb start $(date +%T)"
  if ~/anaconda3/envs/desi-pm/bin/jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.kernel_name=desi-pm --ExecutePreprocessor.timeout=-1 "$nb.ipynb"; then
    echo "### $nb ok $(date +%T)"
  else
    echo "### $nb FAILED $(date +%T)"
  fi
done
