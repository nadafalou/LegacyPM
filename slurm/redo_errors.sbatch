#!/bin/bash
#SBATCH -A desi -q regular -C cpu -N 1 -t 06:00:00
#SBATCH -J redo-errors
#SBATCH --image=docker:nadafalou/legacypm:5
#SBATCH -o /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/%x-%j.out
# Finished bricks only: rewrite the total errors (py_files/error_model.json) and the exposure zero points
# (zeropoints.py table), then refit the PMs. Bricks already made with the current error model AND table are skipped,
# so resubmit the same command to continue. See RUNBOOK.md sections 5 and 8.
#   cd slurm && sbatch redo_errors.sbatch 000-079        # extra args go to run_bricks.py (later ones win)
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
shifter python run_bricks.py "$@" --redo-errors --workers 64
