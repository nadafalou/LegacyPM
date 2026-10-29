#!/bin/bash
# Run the LegacyPM per-brick pipeline for whole RA slices on one CPU node. See RUNBOOK.md.
#   sbatch run_bricks.sbatch 000-015            # inputs picked automatically: Dustin's pscratch run, his CFS run, our archive
#   sbatch run_bricks.sbatch 060 --workers 64   # extra options are passed to run_bricks.py (later ones win)
# Restartable: resubmit the same command and finished bricks are skipped.
#SBATCH --job-name=legacypm
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --image=docker:nadafalou/legacypm:5
#SBATCH --output=/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/%x-%j.out

# run from the repo clone this script lives in (Slurm copies the script, so use the submit dir)
cd "$SLURM_SUBMIT_DIR"
# one process per core: stop numpy/BLAS from starting extra threads in every worker
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
shifter python run_bricks.py --workers 128 "$@"
