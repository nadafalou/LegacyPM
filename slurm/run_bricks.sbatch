#!/bin/bash
# Run the LegacyPM per-brick pipeline for whole RA slices on one CPU node. See RUNBOOK.md.
#   sbatch run_bricks.sbatch 060 061 062 063                 # new bricks from Dustin
#   sbatch run_bricks.sbatch 060 061 062 063 --from-archive  # rerun the archived bricks
# Restartable: resubmit the same command and finished bricks are skipped.
#SBATCH --job-name=legacypm
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --image=docker:nadafalou/legacypm:5
#SBATCH --output=/pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs/%x-%j.out

# run from the repo clone this script lives in (Slurm copies the script, so use the submit dir)
cd "$SLURM_SUBMIT_DIR"
shifter python run_bricks.py "$@" --workers 64
