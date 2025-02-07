#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=00:10:00            # Request 10 minutes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=regular                # Use debug queue
#SBATCH --constraint=cpu

echo "Hello from Slurm!"
