#!/bin/bash
#SBATCH --job-name=fireworks_run      # Job name
#SBATCH --output=fw_%j.out           # Standard output
#SBATCH --error=fw_%j.err            # Standard error
#SBATCH --time=12:00:00              # Time limit (HH:MM:SS)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=4                   # Number of tasks
#SBATCH --cpus-per-task=1            # Number of CPUs per task
#SBATCH --qos=regular                # Queue
#SBATCH --constraint=cpu

# Load Shifter
module load shifter

# Run the Docker image using Shifter
shifter --image=docker:nadafalou/legacypm:3 python legacypm_workflow.py

