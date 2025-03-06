#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64  # Adjust based on the required parallelism
#SBATCH --partition=gpuq
#SBATCH --output=clustermatch_%j.out

# Load necessary modules
module load python
module load psutil  # If your script monitors resources

# Run Clustermatch in parallel (modify the script if it handles multiprocessing internally)
python3 -impl.py
