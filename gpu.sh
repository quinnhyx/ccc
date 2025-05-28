#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64  # Adjust based on the required parallelism
#SBATCH --partition=gpuq
#SBATCH --output=ccc_%j.out
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00

# Load necessary modules
module load python
module load psutil  # If your script monitors resources
module load nvidia-smi
module load CUDA/11.2

# Run Clustermatch with IPython
srun --exact -n 1 --ntasks-per-node=64 --time=01:00:00 --pty bash -c "
ipython3 -c '
import numpy as np
import pandas as pd
from ccc.coef import ccc
x = np.random.normal(size=10000)
y = np.random.normal(size=10000)
ccc(x, y, n_jobs=1) '
"
