#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64  # Adjust based on the required parallelism
#SBATCH --partition=defq
#SBATCH --output=clm_%j.out

# Load necessary modules
module load python
module load psutil  # If your script monitors resources
module load CUDA/11.2

# Run Clustermatch with IPython
ipython3 -c "
import numpy as np
import pandas as pd
from ccc.coef import ccc
x = np.random.normal(size=600000)
y = np.random.normal(size=600000)
%prun ccc(x, y, n_jobs=1)
"
