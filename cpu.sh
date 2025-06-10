#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64  # Adjust based on the required parallelism
#SBATCH --partition=defq
#SBATCH --output=cpu_%j.out
#SBATCH --output=ccc_%j.out
#SBATCH --error=ccc_scaling_%j.err


# Load necessary modules
#module load python
#module load psutil  # If your script monitors resources
#module load CUDA/11.2

# Run Clustermatch with IPython
#ipython3 -c "
#import numpy as np
#import pandas as pd
#import time
#from ccc.coef import ccc

#x = np.random.normal(size=(8,10000))
#y = np.random.normal(size=10000)

#start = time.time()
#ccc(x, y)
#end = time.time()
#diff = (end-start)*1000
#print(f"Time taken: {diff} ms")
#"

python test_cpu_ccc.py
