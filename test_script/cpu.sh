#!/bin/bash

#SBATCH --partition=gpu                # Use the same partition where gypsum-gpu106 belongs

#SBATCH --nodelist=gypsum-gpu[106,108-115,118-119,121,124]       # Force it to run on this node

#SBATCH --gres=gpu:1080_ti:1

#SBATCH --nodes=1

#SBATCH --cpus-per-task=24

#SBATCH --time=08:00:00

#SBATCH --mem=370g

#SBATCH --job-name=ccc_cpu

#SBATCH --output=ccc_cpu_%j.out

#SBATCH --error=ccc_cpu_%j.err

#SBATCH --exclusive



PYTHON_SCRIPT=test_cpu_ccc.py

LOGFILE="logs/cpu_ccc_scaling.log"



export NODES=1

export THREADS=24    # Optional, only if your code uses threads



echo "NODES THREADS SIZE FEATURES TIME(s)" > $LOGFILE



for FEATURES in {2..20..2}; do

  for SIZE in 10000 100000 1000000 10000000; do

    export FEATURES

    export SIZE

    echo "Running CCC (CPU) with SIZE=$SIZE, FEATURES=$FEATURES..."

    OUTPUT=$(python $PYTHON_SCRIPT)

    echo "$OUTPUT" | tee -a $LOGFILE

  done

done

