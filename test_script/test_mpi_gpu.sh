#!/bin/bash
#SBATCH --job-name=ccc_simple_test
#SBATCH --output=ccc_simple_test_%j.out
#SBATCH --error=ccc_simple_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=1080ti
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclusive

set -e

# Parameters 
PYTHON_SCRIPT=test_mpi_gpu.py
PYTHONPATH=./libs

SIZE=10000000
FEATURES=16
RANKS=1

OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
        SIZE=$SIZE FEATURES=$FEATURES \
        mpiexec -n $RANKS python $PYTHON_SCRIPT)

TIME_TAKEN=$(echo "$OUTPUT" \
            | grep "\[rank 0\]" \
            | grep -oE "[0-9]+\.[0-9]+")

echo "$OUTPUT"
echo "Time taken: $TIME_TAKEN seconds"
