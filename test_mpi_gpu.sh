#!/bin/bash
#SBATCH --job-name=ccc_simple_test
#SBATCH --output=ccc_gpu_24rank8gpu_%j.out
#SBATCH --error=ccc_gpu_24rank8gpu_%j.err
#SBATCH --nodes=1
#SBATCH --mem=370G
#SBATCH --ntasks-per-node=24
#SBATCH --gpus=8
#SBATCH --constraint=1080ti
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclusive

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters
PYTHON_SCRIPT=test_mpi_gpu.py
PYTHONPATH=./libs

SIZE=10000000
FEATURES=16
# RANKS=24

# OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
#         SIZE=$SIZE FEATURES=$FEATURES \
#         mpiexec -n $RANKS python $PYTHON_SCRIPT)

OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
       SIZE=$SIZE FEATURES=$FEATURES \
       python $PYTHON_SCRIPT)

# TIME_TAKEN=$(echo "$OUTPUT" \
#             | grep "\[rank 0\]" \
#             | grep -oE "[0-9]+\.[0-9]+")

TIME_TAKEN=$(echo "$OUTPUT")

echo "$OUTPUT"
echo "Time taken: $TIME_TAKEN seconds"
