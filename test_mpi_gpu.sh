#!/bin/bash
#SBATCH --job-name=ccc_simple_test
#SBATCH --output=ccc_test_4rank2gpu_%j.out
#SBATCH --error=ccc_test_4rank2gpu_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclusive

set -e

# Parameters 
PYTHON_SCRIPT=test_ccc_mpi_gpu.py
PYTHONPATH=./libs

SIZE=10000
FEATURES=20
RANKS=4

OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
        SIZE=$SIZE FEATURES=$FEATURES \
        mpiexec -n $RANKS python $PYTHON_SCRIPT)

TIME_TAKEN=$(echo "$OUTPUT" \
            | grep "\[rank 0\]" \
            | grep -oE "[0-9]+\.[0-9]+")

echo "$OUTPUT"
echo "Time taken: $TIME_TAKEN seconds"
