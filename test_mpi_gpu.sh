#!/bin/bash
#SBATCH --job-name=ccc_simple_test
#SBATCH --output=ccc_test_4rank2gpu_%j.out
#SBATCH --error=ccc_test_4rank2gpu_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=gypsum-gpu[160-164,166,168,171,173-177,181,190-192]
#SBATCH --mem=370G
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclusive

set -e

# Parameters 
PYTHON_SCRIPT=test_mpi_gpu.py
PYTHONPATH=./libs

SIZE=10000000
FEATURES=4
RANKS=24

OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
        SIZE=$SIZE FEATURES=$FEATURES \
        mpiexec -n $RANKS python $PYTHON_SCRIPT)

TIME_TAKEN=$(echo "$OUTPUT" \
            | grep "\[rank 0\]" \
            | grep -oE "[0-9]+\.[0-9]+")

echo "$OUTPUT"
echo "Time taken: $TIME_TAKEN seconds"
