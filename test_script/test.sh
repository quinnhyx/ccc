#!/bin/bash
#SBATCH --job-name=ccc_scaling
#SBATCH --output=8rank_1gpu_%j.out
#SBATCH --error=8rank_1gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --constraint=1080ti
#SBATCH --mem=370G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive

set -e
export CUDA_VISIBLE_DEVICES=0

# Parameters 
PYTHON_SCRIPT=test.py
PYTHONPATH=./libs
SIZE=10000000

# Combinations
declare -a COMBINATIONS=(
    "1 24"
)

# features 
declare -a FEATURE_COUNTS=(16)  
declare -a NODE_COUNTS=(1) 


# Initialize log file 
echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST > 4test_hosts.txt
cat 4test_hosts.txt

# Iterate the combination
for NODES in "${NODE_COUNTS[@]}"; do
    head -n $NODES 4test_hosts.txt > 4test_hosts_sub.txt

    for FEATURES in "${FEATURE_COUNTS[@]}"; do
        for combo in "${COMBINATIONS[@]}"; do
            read RANKS_PER_NODE THREADS <<< "$combo"

            TOTAL_RANKS=$(( RANKS_PER_NODE * NODES ))
            echo "=== NODES=$NODES RANKS_PER_NODE=$RANKS_PER_NODE TOTAL_RANKS=$TOTAL_RANKS THREADS=$THREADS FEATURES=$FEATURES SIZE=$SIZE==="

            OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
                    N_JOBS=$THREADS SIZE=$SIZE FEATURES=$FEATURES \
                    mpiexec -n $TOTAL_RANKS -f 4test_hosts_sub.txt python $PYTHON_SCRIPT)
            
            TIME_TAKEN=$(echo "$OUTPUT" \
                        | grep "\[rank 0\]" \
                        | grep -oE "[0-9]+\.[0-9]+")
            
            echo "$OUTPUT"
            echo "Time taken: $TIME_TAKEN seconds"
        done
    done
done
