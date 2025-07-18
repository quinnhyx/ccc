#!/bin/bash
#SBATCH --job-name=ccc_scaling
#SBATCH --output=1node_8gpu_%j.out
#SBATCH --error=1node_8gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gpus=8
#SBATCH --constraint=1080ti
#SBATCH --mem=370G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters 
PYTHON_SCRIPT=mpi_gpu_scaling.py
PYTHONPATH=./libs
SIZE=10000000

# Combinations
declare -a COMBINATIONS=(
    "24 1"
    "8 1"
)

# features 
declare -a FEATURE_COUNTS=(16)  
declare -a NODE_COUNTS=(1) 


# Initialize log file 
echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST > 8gpu_hosts.txt
cat 8gpu_hosts.txt

# Iterate the combination
for NODES in "${NODE_COUNTS[@]}"; do
    head -n $NODES 8gpu_hosts.txt > 8gpu_hosts_sub.txt

    for RUN_ID in {1..3}; do
        for FEATURES in "${FEATURE_COUNTS[@]}"; do
            for combo in "${COMBINATIONS[@]}"; do
                read RANKS_PER_NODE THREADS <<< "$combo"

                TOTAL_RANKS=$(( RANKS_PER_NODE * NODES ))
                echo "=== NODES=$NODES RANKS_PER_NODE=$RANKS_PER_NODE TOTAL_RANKS=$TOTAL_RANKS THREADS=$THREADS FEATURES=$FEATURES SIZE=$SIZE RUN_ID=$RUN_ID==="

                OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
                        SIZE=$SIZE FEATURES=$FEATURES \
                        mpiexec -n $TOTAL_RANKS -f 8gpu_hosts_sub.txt python $PYTHON_SCRIPT)
                
                TIME_TAKEN=$(echo "$OUTPUT" \
                            | grep "\[rank 0\]" \
                            | grep -oE "[0-9]+\.[0-9]+")
                
                echo "$OUTPUT"
		echo "Time taken:$TIME_TAKEN"
            done
        done
    done
done
