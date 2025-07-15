#!/bin/bash
#SBATCH --job-name=ccc_scaling
#SBATCH --output=24task_1gpu_%j.out
#SBATCH --error=24task_1gpu_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH --gpus=8
#SBATCH --constraint=1080ti
#SBATCH --mem=370G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive

set -e
export CUDA_VISIBLE_DEVICES=0

# Parameters 
PYTHON_SCRIPT=mpi_gpu_scaling.py
PYTHONPATH=./libs
LOGFILE="1gpu.log"
SIZE=10000000

# Combinations
declare -a COMBINATIONS=(
    "24 1"
    "8 3"
    "4 6"
    "12 2"
    "6 4"
    "2 12"
    "3 8"
    "1 24"
)

# features 
declare -a FEATURE_COUNTS=(2 4 8 16)  
declare -a NODE_COUNTS=(8 4 2 1) 


# Initialize log file 
echo "NODES RANKS_PER_NODE THREADS_PER_RANK SIZE FEATURES RUN_ID TIME(s)" > $LOGFILE

echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST > 1gpu_hosts.txt
cat 1gpu_hosts.txt

# Iterate the combination
for NODES in "${NODE_COUNTS[@]}"; do
    head -n $NODES 1gpu_hosts.txt > 1gpu_hosts_sub.txt

    for RUN_ID in {1..3}; do
        for FEATURES in "${FEATURE_COUNTS[@]}"; do
            for combo in "${COMBINATIONS[@]}"; do
                read RANKS_PER_NODE THREADS <<< "$combo"

                TOTAL_RANKS=$(( RANKS_PER_NODE * NODES ))
                echo "=== NODES=$NODES RANKS_PER_NODE=$RANKS_PER_NODE TOTAL_RANKS=$TOTAL_RANKS THREADS=$THREADS FEATURES=$FEATURES SIZE=$SIZE RUN_ID=$RUN_ID==="

                OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
                        N_JOBS=$THREADS SIZE=$SIZE FEATURES=$FEATURES \
                        mpiexec -n $TOTAL_RANKS -f 1gpu_hosts_sub.txt python $PYTHON_SCRIPT)
                
                TIME_TAKEN=$(echo "$OUTPUT" \
                            | grep "\[rank 0\]" \
                            | grep -oE "[0-9]+\.[0-9]+")
                
                echo "$OUTPUT"
                echo "$NODES $RANKS_PER_NODE $THREADS $SIZE $FEATURES $RUN_ID $TIME_TAKEN" \
                    | tee -a $LOGFILE
            
            done
        done
    done
done
