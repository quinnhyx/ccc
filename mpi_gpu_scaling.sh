#!/bin/bash
#SBATCH --job-name=ccc_scaling
#SBATCH --output=ccc_scaling_%j.out
#SBATCH --error=ccc_scaling_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=gypsum-gpu[160-164,166,168,171,173-177,181,190-192]
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:8    
#SBATCH --mem=370G
#SBATCH --constraint=mpi
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive

set -e

# Parameters 
PYTHON_SCRIPT=mpi_gpu_scaling.py
PYTHONPATH=./libs
LOGFILE="multi_nodes_gpu.log"

SIZE=10000000

# Combinations
declare -a COMBINATIONS=(
    "24 1
    8 3"
)

# features 
declare -a FEATURE_COUNTS=(2 4 8 16)  
declare -a NODE_COUNTS=(1) 


# Initialize log file 
echo "NODES RANKS_PER_NODE THREADS_PER_RANK SIZE FEATURES RUN_ID TIME(s)" > $LOGFILE

echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST > hosts.txt
cat hosts.txt

# Iterate the combination
for NODES in "${NODE_COUNTS[@]}"; do
    head -n $NODES hosts.txt > hosts_sub.txt

    for RUN_ID in {1..3}; do
        for FEATURES in "${FEATURE_COUNTS[@]}"; do
            for combo in "${COMBINATIONS[@]}"; do
                read RANKS_PER_NODE THREADS <<< "$combo"

                TOTAL_RANKS=$(( RANKS_PER_NODE * NODES ))
                echo "=== NODES=$NODES RANKS_PER_NODE=$RANKS_PER_NODE TOTAL_RANKS=$TOTAL_RANKS THREADS=$THREADS FEATURES=$FEATURES SIZE=$SIZE RUN_ID=$RUN_ID==="

                OUTPUT=$(env PYTHONPATH=$PYTHONPATH \
                        N_JOBS=$THREADS SIZE=$SIZE FEATURES=$FEATURES \
                        mpiexec -n $TOTAL_RANKS -f hosts_sub.txt python $PYTHON_SCRIPT)
                
                TIME_TAKEN=$(echo "$OUTPUT" \
                            | grep "\[rank 0\]" \
                            | grep -oE "[0-9]+\.[0-9]+")
                
                
                echo "$NODES $RANKS_PER_NODE $THREADS $SIZE $FEATURES $RUN_ID $TIME_TAKEN" \
                    | tee -a $LOGFILE
            
            done
        done
    done
done
