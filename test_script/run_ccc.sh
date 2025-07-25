#!/bin/bash
#SBATCH --partition=gpu                # Use the same partition where gypsum-gpu106 belongs
#SBATCH --nodelist=gypsum-gpu[106,108-115,118-119,121,124]       # Force it to run on this node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1080_ti:8           # Request 8 GTX 1080 Ti
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --mem=370g
#SBATCH --job-name=ccc_8gpu
#SBATCH --output=ccc_8gpu_%j.out
#SBATCH --error=ccc_8gpu_%j.err
#SBATCH --exclusive


PYTHON_SCRIPT=test_gpu_ccc.py

LOGFILE="test_8gpu_ccc_scaling.log"



export NODES=1

export GPUS_USED=8

# export CUDA_VISIBLE_DEVICES=0,1  # <<< This enforces 4 visible GPUs


echo "NODES GPUS_USED SIZE FEATURES TIME(s)" > logs/$LOGFILE



for FEATURES in {2..20..2}; do

  for SIZE in 10000 100000 1000000 10000000; do

    export SIZE

    export FEATURES



    echo "Running CCC with SIZE=$SIZE, FEATURES=$FEATURES..."

    OUTPUT=$(python $PYTHON_SCRIPT)



    echo "$OUTPUT" | tee -a logs/$LOGFILE

  done

done
