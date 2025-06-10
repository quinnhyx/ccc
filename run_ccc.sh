#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=gpuq
#SBATCH --output=gpu_%j.out
#SBATCH --gres=gpu:1

# Run your Python script
python test_gpu_ccc.py
