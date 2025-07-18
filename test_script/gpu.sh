#!/bin/bash
#SBATCH --job-name=clustermatch_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu[022-024]
#SBATCH --output=8gpu_cont_%j.out
#SBATCH --gres=gpu:8

# Run your Python script
python test_gpu_ccc.py
