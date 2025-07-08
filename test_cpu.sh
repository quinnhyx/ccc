#!/bin/bash

#SBATCH --partition=gpu-preempt                # Use the same partition where gypsum-gpu106 belongs

#SBATCH --nodelist=gpu[013-021,022-024]       # Force it to run on this node

#SBATCH --gres=gpu:a100:1

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=112

#SBATCH --time=08:00:00

#SBATCH --mem=2010g

#SBATCH --job-name=ccc_cpu

#SBATCH --output=cpu_coef_%j.out


export CUDA_VISIBLE_DEVICE=""
python test_cpu_ccc.py
