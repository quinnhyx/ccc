#!/bin/bash

#SBATCH --job-name=clustermatch_parallel

#SBATCH --nodelist=gpu[013-021,022-024]

#SBATCH --nodes=1

#SBATCH --gres=gpu:a100:1

#SBATCH --cpus-per-task=112

#SBATCH --ntasks=1

#SBATCH --time=08:00:00

#SBATCH --mem=2010g

#SBATCH --partition=gpu-preempt

#SBATCH --output=1gpu_coef_%j.out



# Run your Python script
python test_gpu_ccc.py
