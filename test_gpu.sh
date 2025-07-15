#!/bin/bash

#SBATCH --job-name=clustermatch_parallel

#SBATCH --nodes=1

#SBATCH --gres=gpu:1080ti:8

#SBATCH --ntasks-per-node=24

#SBATCH --time=01:00:00

#SBATCH --mem=370G

#SBATCH --partition=gpu

#SBATCH --output=8gpu_test_%j.out



# Run your Python script
PYTHONPATH=./libs python test_gpu_ccc.py
