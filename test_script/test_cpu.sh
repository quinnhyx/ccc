#!/bin/bash

#SBATCH --partition=gpu                # Use the same partition where gypsum-gpu106 belongs

#SBATCH --constraint=1080ti

#SBATCH --gpus=1

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --time=08:00:00

#SBATCH --mem=370g

#SBATCH --job-name=ccc_cpu

#SBATCH --output=cpu_%j.out


export CUDA_VISIBLE_DEVICE=""
PYTHONPATH=./libs python test_cpu_ccc.py
