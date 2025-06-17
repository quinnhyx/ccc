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


# Run your Python script

python test.py
