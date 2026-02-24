#!/bin/bash
#SBATCH --job-name "non-canonical-training"
#SBATCH -p proxima
#SBATCH --output non-canonical-training-%x.%J.%N.out
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --mem=24GB
 
python3 scripts/train.py --config=configs/train-config.yaml
