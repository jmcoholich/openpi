#!/bin/bash
#SBATCH --job-name=libero_training
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-gpu=12
#SBATCH --nodes=1
#SBATCH --exclude=voltron
#SBATCH --output=libero_training.out
#SBATCH --error=libero_training.err

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero_reproduce --exp-name=pi_fast_libero_reprod --overwrite