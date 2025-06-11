#!/bin/bash
#SBATCH --job-name=libero_training_long_horizon
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-gpu=12
#SBATCH --nodes=1
#SBATCH --exclude=voltron,ig-88
#SBATCH --output=libero_training_long_horizon.out
#SBATCH --error=libero_training_long_horizon.err

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero_reproduce_20_horizon --exp-name=double_action_horizon --overwrite