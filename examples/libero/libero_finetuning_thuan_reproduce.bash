#!/bin/bash
#SBATCH --job-name=openpi_finetuning_on_libero_thuan_reproduce
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-gpu=12
#SBATCH --nodes=1
#SBATCH --exclude=voltron,ig-88

# Adapted from libero_train.bash for finetuning reproduction with base folder setup from libero_eval_multiple_task_suites.bash

# Instructions: Make sure to run from `openpi` as current folder.
# IMPORTANT: Before running this script, you must complete two prerequisite steps (see main README):
#   1. Run convert_libero_data_to_lerobot.py to convert your LIBERO data
#   2. Run compute_norm_stats.py to compute normalization statistics
# Run (change paths and api key to your liking):
# sbatch --export=base_folder=examples/libero/finetuning_thuan_reproduce,wandb_api_key=your_wandb_api_key_here,hf_lerobot_home=/path/to/your/libero_to_lerobot/cache examples/libero/libero_finetuning_thuan_reproduce.bash

# Activate the virtual environment
source examples/libero/.venv/bin/activate

base_folder=${base_folder:-examples/libero/finetuning_folder}
mkdir -p "${base_folder}"

# Set Wandb API key if provided
if [ -n "$wandb_api_key" ]; then
    export WANDB_API_KEY="$wandb_api_key"
fi

# Set HF_LEROBOT_HOME to the folder where you did the libero to lerobot conversion
if [ -z "$hf_lerobot_home" ]; then
    echo "ERROR: hf_lerobot_home must be provided via sbatch --export"
    echo "This should be the path where you ran convert_libero_data_to_lerobot.py and compute_norm_stats.py"
    exit 1
fi
export HF_LEROBOT_HOME="$hf_lerobot_home"

exec > "$base_folder/stdout.log" 2> "$base_folder/stderr.log"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero_reproduce --exp-name=pi_fast_libero_reprod --overwrite

