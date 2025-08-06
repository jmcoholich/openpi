#!/bin/bash
#SBATCH --job-name=openpi_libero_eval_multiple
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-gpu=12
#SBATCH --nodes=1
#SBATCH --exclude=voltron


# Adapted from Jeremiah's libero_eval.bash to loop through multiple task suites (libero_goal, libero_spatial, etc.) in one run. 

# Instructions: Make sure to run from `openpi` as current folder.
# Run (change base folder to your liking):
# sbatch --export=base_folder=examples/libero/eval_4 examples/libero/libero_eval_multiple_task_suites.bash

# Activate the virtual environment
source examples/libero/.venv/bin/activate
# export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

base_folder=${base_folder:-examples/libero/eval_folder}
mkdir -p "${base_folder}/videos"

exec > "$base_folder/stdout.log" 2> "$base_folder/stderr.log"

# Run both commands in parallel
OPENPI_DATA_HOME="/coc/testnvme/tnguyen868/openpi/.cache" uv run scripts/serve_policy.py --env LIBERO &
PID1=$!

python examples/libero/libero_eval_multiple_task_suites.py \
--task_suite_names libero_spatial libero_object libero_10 libero_90 libero_goal \
--log_file_path "$base_folder/log_file.txt" \
--video_out_path "$base_folder/videos" &
PID2=$!

# Exit if either process finishes
wait -n

# Optionally: kill the other process if one finishes
kill $PID1 $PID2 2>/dev/null