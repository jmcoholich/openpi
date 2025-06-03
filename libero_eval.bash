#!/bin/bash
#SBATCH --job-name=libero_run
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --exclude=voltron
#SBATCH --output=libero_run.out
#SBATCH --error=libero_run.err

# Activate the virtual environment
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run both commands in parallel
uv run scripts/serve_policy.py --env LIBERO &
PID1=$!

python examples/libero/main.py &
PID2=$!

# Exit if either process finishes
wait -n

# Optionally: kill the other process if one finishes
kill $PID1 $PID2 2>/dev/null
