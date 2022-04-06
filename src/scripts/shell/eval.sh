#!/bin/bash
FLAGS="--job-name=eval --output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1 --error=logs/eval.err --cpus-per-task=1 --exclude=worker-6,worker-1,worker-2,worker-4,worker-5"

sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/evaluate_stored_model_hydra.py
