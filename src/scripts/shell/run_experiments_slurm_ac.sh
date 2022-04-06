#!/bin/bash
FLAGS="--job-name=ac --output=/dev/null --time=120:00:00 --ntasks=1 --gres=gpu:1 --error=logs/ac.err"

sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ac/ac  
