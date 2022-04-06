#!/bin/bash
FLAGS="--job-name=twin_grcn --output=/dev/null --time=120:00:00 --ntasks=1 --gres=gpu:1 --error=logs/twin_grcn_ac.err --cpus-per-task=1"


sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=twin_grcn_ac agent/model@model=grcn_twin_att_big train_every=32
#slow_target_fraction=0.01 update_target_every=1