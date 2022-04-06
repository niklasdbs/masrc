#!/bin/bash
FLAGS="--job-name=twin_grcn --output=/dev/null --time=150:00:00 --ntasks=1 --gres=gpu:1 --error=logs/twin_grcn_at_trans.err --cpus-per-task=16"


sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=twin_grcn_at agent/model@model=grcn_twin_att_trans model_optimizer.lr=0.0001 model.grcn.normalization_type=sum
#slow_target_fraction=0.01 update_target_every=1
#add_other_agents_targets_to_resource=True add_x_y_position_of_resource=True