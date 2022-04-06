#!/bin/bash
FLAGS="--job-name=twin_grcn --output=/dev/null --time=72:00:00 --ntasks=1 --gres=gpu:1 --error=logs/twin_grcn.err --cpus-per-task=16"


sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=twin_grcn reward_clipping=tanh model.grcn.other_agent_after_q=False add_other_agents_targets_to_resource=True add_x_y_position_of_resource=True number_of_agents=4
#slow_target_fraction=0.01 update_target_every=1
#add_other_agents_targets_to_resource=True add_x_y_position_of_resource=True