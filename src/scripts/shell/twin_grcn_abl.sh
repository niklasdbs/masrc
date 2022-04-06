#!/bin/bash
FLAGS="--job-name=twin_abl --output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1 --error=logs/twin_grcn_abl.err --cpus-per-task=8"

sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ablations/shared_reward model.grcn.normalization_type=softmax area=docklands replay_size=250000 number_of_environment_steps=25000000 update_target_every=2500 number_of_agents=4