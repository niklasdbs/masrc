#!/bin/bash
FLAGS="--job-name=shared_ind --output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1 --error=logs/shared_ind.err --cpus-per-task=8 --exclude=worker-6,worker-1,worker-4"


#sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=grcn_shared_agent reward_clipping=tanh number_of_agents=8 number_of_environment_steps=25000000 number_of_parallel_envs=8 area=downtown
sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=grcn_shared_agent reward_clipping=tanh number_of_agents=1 number_of_environment_steps=25000000 number_of_parallel_envs=8 area=downtown add_other_agents_targets_to_resource=False shared_reward=False

#area=downtown
#slow_target_fraction=0.01 update_target_every=1