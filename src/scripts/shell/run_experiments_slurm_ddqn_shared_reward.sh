#!/bin/bash

#SBATCH --array=1-15
#SBATCH --job-name=ddqn_sr
#SBATCH --output=/dev/null
#SBATCH --error=logs/arrayJob_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBTACH --cpus-per-task=2

/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$SLURM_ARRAY_TASK_ID agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=false shared_reward=true