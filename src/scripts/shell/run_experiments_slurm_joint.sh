#!/bin/bash

##SBATCH --array=1-15
#SBATCH --array='2, 4, 8'
#SBATCH --job-name=joint
#SBATCH --output=/dev/null
#SBATCH --error=logs/arrayJob_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/joint number_of_agents=$SLURM_ARRAY_TASK_ID agent/model@model=ddqn/joint_grcn model.grcn.recursive_gradient=true,false model.grcn.action_normalization=true,false reward_clipping=true,false