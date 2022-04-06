#!/bin/bash

#SBATCH --array=2,4,8
#SBATCH --job-name=greedy
#SBATCH --output=/dev/null
#SBATCH --error=logs/arrayJob_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0


/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=greedy/greedy_with_wait number_of_agents=$SLURM_ARRAY_TASK_ID area=queensberry,docklands,downtown
#/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=greedy/greedy_without_wait number_of_agents=$SLURM_ARRAY_TASK_ID area=queensberry
