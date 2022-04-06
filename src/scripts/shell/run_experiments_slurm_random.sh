#!/bin/bash

#SBATCH --array=1-15
#SBATCH --job-name=random
#SBATCH --output=/dev/null
#SBATCH --error=logs/arrayJob_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0


/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=random/random number_of_agents=$SLURM_ARRAY_TASK_ID
