#!/bin/bash

#SBATCH --array=8
#SBATCH --job-name=mardam
#SBATCH --output=/dev/null
#SBATCH --error=logs/mardam.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=worker-3,worker-4


/home/wiss/strauss/mtop_python_env/python -u -O /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=mardam/mardam number_of_agents=$SLURM_ARRAY_TASK_ID area=downtown train_every=8