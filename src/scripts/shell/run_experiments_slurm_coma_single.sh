#!/bin/bash
###SBATCH --array=1-15
##SBATCH --array='2,4,6, 8'
#SBATCH --job-name=coma
#SBATCH --output=/dev/null
#SBATCH --error=logs/coma.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=worker-3

/home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=coma/coma agent/model@model=attention_graph_decoder number_of_agents=2 td_lambda=0.0 shared_reward=true add_other_agents_targets_to_resource=true
