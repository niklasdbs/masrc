#!/bin/bash

###SBATCH --array=1-15
##SBATCH --array='2,4,6, 8'
##SBATCH --job-name=coma
##SBATCH --output=/dev/null
##SBATCH --error=logs/arrayJob_%A_%a.err
##SBATCH --time=168:00:00
##SBATCH --ntasks=1
##SBATCH --gres=gpu:1

FLAGS="--output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1"

for td in 0.0 0.2 0.4 0.8 1.0; do
  for num_agents in 2 4 6 8; do
    for model in "grcn_shared" "grcn_rnn"; do
      JOBNAME="coma_"$model"_n_ag_"$num_agents"_td_"$td
      sbatch $FLAGS --error=logs/$JOBNAME --job-name=$JOBNAME /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=coma/coma agent/model@model=$model number_of_agents=$num_agents td_lambda=$td
    done
  done
done

