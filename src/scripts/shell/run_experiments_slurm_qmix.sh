#!/bin/bash
###SBATCH --array=1-15
##SBATCH --array='2,4,6, 8'
#SBATCH --job-name=qmix_td_n
##SBATCH --output=/dev/null
##SBATCH --error=logs/arrayJob_%A_%a.err
##SBATCH --time=168:00:00
##SBATCH --ntasks=1
##SBATCH --gres=gpu:1

FLAGS="--output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1"

for nsteps in 1 5 10 20; do
  for num_agents in 2 4 6 8; do
    for model in "grcn_shared" "grcn_rnn"; do
      for mixer in "qmixer" "qmixer_large"; do
        JOBNAME="qmix"$model"_n_st_"$nsteps"_ag_"$num_agents"_mix"$mixer
        max_sequence_length=$(($nsteps+1))
        ADDITIONAL_FLAGS="batch_size=32 "
        if [ $model = "grcn_rnn" ];
        then
          batch_size=$((32/$num_agents))
          max_sequence_length=$((128/$num_agents))
          ADDITIONAL_FLAGS="batch_size="$batch_size" train_steps=4"
        fi

        sbatch $FLAGS --error=logs/$JOBNAME --job-name=$JOBNAME /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=qmix/qmix agent/model@model=$model number_of_agents=$num_agents n_steps=$nsteps agent/mixer@mixer=$mixer max_sequence_length=$max_sequence_length $ADDITIONAL_FLAGS
      done
    done
  done
done



#/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=qmix/qmix number_of_agents=$SLURM_ARRAY_TASK_ID n_steps=5,10,20