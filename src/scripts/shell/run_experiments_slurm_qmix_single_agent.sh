#!/bin/bash
FLAGS="--job-name=qmix_sag --output=/dev/null --time=72:00:00 --ntasks=1 --gres=gpu:1 --error=logs/qmix_single_agent.err --exclude=worker-3"

#sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=attention/single_agent_attention n_steps=1 semi_markov=true  train_steps=256 max_sequence_length=2 "epsilon_decay=0.000001" batch_size=32 optimistic_in_violation=false model_optimizer.optimizer=Adam model_optimizer.lr=0.0001 special_experiment_name='at'
#sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=attention/single_agent_attention n_steps=1 semi_markov=true  train_steps=256 max_sequence_length=2 "epsilon_decay=0.000001" batch_size=32 optimistic_in_violation=false model_optimizer.optimizer=RMSprop model_optimizer.lr=0.0001 special_experiment_name='at'

#/home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=qmix/qmix number_of_agents=4 n_steps=5 agent/model@model=attention_graph_decoder agent/mixer@mixer=qmixer_mha shared_reward=true semi_markov=true  train_steps=128 max_sequence_length=5 "epsilon_decay=0.000001" start_learning=16 add_other_agents_targets_to_resource=true batch_size=16 optimistic_in_violation=false

sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=qmix/qmix agent/model@model=grcn_shared_info agent/mixer@mixer="none" observation=FullObservationAttention state_observation=ZeroObservation number_of_agents=1 n_steps=1  train_steps=64 max_sequence_length=2 update_target_every=5000 batch_size=32 optimistic_in_violation=false model_optimizer.lr=0.00001 special_experiment_name='ds'