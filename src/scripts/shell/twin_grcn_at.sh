#!/bin/bash
FLAGS="--job-name=twin_grcn --output=/dev/null --time=168:00:00 --ntasks=1 --gres=gpu:1 --error=logs/twin_grcn_at.err --cpus-per-task=8 --exclude=worker-6,worker-3,worker-1,worker-4,worker-2"


#sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=twin_grcn_at agent/model@model=grcn_twin_att_big model_optimizer.lr=0.0001 model.grcn.normalization_type=softmax area=downtown replay_size=250000 number_of_environment_steps=50000000 update_target_every=2500 number_of_agents=8 add_other_agents_targets_to_resource=False add_x_y_position_of_resource=True model.grcn.use_one_hot_id=True model_optimizer.optimizer=RMSprop move_other_agents_between_edges=True semi_markov=True gamma=0.999 reward_clipping=tanh prioritized_replay=False gradient_clipping=False model.grcn.action_specific_bias=False epsilon_min=0.01 batch_size=128
sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=twin_grcn_at agent/model@model=grcn_twin_att_big model_optimizer.lr=0.0001 model.grcn.normalization_type=softmax model.grcn.action_specific_bias=False area=downtown replay_size=250000 number_of_environment_steps=50000000 update_target_every=5000 number_of_agents=8 add_other_agents_targets_to_resource=False add_x_y_position_of_resource=True model.grcn.use_one_hot_id=True model_optimizer.optimizer=RMSprop move_other_agents_between_edges=True semi_markov=True gamma=0.999 reward_clipping=tanh prioritized_replay=False batch_size=128 load_step=3200 start_learning=100000 epsilon_initial=0.01 epsilon_min=0.01 load_agent_model=True steps_till_min_epsilon=500000 eval_every=800 save_every=400 gradient_clipping=False max_gradient_norm=2 train_every=32 'path_to_model="/home/wiss/strauss/projects/mtop/src/multirun/standard/twin_grcn_at/8/downtown/2022-03-29/16-08-50/0/wandb/run-20220329_161213-1o3x52j1/../../models"'
#add_other_agents_targets_to_resource=True add_x_y_position_of_resource=True