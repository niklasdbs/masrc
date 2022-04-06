#!/bin/bash


FLAGS="--job-name=ddqn_add_tar --output=/dev/null --time=24:00:00 --ntasks=1 --gres=gpu:1 --error=logs/ddqn_add_tar.err"

for num_agents in 2 4; do
#  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=false optimistic_in_violation=true do_not_use_fined_status=true special_experiment_name='feature_ablations'
#  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=false optimistic_in_violation=true do_not_use_fined_status=false special_experiment_name='feature_ablations'
#  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=false optimistic_in_violation=false do_not_use_fined_status=true special_experiment_name='feature_ablations'
#  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=false optimistic_in_violation=false do_not_use_fined_status=false special_experiment_name='feature_ablations'
 # sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=true optimistic_in_violation=true do_not_use_fined_status=true special_experiment_name='feature_ablations'
#  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=true optimistic_in_violation=true do_not_use_fined_status=false special_experiment_name='feature_ablations'
  #sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=true optimistic_in_violation=false do_not_use_fined_status=true special_experiment_name='feature_ablations'
  sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u -O /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=ddqn/grcn add_other_agents_targets_to_resource=true do_not_use_fined_status=false special_experiment_name='feature_ablations'

 #sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=grcn_shared_info add_other_agents_targets_to_resource=false add_x_y_position_of_resource=true observation=FullObservationAttention special_experiment_name='id'
 #sbatch $FLAGS /home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn number_of_agents=$num_agents agent/model@model=grcn_shared_info add_other_agents_targets_to_resource=false add_x_y_position_of_resource=true observation=FullObservationAttention special_experiment_name='id'

done