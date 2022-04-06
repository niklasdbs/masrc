#!/bin/bash
set +o braceexpand -o noglob +o histexpand
PYTHONOPTIMIZE=1
#/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=ddqn/ddqn 'number_of_agents=range(1,15)' 'add_other_agents_targets_to_resource=choice(true,false)'
/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=greedy/greedy_with_wait 'number_of_agents=range(1,15)'
/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=greedy/greedy_without_wait 'number_of_agents=range(1,15)'
/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=random/random 'number_of_agents=range(1,15)'
/home/wiss/strauss/mtop_python_env/python -u  /home/wiss/strauss/projects/mtop/src/main.py -m +experiment=mardam/mardam 'number_of_agents=range(1,15)'