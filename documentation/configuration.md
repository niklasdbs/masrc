# Configuration

## Misc

### experiment_name


## Environment

### speed_in_kmh

### number_of_agents

### gamma
discount factor

### semi_markov
using temporal (semi-markov) discounting

### parallel_env
Set this flag to true so that the environment will not be sequential but instead parallel

### shared_reward
Using shared or individual rewards

### create_observation_between_steps
Create observations for agents that do not need to act when other agents need to act. 
Useful in the parallel setting.


## Observations
There observation and state_observation. 
The former are observations for individual agents the later are global observations of the env, 
this is necessary for certain algorithms.


### NoObservation
Will create None as an observation

### ZeroObservation
Will create a zero array of shape (1,1)

### FullObservationDDQN
Based only on resource encodings

### FullObservationAttention
agent_observations, resource_observations, other_agent_resource_observations, distance_observations, current_agent_index

### FullObservationGRCNTwin
resource_observations
other_agent_resource_observations: spaces.Box(-1, 2, shape=(number_of_agents-1, *self.resource_encoder.observation_space.shape)),
distance_to_action: spaces.Box(0, 1, shape=(1,)),
current_agent_id: spaces.Discrete(number_of_agents),
current_agent_observations: spaces.Box(0, 1, shape=(self.agent_observation_feature_size,)),
other_agent_observations: spaces.Box(0,1,shape=(number_of_agents-1, self.agent_observation_feature_size)),


## Trainer
### SequentialTrainer
Agents act sequentially

### CLDETrainer
Trainer for clde setting. In this setting agents will act in parallel.

### ParallelSequentialTrainer
Trainer that uses multiple processes to step in multiple envs in the same time. Agents still act sequentially

### Common Parameters
#### batch_size
#### replay_whole_episodes
#### on_policy_replay
#### train_every
Train every n steps. If train_at_episode_end and replay_whole_episodes than a step is an episode. 
In CLDE Trainer it is always episode.
#### train steps
#### start_learning


## Agents
### DDQN
### Actor Critic
### MARDAM
### QMIX
### COMA
### Greedy
### LERK

## Models

## Logging

## Misc
### Areas