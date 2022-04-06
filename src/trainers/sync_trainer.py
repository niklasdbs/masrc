from collections import defaultdict

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from trainers import SequentialTrainer
from utils.logging.logger import Logger
from utils.rl.replay.transition import Transition


class SyncTrainer(SequentialTrainer):

    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        assert config.create_observation_between_steps
        #assert not config.shared_reward


    def train(self):
        for agent in self.policy_agents.values():
            agent.set_test(False)

        metrics = defaultdict(list)

        current_step = 0
        current_episode = 1  # todo support the use of multiple rollouts
        t = tqdm(total=self.number_of_environment_steps)
        while current_step < self.number_of_environment_steps:
            current_episode_data_per_agent = {agent: [] for agent in
                                              self.train_env.possible_agents}

            _ = self.train_env.reset()

            agent_that_acted_last = 0
            previous_observation = None
            previous_action = None
            previous_info = None
            previous_actions_for_agent = {agent: None for agent in self.train_env.possible_agents}
            previous_additional_information_for_agent = {agent: None for agent in self.train_env.possible_agents}

            for agent_id in self.train_env.agent_iter(max_iter=self.number_of_environment_steps - current_step):
                t.update()
                agent = self.policy_agents[agent_id]
                observation, _, _, done, info = self.train_env.last()  # use the output from last, because a step may advance multiple agents
                #do not use rewards from last!
                reward_before =  self.train_env.rewards[agent_that_acted_last]
                discounted_reward_before = self.train_env.discounted_rewards[agent_that_acted_last]

                action, additional_information = agent.act(observation)
                step_advanced_env = self.train_env.step(action)

                # reward_after =  self.train_env.rewards[agent_that_acted_last]
                # discounted_reward_after = self.train_env.discounted_rewards[agent_that_acted_last]

                agent.after_env_step(n_steps=1)  # this increases the step of the agent

                if step_advanced_env:
                    current_step += 1

                logging_step = current_episode if self.use_episode_for_logging else current_step

                if previous_observation is not None:
                    next_observation = observation
                    transition = Transition(previous_observation,
                                            previous_action,
                                            next_observation,
                                            discounted_reward_before if self.semi_markov else reward_before,
                                            done,
                                            previous_info,
                                            previous_additional_information_for_agent[agent_that_acted_last])

                    if not self.replay_whole_episodes:
                        agent.buffer.add_transition(transition)

                    current_episode_data_per_agent[agent_id].append(transition)

                previous_observation = observation
                previous_action = action
                previous_info = info
                previous_actions_for_agent[agent_id] = action
                previous_additional_information_for_agent[agent_id] = additional_information
                agent_that_acted_last = agent_id

                should_train = \
                    (current_step % self.train_every == 0 and step_advanced_env) \
                        if self.shared_agent else \
                        (agent.current_step % self.train_every == 0)

                if should_train and current_step > self.start_learning and not self.train_at_episode_end:
                    for train_iteration in range(self.train_steps):
                        new_metrics = agent.learn()
                        if self.shared_agent:
                            [metrics["learner/" + key].append(value) for key, value in new_metrics.items()]
                        else:
                            [metrics[f"learner_{agent_id}/{key}"].append(value) for key, value in new_metrics.items()]

                if self.use_episode_for_logging:
                    # we only want to log at the end of the episode in this case
                    # so we check done and not self.train_env.agents (i.e., all agents are done)
                    log_eval_save_allowed = done and not self.train_env.agents
                else:
                    log_eval_save_allowed = step_advanced_env

                if logging_step % self.log_metrics_every == 0 and log_eval_save_allowed:
                    for name, values in metrics.items():
                        self.writer.add_scalar(name,
                                               np.mean(np.array(values)),
                                               current_step,
                                               epoch=current_episode,
                                               current_step=current_step)
                        metrics[name].clear()
                    self.writer.write()

                if logging_step % self.log_metrics_every == 0:
                    for name, values in agent.get_agent_metrics_for_logging().items():
                        self.writer.add_scalar(f"policy_agent_{agent_id}/" + name,
                                               np.mean(np.array(values)),
                                               current_step,
                                               epoch=current_episode,
                                               current_step=current_step
                                               )

                if logging_step % self.eval_every == 0 and log_eval_save_allowed:
                    if self.shared_agent:
                        agent.log_weight_histogram(logger=self.writer, global_step=current_step)
                    else:
                        for i, a in enumerate(self.policy_agents):
                            a.log_weight_histogram(logger=self.writer, global_step=current_step, prefix=str(i))

                    result = self.evaluate(logging_step)
                    if self.early_stopping(result):
                        self.save_model(logging_step)
                        self.writer.write()
                        t.close()
                        return

                if logging_step % self.save_model_every == 0 and log_eval_save_allowed:
                    self.save_model(logging_step)

                if done:
                    if self.replay_whole_episodes:
                        agent.buffer.add_episode(current_episode_data_per_agent[agent_id])

                    step_metric_for_training = current_episode if self.replay_whole_episodes else current_step

                    if self.train_at_episode_end \
                            and step_metric_for_training >= self.start_learning \
                            and current_episode % self.train_every == 0:
                        for train_iteration in range(self.train_steps):
                            new_metrics = agent.learn()
                            if self.shared_agent:
                                [metrics["learner/" + key].append(value) for key, value in new_metrics.items()]
                            else:
                                [metrics[f"learner_{agent_id}/{key}"].append(value) for key, value in
                                 new_metrics.items()]

                    episode_length = len(current_episode_data_per_agent[agent_id])
                    score = sum((transition.reward for transition in current_episode_data_per_agent[agent_id]))
                    score_discounted = sum(
                            (self.gamma ** transition.info["dt"] * transition.reward for transition in
                             current_episode_data_per_agent[agent_id]))
                    consumed_time = sum((transition.info["dt"] for transition in
                                         current_episode_data_per_agent[agent_id]))

                    self.writer.add_scalar(f"train/episode_length_{agent_id}",
                                           episode_length,
                                           logging_step,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/return_{agent_id}",
                                           score,
                                           logging_step,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/return_discounted_{agent_id}",
                                           score_discounted,
                                           logging_step,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/consumed_time_{agent_id}",
                                           consumed_time,
                                           logging_step,
                                           epoch=current_episode,
                                           current_step=current_step)

                    current_episode_data_per_agent[agent_id] = []
                else:
                    pass

            current_episode += 1
        t.close()
        self.save_model(current_step)


