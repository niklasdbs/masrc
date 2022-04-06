import collections
import logging
import pathlib

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from agents import agents as agents_classes
from trainers.base_trainer import BaseTrainer
from utils.logging.logger import Logger
from utils.rl.replay.replay_buffer import Transition


class SequentialTrainer(BaseTrainer):
    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        self.shared_agent = config.shared_agent
        agent_to_use = getattr(agents_classes, config.agent)

        if self.shared_agent:
            agent_model = agent_to_use(action_space=train_env.action_space(0),
                                       observation_space=train_env.observation_space(0),
                                       graph=train_env.unwrapped.graph,
                                       config=config)

            if self.load_agent_model:
                agent_model.load_model(pathlib.Path(config.path_to_model), step=self.load_step, agent_number=0)

            self.policy_agents = {agent: agent_model for agent in self.train_env.possible_agents}
        else:
            create_agent = lambda agent: agent_to_use(action_space=train_env.action_space(agent),
                                                      observation_space=train_env.observation_space(agent),
                                                      graph=train_env.unwrapped.graph,
                                                      config=config)

            self.policy_agents = {agent: create_agent(agent) for agent in self.train_env.possible_agents}

            if self.load_agent_model:
                for agent_id, agent in self.policy_agents.items():
                    agent.load_model(pathlib.Path(config.path_to_model), step=self.load_step, agent_number=agent_id)

    def train(self):
        for agent in self.policy_agents.values():
            agent.set_test(False)

        metrics = collections.defaultdict(list)

        current_step = 0
        current_episode = 1  # todo support the use of multiple rollouts
        t = tqdm(total=self.number_of_environment_steps)
        while current_step < self.number_of_environment_steps:
            current_episode_data_per_agent = {agent: [] for agent in
                                              self.train_env.possible_agents}

            _ = self.train_env.reset()

            previous_observations_for_agent = {agent: None for agent in self.train_env.possible_agents}
            previous_actions_for_agent = {agent: None for agent in self.train_env.possible_agents}
            previous_additional_information_for_agent = {agent: None for agent in self.train_env.possible_agents}

            for agent_id in self.train_env.agent_iter(max_iter=self.number_of_environment_steps - current_step):
                t.update()
                agent = self.policy_agents[agent_id]
                observation, reward, discounted_reward, done, info = self.train_env.last()  # use the output from last, because a step may advance multiple agents

                action, additional_information = agent.act(observation)
                step_advanced_env = self.train_env.step(action)
                agent.after_env_step(n_steps=1)  # this increases the step of the agent

                if step_advanced_env:
                    current_step += 1

                logging_step = current_episode if self.use_episode_for_logging else current_step

                if previous_observations_for_agent[agent_id] is not None:
                    next_observation = observation
                    transition = Transition(previous_observations_for_agent[agent_id],
                                            previous_actions_for_agent[agent_id],
                                            next_observation,
                                            discounted_reward if self.semi_markov else reward,
                                            done,
                                            info,
                                            previous_additional_information_for_agent[agent_id])

                    if not self.replay_whole_episodes:
                        agent.buffer.add_transition(transition)

                    current_episode_data_per_agent[agent_id].append(transition)

                previous_observations_for_agent[agent_id] = observation
                previous_actions_for_agent[agent_id] = action
                previous_additional_information_for_agent[agent_id] = additional_information

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

    def evaluate(self, current_step, mode="validation", env=None) -> float:
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        for agent in self.policy_agents.values():
            agent.set_test(True)
        result = 0.0

        render = self.render and mode == "test"

        with torch.no_grad():
            if not env:
                env = self.evaluation_env

            episode_data = {agent: collections.defaultdict(list) for agent in env.possible_agents}

            for episode in tqdm(range(self.eval_episodes)):
                first_reset_in_episode = True  # this is used to set the flag of only doing a

                while True:
                    env_reset_result = env.reset(reset_days=first_reset_in_episode, only_do_single_episode=True)
                    first_reset_in_episode = False

                    if env_reset_result == False:
                        break

                    if render:
                        video = self._create_cv_writer(env, episode)

                    current_episode_data_per_agent = {agent: [] for agent in env.possible_agents}

                    for agent in env.agent_iter():
                        observation, reward, discounted_reward, done, info = env.last()

                        action, additional_info = self.policy_agents[agent].act(observation, return_q_values=render)

                        step_advanced_env = env.step(action)

                        if render:
                            env.renderer.add_q_values(agent, additional_info)

                        if render and step_advanced_env:
                            imgs = env.render(show=False, additional_info=additional_info)
                            for img in imgs:
                                video.write(img)

                        transition = Transition(observation,
                                                action,
                                                observation,
                                                reward,
                                                done,
                                                info)  # todo save next observation correct
                        current_episode_data_per_agent[agent].append(transition)

                    self._collect_stats_for_day(current_episode_data_per_agent, episode_data)

                    if render:
                        cv2.destroyAllWindows()
                        video.release()

                metrics = self.log_evaluation_metrics(current_step, env, episode_data, mode)

                result += np.sum(metrics["return_mean"])

        for agent in self.policy_agents.values():
            agent.set_test(False)
        self.writer.write()

        return result / self.eval_episodes

    def save_model(self, current_step: int):
        for agent_id, agent in self.policy_agents.items():
            agent.save_model(self.save_model_path, current_step, agent_id)
