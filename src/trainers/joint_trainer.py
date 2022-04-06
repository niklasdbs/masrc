from collections import defaultdict
from typing import Dict, Any

import logging

import cv2
import numpy as np
import pathlib
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from agents import agents as agents_classes
from agents.agent import Agent
from trainers.base_trainer import BaseTrainer

from envs.wrapper.parallel_env_wrapper import ToParallelWrapper
from utils.logging.logger import Logger
from utils.rl.replay.transition import Transition

#todo remove
class JointTrainer(BaseTrainer):
    # joint option terminates when shortest option component terminates

    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        agent_to_use = getattr(agents_classes, config.agent)

        self.policy_agent: Agent = agent_to_use(action_space=train_env.action_space(0),  # todo
                                                observation_space=train_env.state_observation_space,
                                                graph=train_env.unwrapped.graph,
                                                config=config)

        if self.load_agent_model:
            self.policy_agent.load_model(pathlib.Path(config.path_to_model), step=self.load_step, agent_number=0)

        self.max_eval_steps = 999999999999999  # todo

    def train(self):
        current_episode_data = []

        observations = self.train_env.reset()
        state = self.train_env.state()

        time_last_action = self.train_env.current_time
        last_action_for_agents: Dict[Any, int] = {agent: -1 for agent in self.train_env.possible_agents}

        metrics = defaultdict(list)

        for current_step in tqdm(range(1, self.number_of_environment_steps + 1)):
            agents_that_need_to_decide = np.array([agent for agent in self.train_env.agents if
                                                   observations[agent]["needs_to_act"] == 1])
            fixed_agent_actions = {agent: last_action_for_agents[agent] for agent in self.train_env.agents if
                                   observations[agent]["needs_to_act"] == 0}

            agent_that_acted_last = np.min(agents_that_need_to_decide)
            actions, _ = self.policy_agent.act(state,
                                               agents_that_need_to_act=agents_that_need_to_decide,
                                               actions_for_agents_that_do_not_need_to_act=fixed_agent_actions)

            for agent, action in actions.items():
                last_action_for_agents[agent] = action

            next_observations, rewards, discounted_rewards, dones, infos = self.train_env.step(actions)
            next_state = self.train_env.state()

            self.policy_agent.after_env_step()

            # agent_that_finished_action = min(agent for agent in self.train_env.agents if next_observations[agent]["needs_to_act"] == 1)

            joint_reward = rewards[agent_that_acted_last]
            joint_reward_discounted = discounted_rewards[agent_that_acted_last]

            done = any(dones.values())

            time_dif = self.train_env.current_time - time_last_action  # alternative way: infos[agent_that_acted_last]["dt"]
            time_last_action = self.train_env.current_time

            transition = Transition(state,
                                    last_action_for_agents,
                                    next_state,
                                    (joint_reward_discounted if self.semi_markov else joint_reward),
                                    done,
                                    {"dt": time_dif},
                                    additional_information={agent: next_observations[agent]["needs_to_act"] for agent in
                                                            self.train_env.possible_agents})

            self.policy_agent.buffer.add_transition(transition)
            current_episode_data.append(transition)

            observations = next_observations
            state = next_state

            if current_step % self.train_every == 0 and not self.train_at_episode_end:
                for train_iteration in range(self.train_steps):
                    new_metrics = self.policy_agent.learn()
                    [metrics[key].append(value) for key, value in new_metrics.items()]

            if current_step % self.log_metrics_every == 0:
                for name, values in metrics.items():
                    self.writer.add_scalar(name, np.mean(np.array(values)), current_step)
                    metrics[name].clear()
                self.writer.write()

            if current_step % self.eval_every == 0:
                result = self.evaluate(current_step)
                if self.early_stopping(result):
                    self.save_model(current_step)
                    self.writer.write()
                    return

            if current_step % self.save_model_every == 0:
                self.save_model(current_step)

            if done:
                episode_length = len(current_episode_data)
                score = sum((transition.reward for transition in current_episode_data))
                score_discounted = sum(
                        (self.gamma ** transition.info["dt"] * transition.reward for transition in
                         current_episode_data))
                consumed_time = sum((transition.info["dt"] for transition in current_episode_data))

                self.writer.add_scalar(f"train/episode_length_{0}", episode_length, current_step)
                self.writer.add_scalar(f"train/return_{0}", score, current_step)
                self.writer.add_scalar(f"train/return_discounted_{0}", score_discounted, current_step)
                self.writer.add_scalar(f"train/consumed_time_{0}", consumed_time, current_step)

                current_episode_data.clear()

                observations = self.train_env.reset()
                state = self.train_env.state()

                time_last_action = self.train_env.current_time
                last_action_for_agents: Dict[Any, int] = {agent: -1 for agent in self.train_env.possible_agents}

    def evaluate(self, current_step, mode="validation", env: ToParallelWrapper = None) -> float:
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        self.policy_agent.set_test(True)
        result = 0.0

        render = self.render and mode == "test"

        with torch.no_grad():
            if not env:
                env: ToParallelWrapper = self.evaluation_env

            episode_data = {agent: defaultdict(list) for agent in env.possible_agents}

            for episode in tqdm(range(self.eval_episodes)):
                first_reset_in_episode = True  # this is used to set the flag of only doing a

                while True:
                    env_reset_result = env.reset(reset_days=first_reset_in_episode, only_do_single_episode=True)

                    first_reset_in_episode = False

                    if env_reset_result == False:
                        break

                    if render:
                        video = self._create_cv_writer(env, episode)

                    observations = env_reset_result
                    state = env.state()
                    current_episode_data_per_agent = {agent: [] for agent in env.possible_agents}
                    last_action_for_agents: Dict[Any, int] = {agent: -1 for agent in env.possible_agents}
                    time_last_action = env.current_time

                    for step in range(self.max_eval_steps):
                        agents_that_need_to_decide = np.array([agent for agent in env.agents if
                                                               observations[agent]["needs_to_act"] == 1])
                        fixed_agent_actions = {agent: last_action_for_agents[agent] for agent in env.agents
                                               if observations[agent]["needs_to_act"] == 0}

                        agent_that_acted_last = np.min(agents_that_need_to_decide)
                        actions, _ = self.policy_agent.act(state,
                                                           agents_that_need_to_act=agents_that_need_to_decide,
                                                           actions_for_agents_that_do_not_need_to_act=fixed_agent_actions)

                        for agent, action in actions.items():
                            last_action_for_agents[agent] = action

                        next_observations, rewards, discounted_rewards, dones, infos = env.step(actions)
                        next_state = env.state()

                        if render:
                            imgs = env.render(show=False)
                            for img in imgs:
                                video.write(img)

                        joint_reward = rewards[agent_that_acted_last]
                        # joint_reward_discounted = discounted_rewards[agent_that_acted_last]
                        done = any(dones.values())

                        time_dif = env.current_time - time_last_action

                        time_last_action = env.current_time

                        transition = Transition(state,
                                                last_action_for_agents,
                                                next_state,
                                                joint_reward,
                                                done,
                                                {"dt": time_dif},
                                                additional_information={agent: next_observations[agent]["needs_to_act"]
                                                                        for agent in env.possible_agents})
                        current_episode_data_per_agent[0].append(transition)

                        observations = next_observations
                        state = next_state

                        if done:
                            break

                    self._collect_stats_for_day(current_episode_data_per_agent, episode_data)

                    if render:
                        cv2.destroyAllWindows()
                        video.release()

            metrics = self.log_evaluation_metrics(current_step, env, episode_data, mode)
            result += np.sum(metrics["return_mean"])

        self.policy_agent.set_test(False)
        self.writer.write()

        return result / self.eval_episodes

    def save_model(self, current_step: int):
        self.policy_agent.save_model(self.save_model_path, current_step, agent_number=0)
