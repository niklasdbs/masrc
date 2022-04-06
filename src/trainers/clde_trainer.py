import logging
from collections import defaultdict
from typing import Any, Dict

import cv2
import numpy as np
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from agents import agents as agents_classes
from agents.agent import Agent
from envs.wrapper.parallel_env_wrapper import ToParallelWrapper
from trainers.base_trainer import BaseTrainer
from utils.logging.logger import Logger
from utils.rl.replay.clde_transition import CLDETransition


def _collate(dict: Dict) -> np.array:
    """
    creates a continuous np array out of a dictionary using the values
    @param dict:
    @return:
    """
    return np.array([x for x in dict.values()])


class CLDETrainer(BaseTrainer):
    # TODO this trainer still needs some improvements (e.g., per agent statistics, ...)

    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        self.max_eval_steps = 9999999999999999  # todo
        self.number_of_agents = config.number_of_agents
        self.number_of_actions = train_env.action_space(0).n
        self.shared_reward: bool = config.shared_reward
        agent_to_use = getattr(agents_classes, config.agent)

        self.policy_agent: Agent = agent_to_use(action_space=train_env.action_space(0),
                                                observation_space=train_env.observation_space(0),
                                                state_observation_space=train_env.state_observation_space,
                                                graph=train_env.unwrapped.graph,
                                                config=config)

    def train(self):
        self.train_env: ToParallelWrapper
        self.policy_agent.set_test(False)

        metrics = defaultdict(list)

        current_step = 0
        current_episode = 0
        t = tqdm(total=self.number_of_environment_steps)

        while current_step < self.number_of_environment_steps:
            current_episode += 1
            observations = self.train_env.reset()
            state = self.train_env.state()

            last_action_for_agents: Dict[Any, int] = {agent: 0 for agent in self.train_env.possible_agents}
            current_episode_transitions = []
            for _ in range(self.number_of_environment_steps - current_step):
                current_step += 1
                t.update()

                agent_that_acted_last = min(agent for agent in self.train_env.agents if
                                            observations[agent]["needs_to_act"] == 1)

                actions, _ = self.policy_agent.act(observations, global_state=state)
                for agent, action in actions.items():
                    last_action_for_agents[agent] = action

                next_observations, rewards, discounted_rewards, dones, infos = self.train_env.step(actions)
                self.policy_agent.after_env_step()

                if self.shared_reward:
                    # this is needed to have uniform rewards (otherwise the reward would always be from the beginning
                    # of the action of the agent)
                    joint_reward = np.float32(rewards[agent_that_acted_last])
                    joint_reward_discounted = np.float32(discounted_rewards[agent_that_acted_last])

                    rewards = {agent: joint_reward for agent in rewards.keys()}
                    discounted_rewards = {agent: joint_reward_discounted for agent in discounted_rewards.keys()}
                    time_dif = infos[agent_that_acted_last]["dt"]
                    infos = {agent: {"dt": time_dif} for agent in infos.keys()}  # ensure correct time difference

                transition = self.create_transition(dones,
                                                    infos,
                                                    last_action_for_agents,
                                                    observations,
                                                    discounted_rewards if self.semi_markov else rewards,
                                                    state)

                current_episode_transitions.append(transition)
                state = self.train_env.state()
                observations = next_observations

                if any(dones.values()):
                    self.policy_agent.reset_hidden_state()
                    self.policy_agent.buffer.add_episode(current_episode_transitions)
                    consumed_time, episode_length, score, score_discounted = self._collect_stats_for_day_clde(
                            current_episode_transitions)

                    self.writer.add_scalar(f"train/episode_length_{0}",
                                           episode_length,
                                           current_episode,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/return_{0}", score, current_episode, epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/return_discounted_{0}",
                                           score_discounted,
                                           current_episode,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/consumed_time_{0}",
                                           consumed_time,
                                           current_episode,
                                           epoch=current_episode,
                                           current_step=current_step)
                    self.writer.add_scalar(f"train/current_step_env",
                                           current_step,
                                           current_episode,
                                           epoch=current_episode,
                                           current_step=current_step)

                    if current_episode > self.start_learning and current_episode % self.train_every == 0:
                        for train_iteration in range(self.train_steps):
                            new_metrics = self.policy_agent.learn()
                            [metrics["learner/" + key].append(value) for key, value in new_metrics.items()]

                    if current_episode % self.log_metrics_every == 0:
                        for name, values in metrics.items():
                            self.writer.add_scalar(name, np.mean(np.array(values)), current_step, epoch=current_episode,
                                                   current_step=current_step)
                            metrics[name].clear()

                        for name, values in self.policy_agent.get_agent_metrics_for_logging().items():
                            self.writer.add_scalar("policy_agent/" + name,
                                                   np.mean(np.array(values)),
                                                   current_step,
                                                   epoch=current_episode,
                                                   current_step=current_step)

                        self.writer.write()

                    if current_episode % self.eval_every == 0:
                        self.policy_agent.log_weight_histogram(logger=self.writer, global_step=current_step)

                        result = self.evaluate(current_episode)
                        if self.early_stopping(result):
                            self.save_model(current_episode)
                            self.writer.write()
                            t.close()
                            return

                    if current_step % self.save_model_every == 0:
                        self.save_model(current_step)

                    break
                else:
                    pass

    def _collect_stats_for_day_clde(self, current_episode_transitions):
        episode_length = len(current_episode_transitions)
        score = sum((transition.rewards[0] for transition in current_episode_transitions))
        score_discounted = sum(
                (self.gamma ** transition.infos[0] * transition.rewards[0] for transition in
                 current_episode_transitions))
        consumed_time = sum((transition.infos[0] for transition in current_episode_transitions))
        return consumed_time, episode_length, score, score_discounted

    def create_transition(self, dones, infos, last_action_for_agents, observations, rewards, state) -> CLDETransition:
        # experience will need the following format
        # global state
        # action mask
        # action
        # local obs
        # padding_mask (will be added by collate)

        local_obs = [obs["observation"] for obs in observations.values()]
        local_obs = default_collate(local_obs)  # to create an agent dimension
        infos = np.array([info["dt"] for info in infos.values()])
        action_mask = np.ones((self.number_of_agents, self.number_of_actions), dtype=np.long)

        for agent_id, obs in observations.items():
            if obs["needs_to_act"] == 1:
                pass  # nothing to do it is already one
            else:
                action_mask[agent_id, :] = 0
                action_mask[
                    agent_id, last_action_for_agents[agent_id]] = 1  # agent can only continue with the current action

        transition = CLDETransition(
                local_observations=local_obs,
                global_observation=state,
                actions=_collate(last_action_for_agents),
                action_mask=action_mask,
                rewards=_collate(rewards),
                dones=_collate(dones),
                infos=infos
        )
        return transition

    def evaluate(self, current_step, mode="validation", env=None) -> float:
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        self.policy_agent.set_test(True)
        prev_hidden_state = self.policy_agent._hidden_state  # we need to restore the hidden state in rnn agents
        result = 0.0

        render = self.render and mode == "test"

        if not env:
            env: ToParallelWrapper = self.evaluation_env

        episode_data = {agent: defaultdict(list) for agent in env.possible_agents}

        for episode in tqdm(range(self.eval_episodes)):
            first_reset_in_episode = True  # this is used to set the flag of only doing a

            while True:
                env_reset_result = env.reset(reset_days=first_reset_in_episode, only_do_single_episode=True)
                self.policy_agent.reset_hidden_state()

                first_reset_in_episode = False

                if env_reset_result == False:
                    break

                if render:
                    video = self._create_cv_writer(env, episode)

                observations = env_reset_result
                state = env.state()

                last_action_for_agents: Dict[Any, int] = {agent: 0 for agent in env.possible_agents}
                current_episode_transitions = []

                for step in range(self.max_eval_steps):
                    actions, _ = self.policy_agent.act(observations, global_state=state)
                    for agent, action in actions.items():
                        last_action_for_agents[agent] = action

                    next_observations, rewards, discounted_rewards, dones, infos = env.step(actions)
                    transition = self.create_transition(dones,
                                                        infos,
                                                        last_action_for_agents,
                                                        observations,
                                                        rewards,
                                                        state)

                    current_episode_transitions.append(transition)
                    if render:
                        imgs = env.render(show=False)
                        for img in imgs:
                            video.write(img)

                    done = any(dones.values())
                    state = env.state()
                    observations = next_observations

                    if done:
                        break

                self._collect_stats_for_day_clde(current_episode_transitions)  # todo handle these

                if render:
                    cv2.destroyAllWindows()
                    video.release()

            # metrics = self.log_evaluation_metrics(current_step, env, episode_data, mode) #todo
            advanced_metrics = self.log_advanced_metrics(current_step, env, mode)
            result += advanced_metrics["fined_resources"]
            # result += np.sum(metrics["return_mean"])

        self.policy_agent.set_test(False)
        self.writer.write()
        self.policy_agent._hidden_state = prev_hidden_state  # we need to restore the hidden state in rnn agents

        return result / self.eval_episodes

    def save_model(self, current_step: int):
        self.policy_agent.save_model(self.save_model_path, current_step, agent_number=0)
