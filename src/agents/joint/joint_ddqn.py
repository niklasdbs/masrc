"""
Module contains functionality for a double DQN agent.
"""
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from agents.agent import Agent
from modules.grcn.joint_grcn_model import JointGraphConvolutionResourceNetwork
from envs.utils import get_distance_matrix
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.torch.iterable_rollout_dataset import IterableRolloutDataset
from utils.torch.iterable_transition_dataset import IterableTransitionDataset
from utils.torch.models.mlp import MLP


class JointDDQN(Agent):

    def __init__(self, action_space, observation_space, graph, config: DictConfig):
        super().__init__(action_space, observation_space, graph, config)
        self.batch_size = config.batch_size
        # Epsilon decay for exploration
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_decay_start = config.epsilon_decay_start

        self.gamma = config.gamma

        self.current_gradient_step = 0

        self.semi_markov = config.semi_markov

        self.sequence_length = config.sequence_length
        self.over_sample_ends = config.over_sample_ends
        self.slow_target_fraction = config.slow_target_fraction
        self.update_target_every = config.update_target_every

        self.train_data_set = \
            IterableRolloutDataset(self.buffer,
                                   sequence_length=self.sequence_length,
                                   over_sample_ends=self.over_sample_ends,
                                   seed=config.seed) \
                if self.replay_whole_episodes else \
                IterableTransitionDataset(self.buffer, seed=config.seed)

        self.train_data_loader = torch.utils.data.DataLoader(self.train_data_set,
                                                             batch_size=self.batch_size,
                                                             num_workers=0,
                                                             pin_memory=True)

        self.train_iterator = iter(self.train_data_loader)

        logging.info("CPU/GPU: torch.device is set to %s.", str(self.device))

        self.gradient_clipping = config.gradient_clipping
        self.reward_clipping = config.reward_clipping
        self.max_gradient_norm = config.max_gradient_norm

        self.distance_matrix = torch.from_numpy(get_distance_matrix(graph)) \
            .float().to(self.device)

        self.flatten = config.model.name == "mlp"  # todo

        if config.model.name == "mlp":
            action_dim: int = action_space.n
            state_dim: int = observation_space.shape[0] * observation_space.shape[1]
            self.q_net = MLP(input_size=state_dim, output_size=action_dim, **config.model.mlp)
            self.target_q_net = MLP(input_size=state_dim, output_size=action_dim, **config.model.mlp)
        elif config.model.name == "grcn":
            action_dim: int = action_space.n
            self.q_net = JointGraphConvolutionResourceNetwork(distance_matrix=self.distance_matrix,
                                                              resource_dim=observation_space.shape[1],
                                                              number_of_agents=config.number_of_agents,
                                                              number_of_actions=action_dim,
                                                              config=config.model.grcn)
            self.target_q_net = JointGraphConvolutionResourceNetwork(distance_matrix=self.distance_matrix,
                                                                     resource_dim=observation_space.shape[1],
                                                                     number_of_agents=config.number_of_agents,
                                                                     number_of_actions=action_dim,
                                                                     config=config.model.grcn)
        else:
            raise NotImplementedError

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

        self.loss_criterion = getattr(torch.nn, config.loss_function)()

        self.optimizer = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer)) \
            (self.q_net.parameters(), **config.model_optimizer)

    def act(self, state, agents_that_need_to_act=None, actions_for_agents_that_do_not_need_to_act=None):
        if not self.test and np.random.rand() < self.epsilon:
            return {agent: self.action_space.sample() for agent in agents_that_need_to_act}, None
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().to(self.device)
                if self.flatten:
                    state = state.view(-1)

                q, actions_for_agents = self.q_net(state,
                                                   agents_that_need_to_act,
                                                   {k: torch.tensor([v], dtype=torch.long, device=self.device)
                                                    for k, v in actions_for_agents_that_do_not_need_to_act.items()})
                return {agent: actions_for_agents[0, agent].item() for agent in agents_that_need_to_act}, None

    def _epsilon_decay(self):
        if self.current_step >= self.epsilon_decay_start:
            if (self.epsilon - self.epsilon_decay) >= self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    # logging.debug("Epsilon %s", self.epsilon)

    def learn(self):
        batch = next(self.train_iterator)
        metrics = {}
        observations, actions, next_observations, rewards, dones, dts, additional_info = batch
        observations = observations.to(self.device)
        actions = {k: v.to(self.device) for k, v in actions.items()}
        next_observations = next_observations.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        dts = dts.to(self.device)
        dones = dones.float()
        additional_info = {k: v.to(self.device) for k, v in additional_info.items()}

        if self.reward_clipping:
            rewards = torch.clip(rewards, min=0.0, max=1.0)  # todo refactor reward transformaions
        else:
            rewards = torch.tanh(rewards)#todo make configurable

        if self.flatten:
            observations = observations.flatten(start_dim=1)
            next_observations = next_observations.flatten(start_dim=1)

        current_q_values, _ = self.q_net(observations, [], actions)  # .gather(1, actions.view(-1, 1)).view(-1)
        current_q_values = current_q_values[:, -1]  # as all actions are given, the last q-value is the correct one

        # compute targets
        with torch.no_grad():
            next_state_values, _ = self.target_q_net(next_observations, [], additional_info)
            mask = torch.stack([must_act for agent, must_act in additional_info.items()], dim=-1)
            for i in range(mask.shape[1]):
                mask[:, i] = mask[:, i] * i

            mask = mask.argmax()

            next_state_values = next_state_values.gather(-1, mask.argmax(-1).view(-1, 1)).squeeze(-1)
            if self.semi_markov:
                target_q_values = rewards + (1 - dones) * torch.pow(self.gamma, dts) * next_state_values
            else:
                target_q_values = rewards + (1 - dones) * self.gamma * next_state_values

        loss = self.loss_criterion(current_q_values, target_q_values)
        metrics["loss"] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping:
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient_norm)

        self.optimizer.step()

        self.current_gradient_step += 1
        if self.current_gradient_step % self.update_target_every == 0:
            self.update_target()

        return metrics

    def after_env_step(self, n_steps: int = 1):
        """
        This method is called after each environment step at the end
        @return:
        """
        super().after_env_step(n_steps)
        self._epsilon_decay()

    def update_target(self):
        if self.slow_target_fraction == 1.0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                with torch.no_grad():
                    target_param.data.copy_(self.slow_target_fraction * param.data + (
                            1.0 - self.slow_target_fraction) * target_param.data)

    def save_model(self, path, step, agent_number):
        """Saves the current model of the agent."""
        model_path = path / f"ddqn_target_model_{step}_{agent_number}.pth"
        torch.save(self.target_q_net.state_dict(), model_path)
        model_path = path / f"ddqn_model_{step}_{agent_number}.pth"
        torch.save(self.q_net.state_dict(), model_path)

    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"ddqn_model_{step}_{agent_number}.pth"
        self.q_net.load_state_dict(torch.load(model_path))
        model_path = path / f"ddqn_target_model_{step}_{agent_number}.pth"
        self.target_q_net.load_state_dict(torch.load(model_path))
