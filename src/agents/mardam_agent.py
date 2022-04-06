from itertools import chain
from pathlib import Path
from typing import Tuple, Union, Any

import numpy as np
import torch.distributions
from omegaconf import DictConfig

from agents.agent import Agent
from agents.mac_agent import MACAgent
from modules.mardam.mardam import MARDAM
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.rl.misc.reward_transformation import build_reward_transformation_fn_from_config
from utils.rl.replay.rollout_buffer import RolloutBuffer
from utils.torch.models.mlp import MLP
import torch.nn.functional as F


class MARDAM_Agent(Agent):
    def __init__(self, action_space, observation_space, graph, config: DictConfig):
        super().__init__(action_space, observation_space, graph, config)
        self.mardam = MARDAM(action_space, observation_space, config.model).to(device=self.device)

        #the authors use ta single layer MLP based on the embeddings of the actor
        self.value_net = MLP(input_size=self.mardam.number_of_customers,
                             output_size=1,
                             hidden_size=512,#this size does not matter as it is a single layer
                             number_of_layers=1,
                             activation_after_last_layer=False).to(device=self.device)

        self.buffer = RolloutBuffer(config)

        self.gamma = config.gamma
        self.gradient_clipping = config.gradient_clipping
        self.max_gradient_norm = config.max_gradient_norm
        self.reward_transformation_fn = build_reward_transformation_fn_from_config(config)
        self.semi_markov = config.semi_markov

        self.optimizer: torch.optim.Optimizer = getattr(torch.optim, config.model_optimizer.optimizer) \
            ([{"params": self.mardam.parameters(), "lr": 0.0001},
              {"params": self.value_net.parameters(), "lr": 0.001}])  # todo do not hardcode

    def act(self, state, **kwargs) -> (int, None):
        with torch.no_grad():
            agent_observations, resource_observations, current_agent_index = state

            agent_observations = torch.from_numpy(agent_observations).to(self.device).unsqueeze(1)
            resource_observations = torch.from_numpy(resource_observations).to(self.device).unsqueeze(1)
            current_agent_index = torch.tensor(current_agent_index,
                                               dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(1)

            action = self.mardam.forward(resource_observations, agent_observations, current_agent_index)
            return action, None

    def evaluate(self, states, actions) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # agent_observations, resource_observations, current_agent_index = state
        agent_observations_batch = []
        resource_observations_batch = []
        current_agent_index_batch = []

        for state in states:
            agent_observations, resource_observations, current_agent_index = state

            current_agent_index = current_agent_index

            agent_observations_batch.append(agent_observations)
            resource_observations_batch.append(resource_observations)
            current_agent_index_batch.append(current_agent_index)

        agent_observations_batch = torch.from_numpy(np.stack(agent_observations_batch, axis=1)).to(self.device)
        resource_observations_batch = torch.from_numpy(np.stack(resource_observations_batch, axis=1)).to(self.device)
        current_agent_index_batch = torch.tensor(current_agent_index_batch,
                                                 dtype=torch.long,
                                                 device=self.device).view(1, -1)

        log_prob, logits, entropy = self.mardam.evaluate(resource_observations_batch,
                                                         agent_observations_batch,
                                                         current_agent_index_batch,
                                                         actions)

        return log_prob, self.value_net(logits).view(-1), entropy

    def compute_returns_and_advantage_montecarlo(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float) -> (
    torch.Tensor, torch.Tensor):
        # todo move to utils
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            last_return = 0.0
            for t in reversed(range(len(rewards))):
                last_return = rewards[t] + gamma * last_return
                returns[t] = last_return

            advantages = returns - values

            return returns, advantages

    def compute_returns_and_advantage(self, rewards: torch.Tensor,
                                      values: torch.Tensor,
                                      gamma: float,
                                      dones: torch.Tensor,
                                      next_states, dts: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # todo move to utils
        with torch.no_grad():
            _, next_values, _ = self.evaluate(next_states, torch.zeros_like(rewards, dtype=torch.long))

            if self.semi_markov:
                return_ = rewards + torch.pow(gamma, dts) * (1 - dones.float()) * next_values
            else:
                return_ = rewards + gamma * (1 - dones.float()) * next_values

            advantage = return_ - values

            return return_, advantage

    def compute_returns_and_advantage_gae(self, rewards, values, dones, last_value=None):
        # todo move to utils
        # td lambda return and gae lambda advantage
        with torch.no_grad():
            if last_value is None:
                last_value = torch.zeros([0])

            next_values = torch.cat([values[1:], last_value], dim=-1)

            gae_lambda = 1.0  # todo do not hardcode

            non_terminal = 1.0 - dones.float()

            last_gae_lambda = 0.0
            advantages = torch.zeros_like(rewards)
            for t in reversed(range(len(rewards))):
                # td_error = delta = r_t + gamma * v(s_{t+1})*(1-done[t+1]) -v(s_t)
                delta = rewards[t] + self.gamma * next_values[t] * non_terminal[t + 1] - values[t]

                # advantage = delta + gamma * gae_lambda *(1-done[t+1]) * advantage[t+1])
                advantage = delta + self.gamma * gae_lambda * last_gae_lambda
                advantages[t] = advantage
                last_gae_lambda = advantage

            returns = advantages + values

            return returns, advantages

    def learn(self) -> {}:
        if len(self.buffer) == 0:
            return {}

        states, actions, rewards, dones, dts, next_states = self.buffer.get_tensor_for_on_policy(self.device)

        rewards = self.reward_transformation_fn(rewards)

        log_prob, values, entropy = self.evaluate(states, actions)

        # todo do not hardcode return function
        targets, advantages = self.compute_returns_and_advantage(rewards, values, self.gamma, dones, next_states, dts)

        actor_loss = -(advantages * log_prob).mean()

        critic_loss = F.smooth_l1_loss(targets, values)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping:
            # Clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(chain.from_iterable(group["params"] for group in self.optimizer.param_groups),
                                           self.max_gradient_norm)
            grad_norm = grad_norm.item()
        else:
            grad_norm = 0

        self.optimizer.step()
        self.buffer.clear()
        return {"loss": loss.item(),
                "entropy": entropy.view(-1).mean().item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "grad_norm": grad_norm}

    def set_test(self, test: bool):
        super().set_test(test)
        self.mardam.train(not test)

    def save_model(self, path: Path, step: int, agent_number):
        model_path = path / f"mardam_model_{step}_{agent_number}.pth"
        torch.save(self.mardam.state_dict(), model_path)
        model_path = path / f"mardam_value_net_{step}_{agent_number}.pth"
        torch.save(self.value_net.state_dict(), model_path)


    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"mardam_model_{step}_{agent_number}.pth"
        self.mardam.load_state_dict(torch.load(model_path))
        model_path = path / f"mardam_value_net_{step}_{agent_number}.pth"
        self.value_net.load_state_dict(torch.load(model_path))
