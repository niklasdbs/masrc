"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""
from typing import Optional

import gym
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import GRUCell

from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetworkAgentID(nn.Module):

    def __init__(self, distance_matrix, observation_space: gym.Space, number_of_agents: int, config: DictConfig):
        super().__init__()
        self.rnn = config.rnn
        self.resource_and_scaling_agent_specific = config.resource_and_scaling_agent_specific
        self.number_of_agents = number_of_agents
        self.resource_dim = observation_space["resource_observations"].shape[1]

        self.resource_embedding_dim = config.resource_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        if self.rnn:
            self.rnn_hidden_dim = config.rnn_hidden_dim
            self.rnn_net = GRUCell(input_size=self.resource_embedding_dim + self.number_of_agents,
                                   hidden_size=self.rnn_hidden_dim)
            self.q_net = MLP(input_size=self.rnn_hidden_dim,
                             output_size=1,
                             **config.q_net)
        else:
            self.q_net = MLP(input_size=self.resource_embedding_dim + self.number_of_agents,
                             output_size=1,
                             **config.q_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim +
                                                     (self.number_of_agents
                                                      if self.resource_and_scaling_agent_specific else 0),
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.use_nn_scaling = config.use_nn_scaling

        if self.use_nn_scaling:
            self.scaling_net = MLP(input_size=1 +
                                              (self.number_of_agents
                                               if self.resource_and_scaling_agent_specific else 0),
                                   output_size=1,
                                   **config.scaling_net)
        else:
            self.scaling = 1  # todo

    def initial_hidden_state(self, device) -> torch.Tensor:
        return torch.zeros([self.rnn_hidden_dim], device=device)

    def forward(self, state, hidden_state: Optional[torch.Tensor] = None):
        assert not self.rnn or hidden_state is not None
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources

        resource_observations = state["resource_observations"]
        agent_id = torch.eye(self.number_of_agents, device=state.device)
        if self.resource_and_scaling_agent_specific:
            resource_encoding = self.resource_embedding_net(torch.cat([resource_observations,
                                                                       agent_id.unsqueeze(-2).expand(list(state.shape[
                                                                                                          :-1]) + [-1])],
                                                                      dim=-1))
        else:
            resource_encoding = self.resource_embedding_net(resource_observations)

        # resource_encoding is in format resources x resource_embedding_dim

        if self.use_nn_scaling:
            if self.resource_and_scaling_agent_specific:
                distance_matrix_expanded = self.distance_matrix \
                    .unsqueeze(-1) \
                    .unsqueeze(0) \
                    .expand(self.number_of_agents,
                            -1,
                            -1,
                            -1)

                agent_id_expanded = agent_id \
                    .unsqueeze(-2) \
                    .unsqueeze(-2) \
                    .expand([self.number_of_agents] + list(self.distance_matrix.shape) + [-1])
                similarity_matrix = self.scaling_net(torch.cat([distance_matrix_expanded, agent_id_expanded],dim=-1))\
                    .squeeze(-1)
            else:
                similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        else:
            similarity_matrix = torch.exp(-self.scaling * self.distance_matrix)

        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = similarity_matrix / similarity_matrix.sum(-1).unsqueeze(-1)

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        if self.rnn:
            if len(hidden_state.shape) != len(x.shape):
                hidden_state = hidden_state.unsqueeze(-2).expand(list(x.shape[:-1]) + [-1])
            hidden_state = self.rnn_net(torch.cat([x, agent_id.unsqueeze(-2).expand(list(x.shape[:-1]) + [-1])],
                                                  dim=-1).flatten(end_dim=-2), hidden_state.flatten(end_dim=-2))
            hidden_state = hidden_state.unflatten(0, x.shape[:-1])
            q = self.q_net(hidden_state).squeeze(-1)
        else:
            q = self.q_net(torch.cat([x, agent_id.unsqueeze(-2).expand(list(x.shape[:-1]) + [-1])], dim=-1)).squeeze(-1)
        # q has shape edges_with_resources

        # todo wait action
        if self.rnn:
            return q, hidden_state
        else:
            return q
