"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""
import gym
import torch
from torch import nn

from omegaconf import DictConfig
from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetworkSharedInfo(nn.Module):

    def __init__(self, distance_matrix, observation_space : gym.Space, number_of_agents: int, config: DictConfig):
        super().__init__()
        self.n_agents = number_of_agents
        self.resource_dim = observation_space[1].shape[1]

        self.resource_embedding_dim = config.resource_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.q_net = MLP(input_size=self.resource_embedding_dim,
                         output_size=1,
                         **config.q_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.use_nn_scaling = config.use_nn_scaling
        assert self.use_nn_scaling

        self.scaling_net = MLP(input_size=observation_space[2].shape[-1] + self.n_agents + 1,
                               output_size=1,
                               **config.scaling_net)

        self.normalize_scaling_to_one = False#todo do not hardcode


    def forward(self, state):
        agent_observations, resource_observations, distance_observations, current_agent_index = state
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        resource_encoding = self.resource_embedding_net(resource_observations)
        # resource_encoding is in format resources x resource_embedding_dim


        similarity_matrix = self.calculate_similarity_matrix(distance_observations)

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        q = self.q_net(x).squeeze(-1)
        # q has shape edges_with_resources


        return q


    def calculate_similarity_matrix(self, distance_observations):
        """
        @param distance_observations: batch x time x agents x resources x features
        @return:
        """
        # distance matrix is in format edges_with_resources x resources
        n_actions = self.distance_matrix.size(0)
        batch_shape = distance_observations.shape[:3]

        agent_ids = torch.eye(self.n_agents, dtype=distance_observations.dtype, device=distance_observations.device)
        agent_ids = agent_ids\
            .unsqueeze(0) \
            .unsqueeze(-2)\
            .unsqueeze(-2)\
            .expand(*batch_shape, n_actions, self.distance_matrix.size(1), self.n_agents)


        distance_observations = distance_observations\
            .unsqueeze(-3)\
            .expand(*batch_shape,n_actions, *distance_observations.shape[-2:])

        distance_matrix = self.distance_matrix.expand(distance_observations.shape[:-1])

        augmented_distance_matrix = torch.cat([distance_matrix.unsqueeze(-1), agent_ids, distance_observations], dim=-1)

        similarity_matrix = self.scaling_net(augmented_distance_matrix).squeeze(-1)

        if self.normalize_scaling_to_one:
            similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
        else:
            similarity_matrix = similarity_matrix / (similarity_matrix!=0).sum(-1, keepdim=True)

        return similarity_matrix