"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""

import torch
from torch import nn

from omegaconf import DictConfig
from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetwork(nn.Module):

    def __init__(self, distance_matrix, resource_dim: int, config: DictConfig):
        super().__init__()

        self.resource_dim = resource_dim

        self.resource_embedding_dim = config.resource_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.q_net = MLP(input_size=self.resource_embedding_dim,
                         output_size=1,
                         **config.q_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.use_nn_scaling = config.use_nn_scaling

        if self.use_nn_scaling:
            self.scaling_net = MLP(input_size=1,
                                   output_size=1,
                                   **config.scaling_net)
        else:
            self.scaling = 1  # todo

    def forward(self, state):
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        resource_encoding = self.resource_embedding_net(state)
        # resource_encoding is in format resources x resource_embedding_dim

        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = self.calculate_similarity_matrix()

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        q = self.q_net(x).squeeze(-1)
        # q has shape edges_with_resources

        # todo wait action

        return q

    def calculate_similarity_matrix(self):
        if self.use_nn_scaling:
            # distance matrix is in format edges_with_resources x resources
            similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

            # similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
            similarity_matrix = similarity_matrix / (similarity_matrix != 0).sum(-1, keepdim=True)

        else:
            similarity_matrix = torch.exp(-self.scaling * self.distance_matrix)

        return similarity_matrix
