import gym
import torch
from torch import nn
import torch.nn.functional as F

from omegaconf import DictConfig
from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetworkSharedAgent(nn.Module):

    def __init__(self, distance_matrix, observation_space: gym.Space, number_of_agents: int, config: DictConfig):
        super().__init__()
        self.add_distance_to_action = config.add_distance_to_action
        self.n_agents = number_of_agents
        self.resource_dim = observation_space["resource_observations"].shape[1]

        self.resource_embedding_dim = config.resource_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        q_net_input_dim = self.resource_embedding_dim + self.n_agents
        if self.add_distance_to_action:
            q_net_input_dim += 1

        self.q_net = MLP(input_size=q_net_input_dim,
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
        resource_observations = state["resource_observations"]
        current_agent_id = state["current_agent_id"]
        distance_to_action = state["distance_to_action"]

        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        resource_encoding = self.resource_embedding_net(resource_observations)
        # resource_encoding is in format resources x resource_embedding_dim

        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = self.calculate_similarity_matrix()

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        current_agent_id_one_hot = F.one_hot(current_agent_id, self.n_agents) \
            .unsqueeze(-2) \
            .expand(*x.shape[:3], self.n_agents)

        if self.add_distance_to_action:
            input_q_net = [x, distance_to_action.unsqueeze(-1), current_agent_id_one_hot]
        else:
            input_q_net = [x, current_agent_id_one_hot]

        x = torch.cat(input_q_net, dim=-1)
        q = self.q_net(x).squeeze(-1)
        # q has shape edges_with_resources

        return q


    def calculate_similarity_matrix(self):
        # distance matrix is in format edges_with_resources x resources
        similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        # similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
        similarity_matrix = similarity_matrix / (similarity_matrix != 0).sum(-1, keepdim=True)

        return similarity_matrix
