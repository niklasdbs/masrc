"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""
import gym
import torch
from torch import nn
import torch.nn.functional as F

from omegaconf import DictConfig
from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetworkTwin(nn.Module):

    def __init__(self, distance_matrix, observation_space: gym.Space, number_of_agents: int, config: DictConfig):
        super().__init__()
        self.n_agents = number_of_agents
        self.resource_dim = observation_space["resource_observations"].shape[1]
        self.agent_dim = observation_space["current_agent_observations"].shape[0]
        self.multiply_with_other = config.multiply_with_other
        self.use_nn_scaling = config.use_nn_scaling
        assert self.use_nn_scaling
        self.add_distance_to_action = config.add_distance_to_action
        self.layer_norm = config.layer_norm

        self.resource_embedding_dim = config.resource_embedding_dim
        self.agent_embedding_dim = config.agent_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.agent_embedding_net = MLP(input_size=self.agent_dim,
                                       output_size=self.agent_embedding_dim,
                                       **config.agent_embedding_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.final_resource_embedding_net = MLP(input_size=self.resource_embedding_dim + self.agent_embedding_dim,
                                                output_size=self.resource_embedding_dim,
                                                **config.final_resource_embedding_net)

        if config.use_same_network_for_other_agents_resoruce_embeddings:
            self.resource_embedding_net_other = self.resource_embedding_net
        else:
            self.resource_embedding_net_other = MLP(input_size=self.resource_dim,
                                                    output_size=self.resource_embedding_dim,
                                                    **config.resource_embedding_net)

        if self.layer_norm:
            self.agent_embedding_net = nn.Sequential(
                    self.agent_embedding_net,
                    nn.LayerNorm(self.agent_embedding_dim)
            )


            self.resource_embedding_net = nn.Sequential(
                    self.resource_embedding_net,
                    nn.LayerNorm(self.resource_embedding_dim)
            )

            self.final_resource_embedding_net = nn.Sequential(
                    self.final_resource_embedding_net,
                    nn.LayerNorm(self.resource_embedding_dim)
            )

            if not config.use_same_network_for_other_agents_resoruce_embeddings:
                self.resource_embedding_net_other = nn.Sequential(
                        self.resource_embedding_net_other,
                        nn.LayerNorm(self.resource_embedding_dim)
                )

        self.scaling_net = MLP(input_size=1,
                               output_size=1,
                               **config.scaling_net)

        q_net_input_dim = self.resource_embedding_dim + self.n_agents
        if self.add_distance_to_action:
            q_net_input_dim += 1

        self.q_net = MLP(input_size=q_net_input_dim,
                         output_size=1,
                         **config.q_net)

        self.use_learnable_other_weight = config.get("learnable_other_weight", False)
        if self.use_learnable_other_weight:
            self.other_weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        else:
            self.other_weight = 0.5


        self.use_value_head = False # todo

        if self.use_value_head:
            self.value_reduce = MLP(input_size=q_net_input_dim,#todo
                                  output_size=4,
                                  hidden_size=256,
                                  number_of_layers=2,
                                activation_after_last_layer=True
                                    )

            self.value_head = MLP(input_size=4* 166,#todo
                                  output_size=1,
                                  hidden_size=256,
                                  number_of_layers=2,
                                  activation_after_last_layer=False)



    def forward(self, state):
        resource_observations = state["resource_observations"]
        other_agent_resource_observations = state["other_agent_resource_observations"]
        current_agent_id = state["current_agent_id"]
        distance_to_action = state["distance_to_action"]
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        # resource_encoding is in format resources x resource_embedding_dim

        current_agent_embedding = self.agent_embedding_net(state["current_agent_observations"])
        embedding_other_agent = self.agent_embedding_net(state["other_agent_observations"])

        resource_encoding = self.resource_embedding_net(resource_observations)
        resource_encoding = self.final_resource_embedding_net(torch.cat([resource_encoding,
                                                                         current_agent_embedding.unsqueeze(-2).expand_as(
                                                                             resource_encoding)], dim=-1))
        resource_encoding_other_agents = self.resource_embedding_net_other(other_agent_resource_observations)
        resource_encoding_other_agents = self.final_resource_embedding_net(torch.cat([resource_encoding_other_agents,
                                                                                      embedding_other_agent.unsqueeze(-2).expand_as(
                                                                                          resource_encoding_other_agents)],
                                                                                     dim=-1))
        resource_encoding_other_agents = resource_encoding_other_agents.sum(dim=2)
        # resource_encoding_other_agents = torch.nn.functional.instance_norm(other_agent_encodings) + 1
        # resource_encoding_other_agents = 1/(resource_encoding_other_agents)
        if self.multiply_with_other:
            resource_encoding = torch.exp(resource_encoding) * 1 / (self.other_weight * torch.exp(resource_encoding_other_agents))
        else:
            resource_encoding = resource_encoding - (self.other_weight * resource_encoding_other_agents)
        similarity_matrix = self.calculate_similarity_matrix()

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        # y = similarity_matrix @ resource_encoding_other_agents

        # x = torch.exp(x) * 1/torch.exp(y)

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

        if self.use_value_head:
            y = self.value_reduce(x)
            y = torch.flatten(y, start_dim=-2)
            v = self.value_head(y).squeeze(-1)

            return q,v
        else:
            return q

    def calculate_similarity_matrix(self):
        # distance matrix is in format edges_with_resources x resources
        similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        # similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
        similarity_matrix = similarity_matrix / (similarity_matrix != 0).sum(-1, keepdim=True)

        return similarity_matrix
