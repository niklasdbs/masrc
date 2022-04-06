"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""
import gym
import torch
from torch import nn
import torch.nn.functional as F

from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint

from utils.torch.models.mlp import MLP


class GraphConvolutionResourceNetworkTwinAfAttentionTransfer(nn.Module):

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
        # calculate similarity matrix by putting each distance individually in the network as an input
        self.distance_matrix_only_single_distance = config.get("distance_matrix_only_single_distance", False)
        self.allow_checkpointing = False  # todo do not hardcode
        self.use_agent_embedding = config.use_agent_embedding
        self.concat_agent_embedding = config.get("concat_agent_embedding", False)

        self.resource_embedding_dim = config.resource_embedding_dim
        self.agent_embedding_dim = config.agent_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.agent_embedding_net = MLP(input_size=self.agent_dim,
                                       output_size=self.agent_embedding_dim,
                                       **config.agent_embedding_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.att_resource_agent_net = nn.MultiheadAttention(embed_dim=self.resource_embedding_dim,
                                                            num_heads=8,
                                                            batch_first=True)
        self.att_per_edge_net = nn.MultiheadAttention(embed_dim=self.resource_embedding_dim,
                                                      num_heads=8,
                                                      batch_first=True)

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

            if not config.use_same_network_for_other_agents_resoruce_embeddings:
                self.resource_embedding_net_other = nn.Sequential(
                        self.resource_embedding_net_other,
                        nn.LayerNorm(self.resource_embedding_dim)
                )

        self.scaling_net = MLP(input_size=1,
                               output_size=1,
                               **config.scaling_net)

        action_embedding_net_input_dim = self.resource_embedding_dim + self.n_agents
        if self.add_distance_to_action:
            action_embedding_net_input_dim += 1

        if self.concat_agent_embedding:
            action_embedding_net_input_dim += self.agent_embedding_dim

        self.action_embedding_net = MLP(input_size=action_embedding_net_input_dim,
                                        output_size=self.resource_embedding_dim,  # todo
                                        **config.action_embedding_net)

        self.q_net = MLP(input_size=self.resource_embedding_dim + self.n_agents,
                         output_size=1,  # todo
                         **config.q_net)

        self.other_agent_after_q = config.get("other_agent_after_q", False)
        self.use_learnable_other_weight = config.get("learnable_other_weight", False)
        self.normalization_type = config.get("normalization_type", "n")

        if self.use_learnable_other_weight:
            self.other_weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        else:
            self.other_weight = 0.5

        self.use_value_head = False  # todo

        if self.use_value_head:
            self.value_reduce = MLP(input_size=action_embedding_net_input_dim,  # todo
                                    output_size=4,
                                    hidden_size=256,
                                    number_of_layers=2,
                                    activation_after_last_layer=True
                                    )

            self.value_head = MLP(input_size=4 * 166,  # todo
                                  output_size=1,
                                  hidden_size=256,
                                  number_of_layers=2,
                                  activation_after_last_layer=False)

    def encode_resources(self, resource_observations, agent_observations):
        resource_embedding = self.resource_embedding_net(resource_observations)

        if self.use_agent_embedding:
            agent_embedding = self.agent_embedding_net(agent_observations)
            emb, _ = self.att_resource_agent_net.forward(query=resource_embedding,
                                                         key=agent_embedding,
                                                         value=agent_embedding)

            return emb
        else:
            return resource_embedding

    def forward(self, state):
        resource_observations = state["resource_observations"]
        other_agent_resource_observations = state["other_agent_resource_observations"]
        current_agent_id = state["current_agent_id"]
        distance_to_action = state["distance_to_action"]
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        # resource_encoding is in format resources x resource_embedding_dim

        batch_size = current_agent_id.size(0)
        seq_length = current_agent_id.size(1)

        current_agent_resource_encoding = self.encode_resources(resource_observations.flatten(start_dim=0, end_dim=1),
                                                                state["current_agent_observations"].flatten(start_dim=0,
                                                                                                            end_dim=1).unsqueeze(
                                                                        1))

        other_agent_resource_encoding = self.encode_resources(other_agent_resource_observations.flatten(start_dim=0,
                                                                                                        end_dim=2),
                                                              state["other_agent_observations"].flatten(start_dim=0,
                                                                                                        end_dim=2).unsqueeze(
                                                                      1))

        agent_embedding = self.agent_embedding_net(state["current_agent_observations"].flatten(start_dim=0, end_dim=1))
        agent_embedding_others = self.agent_embedding_net(
                state["other_agent_observations"].flatten(start_dim=0, end_dim=2))

        if self.concat_agent_embedding:
            current_agent_resource_encoding = torch.cat([current_agent_resource_encoding,
                                                         agent_embedding
                                                        .unsqueeze(-2)
                                                        .expand(-1, current_agent_resource_encoding.size(1), -1)],
                                                        dim=-1)
            other_agent_resource_encoding = torch.cat([other_agent_resource_encoding,
                                                       agent_embedding_others
                                                      .unsqueeze(-2)
                                                      .expand(-1, current_agent_resource_encoding.size(1), -1)], dim=-1)

        distance_to_action = distance_to_action.flatten(start_dim=0, end_dim=1)
        distance_to_action_other_agents = state["distance_to_action_other_agents"].flatten(start_dim=0, end_dim=2)

        similarity_matrix_cur_agent = self.calculate_similarity_matrix(agent_embedding, distance_to_action)
        similarity_matrix_other_agent = self.calculate_similarity_matrix(agent_embedding_others,
                                                                         distance_to_action_other_agents)

        current_agent_per_edge = similarity_matrix_cur_agent @ current_agent_resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        other_agent_per_edge = similarity_matrix_other_agent @ other_agent_resource_encoding
        other_agent_per_edge = other_agent_per_edge.unflatten(dim=0,
                                                              sizes=(batch_size * seq_length, self.n_agents - 1))

        current_agent_id_one_hot = F.one_hot(current_agent_id.flatten(start_dim=0, end_dim=1), self.n_agents).unsqueeze(
                -2) \
            .expand(*current_agent_per_edge.shape[:2], self.n_agents)
        other_agent_ids = state["other_agent_ids"].flatten(start_dim=0, end_dim=1)
        other_agent_ids_one_hot = F.one_hot(other_agent_ids, self.n_agents).expand(*other_agent_per_edge.shape[:3],
                                                                                   self.n_agents)

        if self.add_distance_to_action:
            distance_to_action_other_agents = distance_to_action_other_agents.unflatten(dim=0,
                                                                                        sizes=(batch_size * seq_length,
                                                                                               self.n_agents - 1))
            input_action_emb_net_current_agent = [current_agent_per_edge, distance_to_action.unsqueeze(-1),
                                                  current_agent_id_one_hot]
            input_action_emb_net_other_agent = [other_agent_per_edge, distance_to_action_other_agents.unsqueeze(-1),
                                                other_agent_ids_one_hot]
        else:
            input_action_emb_net_current_agent = [current_agent_per_edge, current_agent_id_one_hot]
            input_action_emb_net_other_agent = [other_agent_per_edge, other_agent_ids_one_hot]

        current_agent_per_edge = self.action_embedding_net(torch.cat(input_action_emb_net_current_agent, dim=-1))
        other_agent_per_edge = self.action_embedding_net(torch.cat(input_action_emb_net_other_agent, dim=-1))
        other_agent_per_edge = other_agent_per_edge.sum(dim=1)  # todo do not hardcode

        combined_per_action, _ = self.att_per_edge_net.forward(query=current_agent_per_edge,
                                                               key=other_agent_per_edge,
                                                               value=other_agent_per_edge)

        q = self.q_net(torch.cat([combined_per_action, current_agent_id_one_hot], dim=-1)).squeeze(-1)
        q = q.unflatten(dim=0, sizes=(batch_size, seq_length))

        if self.use_value_head:
            y = self.value_reduce(combined_per_action)
            y = torch.flatten(y, start_dim=-2)
            v = self.value_head(y).squeeze(-1)
            v = v.unflatten(dim=0, sizes=(batch_size, seq_length))
            return q, v
        else:
            return q

    def calculate_similarity_matrix(self, agent_embedding, distance_to_action):
        # distance matrix is in format edges_with_resources x resources

        # dist_matrix_enriched = torch.cat([self.distance_matrix.unsqueeze(0).repeat(agent_embedding.size(0), 1, 1),
        #                                   agent_embedding.unsqueeze(-2).repeat(1, self.distance_matrix.size(-2), 1),
        #                                   distance_to_action.unsqueeze(-1)
        #                                   ], dim=-1)
        similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        if self.normalization_type == "sum":
            similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
        elif self.normalization_type == "n":
            similarity_matrix = similarity_matrix / (similarity_matrix != 0).sum(-1, keepdim=True)
        elif self.normalization_type == "softmax":
            similarity_matrix = F.softmax(similarity_matrix, dim=-1)

        return similarity_matrix
