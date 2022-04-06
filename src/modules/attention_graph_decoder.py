import math

import torch
from omegaconf import DictConfig
from torch import nn

from modules.graph_encoder import GraphEncoder, GraphEncoderSequential
from utils.torch.models.mlp import MLP


class AttentionGraphDecoder(nn.Module):
    def __init__(self,
                 n_actions: int,
                 n_agents: int,
                 resource_dim: int,
                 distance_matrix: torch.Tensor,
                 config: DictConfig):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.embed_dim = config.embed_dim
        self.resource_dim = resource_dim  # number of features per node
        graph_encoder_layers = [GraphEncoder(self.embed_dim, config) for _ in range(config.number_graph_encoder_layers)]
        self.graph_encoder = GraphEncoderSequential(graph_encoder_layers,
                                                    self.resource_dim + self.n_agents,
                                                    self.embed_dim)

        self.decoder = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=config.num_heads, batch_first=True)
        self.final_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, batch_first=True)

        self.resource_and_scaling_agent_specific = config.resource_and_scaling_agent_specific
        self.use_nn_scaling = config.use_nn_scaling
        if self.use_nn_scaling:
            self.scaling_net = MLP(input_size=1 +
                                              (self.n_agents
                                               if self.resource_and_scaling_agent_specific else 0),
                                   output_size=1,
                                   **config.scaling_net)
        else:
            self.scaling = 1  # todo

        self.per_action_embedding = MLP(input_size=self.n_actions,
                                        output_size=config.output_per_edge_embedding_dim,
                                        **config.per_action_embeding)

        self.q_head = MLP(input_size=config.output_per_edge_embedding_dim * self.n_actions,
                          output_size=self.n_actions,
                          **config.output_head)

        self.use_value_head = True #todo
        if self.use_value_head:
            self.value_head = MLP(input_size=self.embed_dim,
                                  output_size=1,
                                  hidden_size=512,
                                  number_of_layers=2)

    def forward(self, state):
        agent_id = torch.eye(self.n_agents, device=state.device)
        batch_shape = state.shape[:-2]
        state = torch.cat([state, agent_id.unsqueeze(-2).expand(list(state.shape[:-1]) + [-1])], dim=-1)
        state = state.reshape(-1, *state.shape[-2:])
        node_embeddings, graph_embedding = self.graph_encoder.forward(state)
        context = graph_embedding.unsqueeze(-2).repeat(1, state.size(-2), 1)  # shape BxRxD

        # match context with individual node embedding
        context, _ = self.decoder.forward(query=context,
                                          key=node_embeddings,
                                          value=node_embeddings,
                                          need_weights=False)

        similarity_matrix = self._calculate_similarity_matrix(agent_id)  # shape (Nx)AxR

        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = similarity_matrix / similarity_matrix.sum(-1).unsqueeze(-1)

        # calculate per resource embeddings
        context = similarity_matrix @ context.view(*batch_shape,
                                                   state.shape[-2],
                                                   -1)  # view as reconstructed batch per resource
        node_embeddings = similarity_matrix @ node_embeddings.view(*batch_shape, state.shape[-2], -1)

        context = context.view(-1, *context.shape[-2:])  # shape BxRxD
        node_embeddings = node_embeddings.view(-1, *node_embeddings.shape[-2:])

        # final_attention, _ = self.final_attention.forward(query=context,
        #                                                   key=node_embeddings,
        #                                                   value=torch.ones_like(node_embeddings),
        #                                                   need_weights=False)

        final_attention = self._final_attention(context, node_embeddings, None)
        # final_attention = self.distance_matrix @ final_attention
        final_attention = self.per_action_embedding(final_attention)
        final_attention = final_attention.flatten(-2, -1)
        q_values = self.q_head(final_attention)

        # q_values = self.q_net(final_attention) # BxAx1
        q_values = q_values.view(*batch_shape, -1)  # shape: BxTxNxA

        if self.use_value_head:
            values = self.value_head(graph_embedding).view(*batch_shape)

            return q_values, values
        else:
            return q_values

    def _final_attention(self, query, key, value):
        logits = torch.matmul(query, key.transpose(-2, -1)).squeeze(-2) / math.sqrt(query.size(-1))
        logits = torch.tanh(logits) * 10.0

        return logits

    def _calculate_similarity_matrix(self, agent_id):
        if self.use_nn_scaling:
            if self.resource_and_scaling_agent_specific:
                distance_matrix_expanded = self.distance_matrix \
                    .unsqueeze(-1) \
                    .unsqueeze(0) \
                    .expand(self.n_agents,
                            -1,
                            -1,
                            -1)

                agent_id_expanded = agent_id \
                    .unsqueeze(-2) \
                    .unsqueeze(-2) \
                    .expand([self.n_agents] + list(self.distance_matrix.shape) + [-1])
                similarity_matrix = self.scaling_net(torch.cat([distance_matrix_expanded, agent_id_expanded], dim=-1)) \
                    .squeeze(-1)
            else:
                similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        else:
            similarity_matrix = torch.exp(-self.scaling * self.distance_matrix)
        return similarity_matrix
