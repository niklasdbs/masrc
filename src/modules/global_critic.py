import torch
from omegaconf import DictConfig
from torch import nn

from modules.graph_encoder import GraphEncoder, GraphEncoderSequential
from utils.torch.models.mlp import MLP


class GlobalCritic(nn.Module):
    def __init__(self,
                 graph_size: int,
                 n_actions: int,
                 n_agents: int,
                 input_dim: int,
                 node_dim: int,
                 config: DictConfig):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.embed_dim = config.embed_dim
        self.node_dim = node_dim  # number of features per node
        self.other_dim = input_dim - graph_size
        self.graph_size = graph_size  # number of nodes * node_dim
        self.number_of_nodes = int(self.graph_size / self.node_dim)
        graph_encoder_layers = [GraphEncoder(self.embed_dim, config) for _ in range(config.number_graph_encoder_layers)]
        self.graph_encoder = GraphEncoderSequential(graph_encoder_layers, self.node_dim, self.embed_dim)

        self.q_dim = self.embed_dim + self.other_dim  # graph dim + other input
        self.q_net = MLP(input_size=self.q_dim, output_size=self.n_actions, **config.q_layer)

    def forward(self, x):
        # split in graph specific part and agent specific part
        batch_shape = x.shape[:-1]
        agent_specific = x[:, :, :, self.graph_size:]
        nodes = x[:, :, :, :self.graph_size].reshape(-1, self.number_of_nodes, self.node_dim)
        _, graph_embedding = self.graph_encoder.forward(nodes)

        agent_specific = agent_specific.reshape(-1, self.other_dim)
        output = self.q_net(torch.cat([graph_embedding, agent_specific], dim=-1))
        output = output.view(*batch_shape, -1)

        return output
