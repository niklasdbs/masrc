import torch
from omegaconf import DictConfig
from torch import nn

from modules.graph_encoder import GraphEncoder, GraphEncoderSequential
from utils.torch.models.mlp import MLP


class AttentionQ(nn.Module):
    def __init__(self,
                 n_actions: int,
                 n_agents: int,
                 resource_dim: int,
                 config: DictConfig):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.embed_dim = config.embed_dim
        self.resource_dim = resource_dim  # number of features per node
        graph_encoder_layers = [GraphEncoder(self.embed_dim, config) for _ in range(config.number_graph_encoder_layers)]
        self.graph_encoder = GraphEncoderSequential(graph_encoder_layers,
                                                    self.resource_dim + self.n_agents,
                                                    self.embed_dim)

        self.q_net = MLP(input_size=self.embed_dim, output_size=self.n_actions, **config.q_layer)

    def forward(self, state):
        agent_id = torch.eye(self.n_agents, device=state.device)
        batch_shape = state.shape[:-2]
        state = torch.cat([state, agent_id.unsqueeze(-2).expand(list(state.shape[:-1]) + [-1])], dim=-1)
        state = state.reshape(-1, *state.shape[-2:])
        _, graph_embedding = self.graph_encoder.forward(state)

        output = self.q_net(graph_embedding)
        output = output.view(*batch_shape, -1)

        return output
