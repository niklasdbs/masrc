"""
Module contains the mixing network for the QMIX architecture.
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from utils.torch.models.mlp import MLP


class QmixMixer(nn.Module):
    """
    The mixing network for the QMIX architecture.

    See also: https://arxiv.org/pdf/1803.11485v2.pdf
    """

    def __init__(self, state_dim :int , number_of_agents : int, config: DictConfig):
        super().__init__()
        self.state_dim = state_dim#state_observation_space.shape[0]

        self.num_agents = number_of_agents

        self.embed_dim = config.embedding_dim

        # first hypernet weights
        self.hyper_w1 = MLP(input_size=self.state_dim,
                            output_size=self.embed_dim * self.num_agents,
                            **config.hyper_w1)

        # second hypernet weights
        self.hyper_final = MLP(input_size=self.state_dim,
                               output_size=self.embed_dim,
                               **config.hyper_final)

        # state dependent bias for hidden layer
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        # state dependent bias for last layer
        self.V = MLP(self.state_dim, output_size=1, **config.V)

    def forward(self, q_values, global_state):
        # q_values has shape BxTxN
        # global_state has shape BxTxStateDim
        batch_size = q_values.size(0)

        q_values = q_values.view(-1, 1, self.num_agents)  # shape B*Tx1N
        states = global_state.reshape(-1, self.state_dim)  # shape B*TxStateDim

        # first layer
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(q_values @ w1 + b1)

        # second layer
        w_final = torch.abs(self.hyper_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # state dependent bias
        v = self.V(states).view(-1, 1, 1)

        # final q_tot calculation
        q_total = hidden @ w_final + v

        q_total = q_total.view(batch_size, -1, 1)

        return q_total
