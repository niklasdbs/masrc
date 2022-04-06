import gym
import torch
from omegaconf import DictConfig
from torch import nn

from modules.grcn.grcn_model import GraphConvolutionResourceNetwork
from modules.mixer.qmix_mixer import QmixMixer
from modules.graph_encoder import GraphEncoder, GraphEncoderSequential


class QmixerEmbedded(nn.Module):
    def __init__(self,
                 state_observation_space: gym.Space,
                 number_of_actions: int,
                 number_of_agents: int,
                 distance_matrix: torch.Tensor,
                 config: DictConfig):
        super().__init__()

        if config.mixer.state_embedding.name == "grcn":
            self.state_embed = GraphConvolutionResourceNetwork(distance_matrix=distance_matrix,
                                                               resource_dim=state_observation_space.shape[1],
                                                               config=config.mixer.state_embedding.grcn)
            self.state_embed_dim = number_of_actions
        elif config.mixer.state_embedding.name == "mha_graph_encoder":
            self.state_embed_dim = config.mixer.state_embedding_dim
            graph_encoder_layers = [GraphEncoder(self.state_embed_dim, config.mixer.state_embedding) for _ in
                                    range(config.mixer.state_embedding.number_graph_encoder_layers)]
            self.state_embed_net = GraphEncoderSequential(graph_encoder_layers,
                                                          input_dim=state_observation_space.shape[1],
                                                          embed_dim=self.state_embed_dim)

            self.state_embed = lambda x : self.state_embed_net(x)[1]#unpack tuple
        else:
            raise NotImplemented()

        self.qmixer = QmixMixer(state_dim=self.state_embed_dim,
                                number_of_agents=number_of_agents,
                                config=config.mixer)

    def forward(self, q_values, global_state):
        shape_of_features = global_state.shape[-2:]
        global_state = global_state.reshape(-1, *shape_of_features)
        state_embed = self.state_embed(global_state)
        # does not need to be reshaped because qmixer would reshape in the same format

        return self.qmixer(q_values, state_embed)
