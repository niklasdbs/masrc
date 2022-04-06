import math

from omegaconf import DictConfig
from torch import nn

from utils.torch.models.mlp import MLP
from utils.torch.models.skip_connection import SkipConnection


class GraphEncoder(nn.Module):
    def __init__(self, embed_dim: int, config: DictConfig):
        super().__init__()
        self.skip_connection = config.skip_connection
        self.batch_normalization = config.batch_normalization
        self.embed_dim = embed_dim
        self.mha_layers = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=config.num_heads, batch_first=True)

        if self.skip_connection:
            self.mha_layers = SkipConnectionMHA(self.mha_layers)

        if self.batch_normalization:
            self.mha_layers = BatchNormMHA(self.mha_layers, num_features=self.embed_dim)

        self.feed_forward_layers = MLP(input_size=self.embed_dim,
                                       output_size=self.embed_dim,
                                       **config.feed_forward_layer)

        if self.skip_connection:
            self.feed_forward_layers = SkipConnection(self.feed_forward_layers)

        if self.batch_normalization:
            self.feed_forward_layers = nn.Sequential(self.feed_forward_layers,
                                                     BatchNorm(self.feed_forward_layers,
                                                               num_features=self.embed_dim))


    def forward(self, x):
        h, _ = self.mha_layers.forward(query=x,
                                       key=x,
                                       value=x,
                                       need_weights=False)

        h = self.feed_forward_layers(h)

        return h


class GraphEncoderSequential(nn.Module):
    def __init__(self, layers: [GraphEncoder], input_dim: int, embed_dim: int):
        super().__init__()
        assert len(layers) > 0

        self.embed_dim = embed_dim
        self.initial_embedding = nn.Linear(in_features=input_dim, out_features=self.embed_dim)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = self.initial_embedding(x)
        for layer in self.layers:
            h = layer.forward(h)

        return h, h.mean(dim=1)


class SkipConnectionMHA(nn.Module):

    def __init__(self, module: nn.MultiheadAttention):
        super().__init__()
        self.module = module

    def forward(self, query, **kwargs):
        h, weights = self.module(query=query, **kwargs)
        return query + h, weights


class BatchNormMHA(nn.Module): #todo rename
    def __init__(self, module: nn.MultiheadAttention, num_features: int):
        super().__init__()
        self.module = module
        #self.norm = nn.BatchNorm1d(track_running_stats=True, num_features=num_features)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, *args, **kwargs):
        h, weights = self.module(*args, **kwargs)
        return self.norm(h.reshape(-1, h.size(-1))).reshape(*h.shape), weights


class BatchNorm(nn.Module): #todo rename
    def __init__(self, module: nn.Module, num_features: int):
        super().__init__()
        self.module = module
        #self.norm = nn.BatchNorm1d(track_running_stats=True, num_features=num_features)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, *args, **kwargs):
        h = self.module(*args, **kwargs)
        return self.norm(h.reshape(-1, h.size(-1))).reshape(*h.shape)
