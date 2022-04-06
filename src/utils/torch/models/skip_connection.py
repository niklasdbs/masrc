from torch import nn


class SkipConnection(nn.Module):

    def __init__(self, module : nn.Module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
