from typing import Union

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 number_of_layers: int,
                 hidden_size: int,
                 activation : Union[torch.nn.Module, str]=nn.ELU,
                 activation_after_last_layer : Union[bool, torch.nn.Module, str] = True,
                 **kwargs):
        """
        Args:
            input_size:
            output_size:
            number_of_layers:
            hidden_size:
            activation:
            activation_after_last_layer:  specifies if and which activation function should be used after the last layer
            true = use activation as activation function , str or module to specify another activation fn
            **kwargs:
        """
        super(MLP, self).__init__()
        self.activation = getattr(torch.nn, activation) if isinstance(activation, str) else activation

        if isinstance(activation_after_last_layer, bool):
            self.activation_after_last_layer = activation_after_last_layer
            if activation_after_last_layer:
                self.activation_fn_after_last_layer = self.activation
        elif isinstance(activation_after_last_layer, str):
            self.activation_after_last_layer = True
            self.activation_fn_after_last_layer = getattr(torch.nn, activation_after_last_layer)
        else:
            self.activation_after_last_layer = True
            self.activation_fn_after_last_layer = activation_after_last_layer

        self.number_of_layers = number_of_layers
        self.size_of_layer = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self.input_size, self.size_of_layer if self.number_of_layers > 1 else self.output_size)]
        if self.number_of_layers > 1:
            model += [self.activation()]
        elif self.activation_after_last_layer:
            model += [self.activation_fn_after_last_layer()]

        for i in range(1, self.number_of_layers):
            if i == self.number_of_layers - 1:
                model += [nn.Linear(self.size_of_layer, self.output_size)]
                if self.activation_after_last_layer:
                    model += [self.activation_fn_after_last_layer()]
            else:
                model += [nn.Linear(self.size_of_layer, self.size_of_layer)]
                model += [self.activation()]

        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
