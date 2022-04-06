import torch
from torch import nn
from torch.distributions import Categorical

from omegaconf import DictConfig

from utils.torch.models.mlp import MLP


class MARDAM(nn.Module):

    def __init__(self, action_space, observation_space, config: DictConfig):
        super().__init__()

        self.customer_features_dim: int = observation_space[1].shape[1]
        self.number_of_customers: int = observation_space[1].shape[0]
        self.vehicle_state_size: int = observation_space[0].shape[1]

        self.number_of_actions: int = action_space.n
        self.model_dim: int = config.model_dim
        self.n_head: int = config.n_head

        self.customer_embedding = nn.Linear(self.customer_features_dim, self.model_dim)  # todo config
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim,
                                                   nhead=self.n_head,
                                                   dropout=0.0)  # todo

        self.customer_encoder = nn.TransformerEncoder(encoder_layer, **config.customer_encoder)

        self.vehicle_embedding = nn.Linear(self.vehicle_state_size, self.model_dim, bias=False)  # todo

        self.fleet_attention = nn.MultiheadAttention(self.model_dim, self.n_head, kdim=self.model_dim,
                                                     vdim=self.model_dim)
        self.vehicle_attention = nn.MultiheadAttention(self.model_dim, self.n_head)  # todo self.model_dim

        self.customers_to_action_distribution_net = MLP(input_size=self.number_of_customers,
                                                        output_size=self.number_of_actions,
                                                        hidden_size=512,
                                                        number_of_layers=2)  # todo

        # self.value_net = MLP(input_size=self.number_of_customers,
        #                      output_size=1,
        #                      hidden_size=512,
        #                      number_of_layers=1,
        #                      activation_after_last_layer=False)  # todo


    def vehicle_representation(self, vehicles: torch.Tensor,
                               vehicle_index: torch.Tensor,
                               customer_representation: torch.Tensor) -> torch.Tensor:
        """
        @param vehicles: vehicle states
        @param vehicle_index: indices of the currently acting vehicle
        """
        vehicles = self.vehicle_embedding(vehicles)

        fleet_representation, _ = self.fleet_attention.forward(query=vehicles,
                                                               key=customer_representation,
                                                               value=customer_representation,
                                                               need_weights=False)

        vehicle_query = fleet_representation.gather(0, vehicle_index.unsqueeze(2).expand(-1, -1, self.model_dim))

        vehicle_representation, _ = self.vehicle_attention.forward(query=vehicle_query,
                                                                   key=fleet_representation,
                                                                   value=fleet_representation,
                                                                   need_weights=False)
        return vehicle_representation

    def encode_customers(self, customers: torch.Tensor) -> torch.Tensor:
        customer_emb = self.customer_embedding.forward(customers)

        customer_representation = self.customer_encoder.forward(customer_emb)

        return customer_representation

    def score_customers(self, customer_representation: torch.Tensor,
                        vehicle_representation: torch.Tensor) -> torch.Tensor:
        # customer_representation: N_customers x Batch x D_customer_model_size
        # vehicle_representation: 1 x Batch x D_vehicle_model_size
        # multiplication: 1xD x DxN = 1 x N_customers = 1 x Batch x N_customers
        compact_representation = torch.bmm(vehicle_representation.permute(1, 0, 2),
                                              customer_representation.permute(1, 2, 0)).permute(1, 0, 2)
        # todo normalization and tanh exploration
        return compact_representation


    def forward(self,
                customers: torch.Tensor,
                vehicles: torch.Tensor,
                currently_acting_vehicle_index: torch.Tensor) -> torch.Tensor:
        # customers : N_cust x Batch x D_cust
        # vehicles: N_vehicles x Batch x D_vehicle
        # currently_acting_vehicle_index 1 x Batch
        customer_representation = self.encode_customers(customers)
        vehicle_representation = self.vehicle_representation(vehicles, currently_acting_vehicle_index,
                                                             customer_representation)

        logits_customer = self.score_customers(customer_representation, vehicle_representation)

        logits = self.customers_to_action_distribution_net(logits_customer)

        distribution = Categorical(logits=logits)

        action = distribution.sample()

        # log_p = distribution.log_prob(action)
        #
        # values = self.value_net(logits_customer)

        return action.view(-1).item()  # , log_p.view(-1), values.view(-1)

    def evaluate(self,
                 customers: torch.Tensor,
                 vehicles: torch.Tensor,
                 currently_acting_vehicle_index: torch.Tensor,
                 action: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        customer_representation = self.encode_customers(customers)
        vehicle_representation = self.vehicle_representation(vehicles, currently_acting_vehicle_index,
                                                             customer_representation)
        logits_customer = self.score_customers(customer_representation, vehicle_representation)

        logits = self.customers_to_action_distribution_net(logits_customer)

        distribution = Categorical(logits=logits)

        log_p = distribution.log_prob(action)

        # values = self.value_net(logits_customer)

        return log_p.view(-1), logits_customer, distribution.entropy().view(-1)
