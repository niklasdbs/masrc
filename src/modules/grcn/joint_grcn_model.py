import torch
from omegaconf import DictConfig
from torch import nn

from utils.torch.models.mlp import MLP


def _soft_arg_max(x) -> torch.Tensor:
    """
    Attention! This method will not return a long tensor, as through the conversion to long the gradient will get lost!
    @param x:
    @return:
    """
    softmax = x.softmax(dim=-1)

    indices = torch.arange(start=0, end=softmax.shape[-1], step=1, device=softmax.device, dtype=softmax.dtype)
    soft_arg_max = torch.matmul(softmax, indices)
    return soft_arg_max


class JointGraphConvolutionResourceNetwork(nn.Module):

    def __init__(self,
                 distance_matrix: torch.Tensor,
                 resource_dim: int,
                 number_of_agents: int,
                 number_of_actions:int,
                 config: DictConfig):
        super().__init__()
        self.number_of_actions = number_of_actions
        self.number_of_agents = number_of_agents
        self.resource_dim = resource_dim

        self.resource_embedding_dim = config.resource_embedding_dim

        self.distance_matrix = distance_matrix / distance_matrix.max()

        self.q_net = MLP(input_size=self.resource_embedding_dim + self.number_of_agents,
                         output_size=1,
                         **config.q_net)

        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.use_nn_scaling = config.use_nn_scaling

        if self.use_nn_scaling:
            self.scaling_net = MLP(input_size=1,
                                   output_size=1,
                                   **config.scaling_net)
        else:
            self.scaling = 1  # todo

        self.action_normalization = self.number_of_actions if config.action_normalization else 1.0#1.0 means no nomrlization, set it to the number of actions otherwise
        self.recursive_gradient = config.recursive_gradient

    def forward(self, state, agents_that_need_to_act, actions_for_agents_that_do_not_need_to_act):
        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        resource_encoding = self.resource_embedding_net(state)
        # resource_encoding is in format resources x resource_embedding_dim

        if self.use_nn_scaling:
            similarity_matrix = self.scaling_net(self.distance_matrix.unsqueeze(-1)).squeeze(-1)
        else:
            similarity_matrix = torch.exp(-self.scaling * self.distance_matrix)
        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = similarity_matrix / similarity_matrix.sum(-1).unsqueeze(-1)

        x = torch.matmul(similarity_matrix, resource_encoding)
        # x has shape edges_with_resources x resource_embedding_dim

        # we will recursively generate actions of the agents that need to act due to the otherwise combinatorial
        # explosion of the state space size

        batch_size = 1 if len(state.shape) == 2 else state.shape[0]
        if batch_size == 1:
            x = x.unsqueeze(0)

        # agents that will need to make a decision at this point in time are indicated with -1
        # ATTENTION: this will not be a long tensor, as we would lose the gradient in certain settings
        # and normalization would not work
        actions_for_agents = torch.full((self.number_of_agents, batch_size),
                                        fill_value=-1.0,
                                        device=state.device,
                                        requires_grad=self.recursive_gradient)

        with torch.no_grad():#no gradient for actions of agents that do not need to act
            for agent_id, action in actions_for_agents_that_do_not_need_to_act.items():
                actions_for_agents[agent_id] = action/self.action_normalization

        actions_for_agents = actions_for_agents.t()#shape: batch x num agents
        q_values_for_choosen_agents = torch.zeros((batch_size, self.number_of_agents), device=state.device)

        for agent_id in range(self.number_of_agents):
            if batch_size == 1:
                if agent_id not in actions_for_agents_that_do_not_need_to_act.keys():
                    continue

            q_values_for_choosen_agents[:, agent_id] = self \
                .q_net(torch.cat([x, actions_for_agents.unsqueeze(-2).repeat((1, x.shape[-2], 1))], dim=-1)) \
                .squeeze(-1) \
                .gather(-1, actions_for_agents_that_do_not_need_to_act[agent_id].view(-1, 1)) \
                .squeeze(-1)

        for agent_id in range(self.number_of_agents):
            q = self.q_net(torch.cat([x, actions_for_agents.unsqueeze(-2).repeat((1, x.shape[-2], 1))], dim=-1)) \
                .squeeze(-1) \

            soft_arg_max = _soft_arg_max(q)
            #ensure that the soft argmax is not out of range
            soft_arg_max = torch.clip(soft_arg_max, min=0, max=self.number_of_actions)

            #actions_for_agents[:, agent_id] will still have the gradient,
            # but the assign itself would modify a leave tensor inplace
            with torch.no_grad():
                actions_for_agents[:, agent_id] = soft_arg_max/self.action_normalization

            q_values_for_choosen_agents[:, agent_id] = self \
                .q_net(torch.cat([x, actions_for_agents.unsqueeze(-2).repeat((1, x.shape[-2], 1))], dim=-1)) \
                .squeeze(-1) \
                .gather(-1, soft_arg_max.long().view(-1, 1)) \
                .squeeze(-1)

        return q_values_for_choosen_agents, (actions_for_agents*self.action_normalization).long()
