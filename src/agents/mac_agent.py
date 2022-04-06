from abc import ABC
from typing import Dict, Any

import torch
from omegaconf import DictConfig

from agents.agent import Agent
from agents.multi.controllers.basic_mac import MAC
from modules.attention_graph_decoder import AttentionGraphDecoder
from modules.attention_q import AttentionQ
from modules.grcn.grcn_model_agent_id import GraphConvolutionResourceNetworkAgentID
from envs.utils import get_distance_matrix
from modules.grcn.grcn_shared_info_model import GraphConvolutionResourceNetworkSharedInfo
from modules.grcn.grcn_twin import GraphConvolutionResourceNetworkTwin
from modules.grcn.grcn_twin_af_atten import GraphConvolutionResourceNetworkTwinAfAttention
from modules.grcn.grcn_twin_after_dist import GraphConvolutionResourceNetworkTwinAfterDist
from modules.mardam.mardam import MARDAM
from utils.torch.models.mlp import MLP


class MACAgent(Agent, ABC):
    def __init__(self, action_space, observation_space, graph, config: DictConfig, **kwargs):
        super().__init__(action_space, observation_space, graph, config, **kwargs)
        self.distance_matrix = torch.from_numpy(get_distance_matrix(graph)).float().to(self.device)

        if config.model.name in ["grcn", "grcn_rnn"]:
            model = GraphConvolutionResourceNetworkAgentID(distance_matrix=self.distance_matrix,
                                                           observation_space=observation_space["observation"],
                                                           number_of_agents=config.number_of_agents,
                                                           config=config.model.grcn)
        elif config.model.name in ["attention"]:
            model = AttentionQ(
                    n_actions=action_space.n,
                    n_agents=config.number_of_agents,
                    resource_dim=observation_space["observation"].shape[1],
                    config=config.model

            )
        elif config.model.name in ["attention_graph_decoder"]:
            model = AttentionGraphDecoder(
                    n_actions=action_space.n,
                    n_agents=config.number_of_agents,
                    resource_dim=observation_space["observation"].shape[1],
                    distance_matrix=self.distance_matrix,
                    config=config.model)
        elif config.model.name in ["grcn_shared_info"]:
            model = GraphConvolutionResourceNetworkSharedInfo(distance_matrix=self.distance_matrix,
                                                              observation_space=observation_space["observation"],
                                                              number_of_agents=config.number_of_agents,
                                                              config=config.model.grcn)
        elif config.model.name in ["grcn_twin"]:
            model = GraphConvolutionResourceNetworkTwin(distance_matrix=self.distance_matrix,
                                                        observation_space=observation_space["observation"],
                                                        number_of_agents=config.number_of_agents,
                                                        config=config.model.grcn)
        elif config.model.name in ["grcn_twin_after_dist"]:
            model = GraphConvolutionResourceNetworkTwinAfterDist(distance_matrix=self.distance_matrix,
                                                                 observation_space=observation_space["observation"],
                                                                 number_of_agents=config.number_of_agents,
                                                                 config=config.model.grcn)
        elif config.model.name in ["grcn_twin_att"]:
            model = GraphConvolutionResourceNetworkTwinAfAttention(distance_matrix=self.distance_matrix,
                                                                    observation_space=observation_space["observation"],
                                                                    number_of_agents=config.number_of_agents,
                                                                    config=config.model.grcn)
        elif config.model.name in ["mardam"]:
            model = MARDAM(action_space, observation_space["observation"], config.model).to(device=self.device)

            self.value_net = MLP(input_size=model.number_of_customers,
                                 output_size=1,
                                 hidden_size=512,
                                 number_of_layers=1,
                                 activation_after_last_layer=False).to(device=self.device)


        else:
            # model = MLP(input_size=state_observation_space.shape[0], output_size=self.n_actions, number_of_layers=3, hidden_size=128)
            raise Exception("not implemented todo")  # todo
        self.mac = MAC(model, config)

    def after_env_step(self, n_steps: int = 1):
        super().after_env_step(n_steps)
        self.mac.after_env_step(n_steps)

    def get_agent_metrics_for_logging(self) -> Dict[str, Any]:
        return self.mac.action_selector.get_metrics_for_logging()
