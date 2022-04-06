from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate

import utils
from utils.rl.action_selectors.action_selector import ActionSelector
import torch.nn.functional as F

from utils.rl.replay.batch import recursive_to
from utils.rl.replay.episode_collate import recursive_pad_sequence


class MAC:
    # TODO add agent_id to obs so that agents can share the same model
    # TODO add last action

    def __init__(self, model: torch.nn.Module, config: DictConfig):
        self.model: torch.nn.Module = model
        self.is_rnn: bool = config.model.is_rnn
        self.truncate_history_backprop = False #todo
        self.add_agent_id = True  # todo (currently id is very ugly/hacky added in the model)
        self.action_selector: ActionSelector = getattr(utils.rl.action_selectors, config.action_selector)(config)
        self.device: torch.device = torch.device("cpu")
        self.output_probs: bool = config.mac_output_probs

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def select_actions(self,
                       state,
                       global_state: Optional = None,
                       test: bool = False,
                       hidden_state: Optional[torch.Tensor] = None) -> Tuple[Dict[Any, int], Optional[torch.Tensor]]:
        train_mode_before = self.model.training
        self.model.train(mode=False)
        observations = [obs["observation"] for agent, obs in state.items()]
        observations = recursive_pad_sequence([recursive_to(default_collate(observations), self.device)], batch_first=True) #convert to tensors and add batch dim
        #as we have an agent dim we do not need to add an additional time dim
        #observations = recursive_pad_sequence([observations]) #add time dim
        if self.is_rnn:
            if hidden_state is None:
                hidden_state = self.model.initial_hidden_state(self.device).expand(observations.size(0), -1)
            model_outputs, hidden_state = self.forward(observations, hidden_state)
        else:
            model_outputs = self.forward(observations)
        selected_actions = self.action_selector(model_outputs, test)[0]  # todo action mask

        # todo this is a bit dangerous because it makes assumptions about the number of agents and the position
        final_actions = {agent: selected_actions[agent].item() for agent, obs in state.items() if
                         obs["needs_to_act"] == 1}
        self.model.train(mode=train_mode_before)

        return final_actions, hidden_state

    def forward(self, local_observations: torch.Tensor, hidden_states: Optional[torch.Tensor] = None):
        if hidden_states is not None:
            model_output, hidden_states = self.model(local_observations, hidden_state=hidden_states)
        else:
            model_output = self.model(local_observations)

        if self.output_probs:
            model_output = F.softmax(model_output, dim=-1)

        if self.is_rnn:
            return model_output, hidden_states
        else:
            return model_output

    def unroll_mac(self, local_observations: torch.Tensor):
        return self._unroll_rnn(local_observations) if self.is_rnn else self._unroll(local_observations)

    def _unroll_rnn(self, local_observations: torch.Tensor):
        number_of_timesteps = local_observations.size(1)
        batch_size = local_observations.size(0)

        q_values = []
        hidden_state: torch.Tensor = self.model.initial_hidden_state(self.device).expand(batch_size,
                                                                                         local_observations.size(2),
                                                                                         -1)
        for t in range(number_of_timesteps):
            q, hidden_state = self.forward(local_observations[:, t], hidden_state)
            if self.truncate_history_backprop:
                hidden_state = hidden_state.detach()
            q_values.append(q)

        q_values = torch.stack(q_values, dim=1)  # stack over time dimension

        return q_values

    def _unroll(self, local_observations: torch.Tensor):
        return self.forward(local_observations)

    def after_env_step(self, n_steps: int = 1):
        self.action_selector.after_env_step(n_steps)
