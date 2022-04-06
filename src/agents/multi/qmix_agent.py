import copy
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from agents.mac_agent import MACAgent
from modules.mixer.qmix_mixer import QmixMixer
from modules.mixer.qmixer_embedded import QmixerEmbedded
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.rl.misc.reward_transformation import build_reward_transformation_fn_from_config
from utils.rl.misc.td_lambda_targets import calculate_n_step_targets
from utils.rl.replay.batch import recursive_flatten
from utils.rl.replay.complete_episode_replay_buffer import CompleteEpisodeReplayBuffer
from utils.rl.replay.episode_collate import EpisodeCollate
from utils.torch.iterable_episode_dataset import IterableEpisodeDataset


class QMIX(MACAgent):

    def __init__(self,
                 action_space,
                 observation_space,
                 graph,
                 config: DictConfig,
                 state_observation_space=None):
        super().__init__(action_space, observation_space, graph, config)
        self.semi_markov = config.semi_markov
        self.reward_transformation_fn = build_reward_transformation_fn_from_config(config)
        self.n_steps = config.n_steps
        self.update_target_every = config.update_target_every
        self.slow_target_fraction = config.slow_target_fraction
        self.current_gradient_step: int = 0
        self.max_gradient_norm = config.max_gradient_norm
        self.semi_markov = config.semi_markov
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.n_agents = config.number_of_agents
        self.n_actions = action_space.n
        self.double_q = config.double_q

        self.target_mac = copy.deepcopy(self.mac)

        if config.mixer.name == "none":
            self.mixer = None
        elif config.mixer.name in ["qmixer_mha", "qmixer_grcn"]:
            self.mixer = QmixerEmbedded(state_observation_space,
                                        number_of_actions=self.n_actions,
                                        number_of_agents=config.number_of_agents,
                                        distance_matrix=self.distance_matrix,
                                        config=config)
        else:
            self.mixer = QmixMixer(state_dim=state_observation_space.shape[0],
                                   number_of_agents=config.number_of_agents,
                                   config=config.mixer)


        self.mac.to(self.device)
        self.target_mac.to(self.device)
        self.target_mac.model.eval() #target model needs to be set to eval mode

        if self.mixer is not None:
            self.target_mixer = copy.deepcopy(self.mixer)
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)
            self.target_mixer.eval()  # target mixer needs to be set to eval mode

            self.params = list(self.mac.model.parameters()) + list(self.mixer.parameters())
        else:
            self.params = list(self.mac.model.parameters())

        self.buffer = CompleteEpisodeReplayBuffer(config)
        self.train_data_set = IterableEpisodeDataset(self.buffer,
                                                     max_sequence_length=config.max_sequence_length,
                                                     over_sample_ends=config.over_sample_ends,
                                                     seed=config.seed)

        self.train_data_loader = DataLoader(self.train_data_set,
                                            batch_size=self.batch_size,
                                            num_workers=0,
                                            pin_memory=False,
                                            collate_fn=EpisodeCollate(n_agents=config.number_of_agents))

        self.train_iterator = iter(self.train_data_loader)

        self.optimizer = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer)) \
            (self.params, **config.model_optimizer)

    def act(self, state, global_state=None):
        actions, self._hidden_state = self.mac.select_actions(state,
                                                              global_state,
                                                              test=self.test,
                                                              hidden_state=self._hidden_state)
        return actions, None

    def learn(self):
        metrics = {}

        # B = batch dim
        # T = episode length
        # N = number of agents
        # A = number of actions

        # batch shape: BxTxN
        batch = next(self.train_iterator)
        #batch["global_observation"] = batch["global_observation"].unsqueeze(2)#todo this is an ugly hack
        batch = batch.to(self.device)

        mask = batch["mask"][:, :-1].float()
        local_observations = batch["local_observations"]
        global_observation = batch["global_observation"]
        actions: torch.LongTensor = batch["actions"][:, :-1]  # shape: BxTxN
        action_mask = batch["action_masks"]
        rewards = batch["rewards"][:, :-1]
        dones = batch["dones"][:, :-1].float()
        dts = batch["infos"][:, :-1]
        mask[:, 1:] = mask[:, 1:] * (1 - dones[:, :-1])
        #  agents that can not make a decision will calcualte the q-value for the same action from the new state (action_mask takes care of this)

        rewards = self.reward_transformation_fn(rewards)

        local_observations = recursive_flatten(local_observations, start_dim=0, end_dim=1) #todo this will not work with rnn
        q_values_whole_seq = self.mac.unroll_mac(local_observations)
        q_values_whole_seq = q_values_whole_seq.unflatten(0, (batch.batch_size, batch.max_sequence_length))
        q_values = q_values_whole_seq[:, :-1]  # shape BxTxNxA

        q_values_for_chosen_actions = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # BxTxN

        with torch.no_grad():
            target_mac_out = self.target_mac.unroll_mac(local_observations)
            target_mac_out = target_mac_out.unflatten(0, (batch.batch_size, batch.max_sequence_length))
            target_q_values = target_mac_out[:, 1:]
            if self.double_q:
                best_actions = q_values_whole_seq[:, 1:].clone()
                best_actions[action_mask[:, 1:] == 0] = -np.inf
                best_actions = best_actions.argmax(dim=-1)
                target_max_q_values = target_q_values.gather(-1, best_actions.unsqueeze(-1)).squeeze(-1)
            else:
                target_q_values[action_mask[:, 1:] == 0] = -np.inf
                target_max_q_values = target_q_values.max(dim=-1)[0]

        if self.mixer is not None:
            q_values_for_chosen_actions = self.mixer.forward(q_values_for_chosen_actions,
                                                             global_observation[:, :-1])  # shape BxTx1
            with torch.no_grad():
                target_max_q_values = self.target_mixer.forward(target_max_q_values, global_observation[:, 1:])

        targets = calculate_n_step_targets(dones,
                                           dts,
                                           mask,
                                           rewards,
                                           target_max_q_values,
                                           self.gamma,
                                           self.n_steps,
                                           self.semi_markov)


        td_error = q_values_for_chosen_actions - targets

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_gradient_norm)
        self.optimizer.step()

        self.current_gradient_step += 1
        if self.current_gradient_step % self.update_target_every == 0:
            self._update_target()

        with torch.no_grad():
            n_masked_elements = max(1, mask.sum().item())
            metrics["loss"] = loss.item()
            metrics["grad_norm"] = grad_norm.item()
            metrics["td_error_abs"] = masked_td_error.abs().sum().item() / n_masked_elements
            metrics["q_taken_mean"] = (q_values_for_chosen_actions * mask).sum().item() / n_masked_elements
            metrics["target_mean"] = (targets * mask).sum().item() / n_masked_elements

        return metrics

    def _update_target(self):
        if self.slow_target_fraction == 1.0:
            self.target_mac.model.load_state_dict(self.mac.model.state_dict())
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        else:
            for param, target_param in zip(self.mac.model.parameters(), self.target_mac.model.parameters()):
                with torch.no_grad():
                    target_param.data.copy_(self.slow_target_fraction * param.data + (
                            1.0 - self.slow_target_fraction) * target_param.data)
            if self.mixer is not None:
                for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                    with torch.no_grad():
                        target_param.data.copy_(self.slow_target_fraction * param.data + (
                                1.0 - self.slow_target_fraction) * target_param.data)

    def save_model(self, path, step, agent_number):
        """Saves the current model of the agent."""
        model_path = path / f"mac_target_model_{step}_{agent_number}.pth"
        torch.save(self.target_mac.model.state_dict(), model_path)
        model_path = path / f"mac_model_{step}_{agent_number}.pth"
        torch.save(self.mac.model.state_dict(), model_path)
        if self.mixer is not None:
            model_path = path / f"mixer_target_{step}_{agent_number}.pth"
            torch.save(self.target_mixer.state_dict(), model_path)
            model_path = path / f"mixer_{step}_{agent_number}.pth"
            torch.save(self.mixer.state_dict(), model_path)

    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"mac_target_model_{step}_{agent_number}.pth"
        self.target_mac.model.load_state_dict(torch.load(model_path))
        model_path = path / f"mac_model_{step}_{agent_number}.pth"
        self.mac.model.load_state_dict(torch.load(model_path))
        if self.mixer is not None:
            model_path = path / f"mixer_target_{step}_{agent_number}.pth"
            self.target_mixer.load_state_dict(torch.load(model_path))
            model_path = path / f"mixer_{step}_{agent_number}.pth"
            self.mixer.load_state_dict(torch.load(model_path))

    def set_test(self, test: bool):
        super(QMIX, self).set_test(test)
        self.mac.model.train(not test)
        if self.mixer is not None:
            self.mixer.train(not test)
