import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from agents.mac_agent import MACAgent
from modules.global_critic import GlobalCritic
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.rl.misc.reward_transformation import build_reward_transformation_fn_from_config
from utils.rl.misc.td_lambda_targets import build_td_lambda_targets_semi_markov, build_td_lambda_targets
from utils.rl.replay.complete_episode_replay_buffer import CompleteEpisodeReplayBuffer
from utils.rl.replay.episode_collate import EpisodeCollate
from utils.torch.iterable_episode_dataset import IterableEpisodeDataset
from utils.torch.models.mlp import MLP


class COMA(MACAgent):
    def __init__(self,
                 action_space,
                 observation_space,
                 graph,
                 config: DictConfig,
                 state_observation_space=None):
        super().__init__(action_space, observation_space, graph, config)

        assert config.on_policy_replay is True
        assert config.mac_output_probs is True

        self.add_last_action_critic = config.add_last_action_critic
        self.update_target_every = config.update_target_every
        self.slow_target_fraction = config.slow_target_fraction
        self.last_target_update_step: int = 0
        self.critic_training_steps: int = 0
        self.td_lambda = config.td_lambda
        self.n_actions = action_space.n
        self.n_agents = config.number_of_agents
        self.max_gradient_norm = config.max_gradient_norm
        self.semi_markov = config.semi_markov
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.reward_transformation_fn = build_reward_transformation_fn_from_config(config)

        self.buffer = CompleteEpisodeReplayBuffer(config)
        self.train_data_set = IterableEpisodeDataset(self.buffer,
                                                     max_sequence_length=config.max_sequence_length,
                                                     seed=config.seed)

        self.train_data_loader = DataLoader(self.train_data_set,
                                            batch_size=self.batch_size,
                                            num_workers=0,
                                            pin_memory=False,
                                            collate_fn=EpisodeCollate(n_agents=config.number_of_agents))

        self.train_iterator = iter(self.train_data_loader)

        #global state
        critic_input_size = state_observation_space.shape[0]
        #actions, last_actions
        critic_input_size += (self.n_actions * self.n_agents) * (2 if self.add_last_action_critic else 1)
        #agent_id
        critic_input_size += self.n_agents

        if "mlp".__eq__( config.critic_name):
            self.critic = MLP(input_size=critic_input_size,
                              output_size=self.n_actions,
                              **config.critic)
        elif "global_critic".__eq__( config.critic_name):
            self.critic = GlobalCritic(state_observation_space.shape[0],
                                       self.n_actions,
                                       self.n_agents,
                                       critic_input_size,
                                       state_observation_space.shape[2],
                                       config.critic)
        else:
            raise NotImplemented()

        self.target_critic = copy.deepcopy(self.critic)

        self.mac.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.agent_parameters = list(self.mac.model.parameters())
        self.critic_parameters = list(self.critic.parameters())

        self.agent_optimizer: Optimizer = ignore_unmatched_kwargs(getattr(torch.optim,
                                                                          config.agent_optimizer.optimizer)) \
            (self.agent_parameters, **config.agent_optimizer)

        self.critic_optimizer: Optimizer = ignore_unmatched_kwargs(getattr(torch.optim,
                                                                           config.critic_optimizer.optimizer)) \
            (self.critic_parameters, **config.critic_optimizer)

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
        batch = batch.to(self.device)

        mask = batch["mask"][:, :-1].float()
        local_observations = batch["local_observations"]
        global_observation = batch["global_observation"]
        actions: torch.LongTensor = batch["actions"][:, :].unsqueeze(-1)  # shape: BxTxNx1
        action_mask = batch["action_masks"][:, :-1]
        rewards = batch["rewards"][:, :-1]
        dones = batch["dones"][:, :-1].float()
        dts = batch["infos"][:, :-1]
        mask[:, 1:] = mask[:, 1:] * (1 - dones[:, :-1])
        #  agents that can not make a decision will calcualte the q-value for the same action from the new state (action_mask takes care of this)
        # todo implement mode for continue action

        rewards = self.reward_transformation_fn(rewards)

        q_values, critic_metrics = self._train_critic(global_observation, rewards, dones, actions, dts, mask)
        actions = actions[:, :-1]  # we do not need the last action from this point on

        mac_out = self.mac.unroll_mac(local_observations[:, :-1]).clone()  # shape BxTxNxA

        # mask out unavailable actions renormalise
        mac_out += 0.000000001  # add a small number so that not everything can become zero and normalization will fail
        mac_out[action_mask == 0] = 0
        mac_out[mask == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[action_mask == 0] = 0
        mac_out[mask == 0] = 0

        mask = mask.reshape(-1)

        # calculate baseline
        q_values = q_values.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_values).sum(dim=-1).detach()

        # calculate masked policy gradient
        q_taken = q_values.gather(dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = pi.gather(dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        advantages = (q_taken - baseline).detach()
        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        self.agent_optimizer.zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_parameters, self.max_gradient_norm)
        self.agent_optimizer.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.update_target_every >= 1.0:
            self._update_target()
            self.last_target_update_step = self.critic_training_steps

        # add critic metrics to metrics
        for key, value in critic_metrics.items():
            metrics[key] = np.mean(value)

        with torch.no_grad():
            metrics["advantage_mean"] = (advantages * mask).sum().item() / mask.sum().item()
            metrics["coma_loss"] = coma_loss.item()
            metrics["pi_max"] = (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item()
            metrics["agent_grad_norm"] = grad_norm.item()

        return metrics

    def _update_target(self):
        if self.slow_target_fraction == 1.0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                with torch.no_grad():
                    target_param.data.copy_(self.slow_target_fraction * param.data + (
                            1.0 - self.slow_target_fraction) * target_param.data)

    def _train_critic(self, global_state, rewards, dones, actions, dts, mask):
        metrics = defaultdict(list)
        with torch.no_grad():
            max_sequence_length = actions.size(1)
            batch_size = actions.size(0)

            critic_inputs = []

            global_state = global_state.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)  # shape B*T*N*D
            critic_inputs.append(global_state)

            # need to feed the actions selected by other agents into the network (we need to mask out the actions of the current agent)
            actions_one_hot = F.one_hot(actions, num_classes=self.n_actions) \
                .float() \
                .view(batch_size, max_sequence_length, 1, -1) \
                .repeat(1, 1, self.n_agents, 1)
            # shape: B*T*N*(N*A)

            agent_mask = (1 - torch.eye(self.n_agents, device=actions.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
            actions_one_hot_masked = actions_one_hot * agent_mask.unsqueeze(0).unsqueeze(0)
            critic_inputs.append(actions_one_hot_masked)
            last_actions = torch.cat([torch.zeros_like(actions_one_hot[:, 0:1]), actions_one_hot[:, :-1]], dim=1)
            if self.add_last_action_critic:
                critic_inputs.append(last_actions)


            agent_ids = torch.eye(self.n_agents, device=actions.device)\
                .unsqueeze(0)\
                .unsqueeze(0)\
                .expand(batch_size,max_sequence_length, -1, -1)

            critic_inputs.append(agent_ids)

            critic_input = torch.cat(critic_inputs, dim=-1)

            target_q_values: torch.FloatTensor = self.target_critic(critic_input)  # global state + actions should be enough here (lcoal observations should not give additional info here )
            target_q_values_taken = target_q_values.gather(dim=3, index=actions).squeeze(3)

            if self.semi_markov:
                targets = build_td_lambda_targets_semi_markov(rewards,
                                                              dones,
                                                              mask,
                                                              target_q_values_taken,
                                                              dts,
                                                              self.n_agents,
                                                              self.gamma,
                                                              self.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards,
                                                  dones,
                                                  mask,
                                                  target_q_values_taken,
                                                  self.n_agents,
                                                  self.gamma,
                                                  self.td_lambda)

        q_values = []

        for t in reversed(range(max_sequence_length - 1)):
            mask_t = mask[:, t].expand(-1, self.n_agents)  # only necessary if we do not have an mask per agent

            if mask_t.sum() == 0:
                continue

            q_t: torch.FloatTensor = self.critic(critic_input[:, t:t+1])
            q_values.insert(0, q_t.view(batch_size, self.n_agents, self.n_actions))
            q_values_for_chosen_actions = q_t.gather(dim=3, index=actions[:, t:t + 1]).squeeze(3).squeeze(1)

            targets_t = targets[:, t]

            td_error = q_values_for_chosen_actions - targets_t.detach()

            masked_td_error = td_error * mask_t

            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.max_gradient_norm)
            self.critic_optimizer.step()
            self.critic_training_steps += 1

            with torch.no_grad():
                metrics["critic_loss"].append(loss.item())
                metrics["critic_grad_norm"].append(grad_norm.item())
                n_masked_elements = mask_t.sum().item()
                metrics["td_error_abs"].append((masked_td_error.abs().sum().item() / n_masked_elements))
                metrics["q_taken_mean"].append((q_values_for_chosen_actions * mask_t).sum().item() / n_masked_elements)
                metrics["target_mean"].append((targets_t * mask_t).sum().item() / n_masked_elements)

        q_values = torch.stack(q_values, dim=1)
        return q_values, metrics

    def save_model(self, path, step, agent_number):
        """Saves the current model of the agent."""
        model_path = path / f"mac_model_{step}_{agent_number}.pth"
        torch.save(self.mac.model.state_dict(), model_path)
        model_path = path / f"critic_target_{step}_{agent_number}.pth"
        torch.save(self.target_critic.state_dict(), model_path)
        model_path = path / f"critic_{step}_{agent_number}.pth"
        torch.save(self.critic.state_dict(), model_path)

    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"mac_model_{step}_{agent_number}.pth"
        self.mac.model.load_state_dict(torch.load(model_path))
        model_path = path / f"critic_target_{step}_{agent_number}.pth"
        self.target_critic.load_state_dict(torch.load(model_path))
        model_path = path / f"critic_{step}_{agent_number}.pth"
        self.critic.load_state_dict(torch.load(model_path))

    def set_test(self, test: bool):
        super(COMA, self).set_test(test)
        self.mac.model.train(not test)
        self.critic.train(not test)
