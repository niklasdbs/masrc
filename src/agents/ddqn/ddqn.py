"""
Module contains functionality for a double DQN agent.
"""
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data.dataloader import default_collate

from agents.agent import Agent
from modules.grcn.grcn_model import GraphConvolutionResourceNetwork
from omegaconf import DictConfig
from envs.utils import get_distance_matrix
from modules.grcn.grcn_model_shared_agent import GraphConvolutionResourceNetworkSharedAgent
from modules.grcn.grcn_shared_info_model import GraphConvolutionResourceNetworkSharedInfo
from modules.grcn.grcn_twin import GraphConvolutionResourceNetworkTwin
from modules.grcn.grcn_twin_af_atten import GraphConvolutionResourceNetworkTwinAfAttention
from modules.grcn.grcn_twin_af_atten_transfer import GraphConvolutionResourceNetworkTwinAfAttentionTransfer
from modules.grcn.grcn_twin_after_dist import GraphConvolutionResourceNetworkTwinAfterDist
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.rl.action_selectors import EpsilonGreedyActionSelector
from utils.rl.misc.reward_transformation import build_reward_transformation_fn_from_config
from utils.rl.replay.batch import recursive_to, recursive_unsqueeze
from utils.rl.replay.episode_collate import recursive_pad_sequence
from utils.rl.replay.prioritized_experience_buffer import PrioritizedExperienceBuffer
from utils.rl.schedules import epsilon_schedule
from utils.rl.schedules.epsilon_schedule import DecayThanFlatEpsilonSchedule
from utils.torch.iterable_rollout_dataset import IterableRolloutDataset
from utils.torch.iterable_transition_dataset import IterableTransitionDataset
from utils.torch.models.mlp import MLP


class DDQN(Agent):

    def __init__(self, action_space, observation_space, graph, config: DictConfig):
        super().__init__(action_space, observation_space, graph, config)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.action_selector = EpsilonGreedyActionSelector(config)  # todo do not hardcode
        self.current_gradient_step = 0

        self.semi_markov = config.semi_markov

        self.sequence_length = config.sequence_length
        self.over_sample_ends = config.over_sample_ends
        self.slow_target_fraction = config.slow_target_fraction
        self.update_target_every = config.update_target_every

        self.train_data_set = \
            IterableRolloutDataset(self.buffer,
                                   sequence_length=self.sequence_length,
                                   over_sample_ends=self.over_sample_ends,
                                   seed=config.seed) \
                if self.replay_whole_episodes else \
                IterableTransitionDataset(self.buffer, seed=config.seed)

        self.train_data_loader = torch.utils.data.DataLoader(self.train_data_set,
                                                             batch_size=self.batch_size,
                                                             num_workers=0,
                                                             pin_memory=True)

        self.train_iterator = iter(self.train_data_loader)

        self.prioritized_replay = config.get("prioritized_replay", False)

        if self.prioritized_replay:
            self.learn = self.learn_prio
            self.buffer = PrioritizedExperienceBuffer(config)


        self.gradient_clipping = config.gradient_clipping
        self.reward_transformation_fn = build_reward_transformation_fn_from_config(config)
        self.max_gradient_norm = config.max_gradient_norm
        self.double_dqn = config.double_dqn

        self.distance_matrix = torch.from_numpy(get_distance_matrix(graph)) \
            .float().to(self.device)

        if config.model.name == "mlp":
            action_dim: int = action_space.n
            state_dim: int = observation_space.shape[0]
            self.q_net = MLP(input_size=state_dim, output_size=action_dim, **config.model.mlp)
            self.target_q_net = MLP(input_size=state_dim, output_size=action_dim, **config.model.mlp)
        elif config.model.name == "grcn":
            self.q_net = GraphConvolutionResourceNetwork(distance_matrix=self.distance_matrix,
                                                         resource_dim=observation_space.shape[1],
                                                         config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetwork(distance_matrix=self.distance_matrix,
                                                                resource_dim=observation_space.shape[1],
                                                                config=config.model.grcn)
        elif config.model.name in ["grcn_shared_info"]:
            self.q_net = GraphConvolutionResourceNetworkSharedInfo(distance_matrix=self.distance_matrix,
                                                                   observation_space=observation_space["observation"],
                                                                   number_of_agents=config.number_of_agents,
                                                                   config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkSharedInfo(distance_matrix=self.distance_matrix,
                                                                          observation_space=observation_space[
                                                                              "observation"],
                                                                          number_of_agents=config.number_of_agents,
                                                                          config=config.model.grcn)
        elif config.model.name in ["grcn_twin"]:
            self.q_net = GraphConvolutionResourceNetworkTwin(distance_matrix=self.distance_matrix,
                                                             observation_space=observation_space,
                                                             number_of_agents=config.number_of_agents,
                                                             config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkTwin(distance_matrix=self.distance_matrix,
                                                                    observation_space=observation_space,
                                                                    number_of_agents=config.number_of_agents,
                                                                    config=config.model.grcn)
        elif config.model.name in ["grcn_twin_after_dist"]:
            self.q_net = GraphConvolutionResourceNetworkTwinAfterDist(distance_matrix=self.distance_matrix,
                                                                      observation_space=observation_space,
                                                                      number_of_agents=config.number_of_agents,
                                                                      config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkTwinAfterDist(distance_matrix=self.distance_matrix,
                                                                             observation_space=observation_space,
                                                                             number_of_agents=config.number_of_agents,
                                                                             config=config.model.grcn)
        elif config.model.name in ["grcn_shared_agent"]:
            assert config.shared_agent
            self.q_net = GraphConvolutionResourceNetworkSharedAgent(distance_matrix=self.distance_matrix,
                                                                    observation_space=observation_space,
                                                                    number_of_agents=config.number_of_agents,
                                                                    config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkSharedAgent(distance_matrix=self.distance_matrix,
                                                                           observation_space=observation_space,
                                                                           number_of_agents=config.number_of_agents,
                                                                           config=config.model.grcn)
        elif config.model.name in ["grcn_twin_att"]:
            assert config.shared_agent
            self.q_net = GraphConvolutionResourceNetworkTwinAfAttention(distance_matrix=self.distance_matrix,
                                                                    observation_space=observation_space,
                                                                    number_of_agents=config.number_of_agents,
                                                                    config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkTwinAfAttention(distance_matrix=self.distance_matrix,
                                                                           observation_space=observation_space,
                                                                           number_of_agents=config.number_of_agents,
                                                                           config=config.model.grcn)
        elif config.model.name in ["grcn_twin_att_transfer"]:
            assert config.shared_agent
            self.q_net = GraphConvolutionResourceNetworkTwinAfAttentionTransfer(distance_matrix=self.distance_matrix,
                                                                    observation_space=observation_space,
                                                                    number_of_agents=config.number_of_agents,
                                                                    config=config.model.grcn)
            self.target_q_net = GraphConvolutionResourceNetworkTwinAfAttentionTransfer(distance_matrix=self.distance_matrix,
                                                                           observation_space=observation_space,
                                                                           number_of_agents=config.number_of_agents,
                                                                           config=config.model.grcn)
        else:
            raise NotImplementedError

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

        if self.prioritized_replay:
            self.loss_criterion = getattr(torch.nn, config.loss_function)(reduction="none")
        else:
            self.loss_criterion = getattr(torch.nn, config.loss_function)()

        self.optimizer = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer)) \
            (self.q_net.parameters(), **config.model_optimizer)

        lr_schedule_lambda = lambda epoch: 1.0
        # max(0.99 ** epoch,
        #                                    0.0001 / config.model_optimizer.lr)  # todo do not hardcode
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=lr_schedule_lambda,
                                                              last_epoch=-1)
    def act(self, state, return_q_values=False):
        # this is a bit faster than using the action selector directly, as we only compute q-values if they are not random
        if not self.test and np.random.rand() < self.action_selector.epsilon:
            return self.action_space.sample(), None
        else:
            with torch.no_grad():
                state = recursive_pad_sequence([recursive_to(pin_memory(default_collate([state])), self.device)], batch_first=True)
                q_values = self.q_net(state)
                action = q_values.argmax()
                if return_q_values:
                    return action.item(), q_values.cpu().numpy()
                else:
                    return action.item(), None

    # def act(self, state):
    #     with torch.no_grad():
    #         observation_batch = recursive_pad_sequence([recursive_to(default_collate([state]), self.device)], batch_first=False)
    #         q_values = self.q_net(observation_batch)
    #
    #         actions = self.action_selector.select_actions(q_values=q_values, test=self.test)
    #         return actions.item(), None

    def multi_act(self, state):
        with torch.no_grad():
            observation_batch = recursive_unsqueeze(recursive_to(pin_memory(default_collate(state)), self.device),
                                                       dim=1)
            q_values = self.q_net(observation_batch)

            actions = self.action_selector.select_actions(q_values=q_values, test=self.test).squeeze(axis=-1)
            return actions, None

    def learn(self):
        batch = next(self.train_iterator)
        batch = recursive_unsqueeze(recursive_to(batch, self.device), dim=0)
        metrics = {}
        observations, actions, next_observations, rewards, dones, dts = batch
        dones = dones.float()

        rewards = self.reward_transformation_fn(rewards)

        current_q_values = self.q_net(observations).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # compute targets
        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.q_net(next_observations).argmax(dim=-1)
                next_state_values = self.target_q_net(next_observations) \
                    .gather(-1, best_actions.unsqueeze(-1)) \
                    .squeeze(-1)
            else:
                next_state_values = self.target_q_net(next_observations).max(-1)[0]

            if self.semi_markov:
                target_q_values = rewards + (1 - dones) * torch.pow(self.gamma, dts) * next_state_values
            else:
                target_q_values = rewards + (1 - dones) * self.gamma * next_state_values

        loss = self.loss_criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping:
            # Clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient_norm)
            metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.current_gradient_step += 1
        if self.current_gradient_step % self.update_target_every == 0:
            self.update_target()

        with torch.no_grad():
            metrics["loss"] = loss.item()
            metrics["td_error_abs"] = (current_q_values - target_q_values).abs().mean().item()
            metrics["q_taken_mean"] = current_q_values.mean().item()
            metrics["target_mean"] = target_q_values.mean().item()
            metrics["lr"] = self.lr_scheduler.get_last_lr()

        return metrics

    def learn_prio(self):
        if self.prioritized_replay:
            batch = self.buffer.sample(self.batch_size)
            indices = batch["indices"]
            batch = {key : pin_memory(default_collate(batch[key])) for key in batch}
            batch = recursive_pad_sequence([batch], batch_first=False)
            #batch = recursive_pad_sequence([batch], batch_first=False)
        else:
            batch = next(self.train_iterator)
            batch = recursive_pad_sequence([recursive_to(batch, self.device)], batch_first=True)
        metrics = {}
        states = recursive_to(batch["states"], self.device)
        next_states = recursive_to(batch["next_states"], self.device)
        actions: torch.LongTensor = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device).float()
        dts = batch["infos"].to(self.device)

        if self.prioritized_replay:
            is_weights = batch["is_weights"].to(self.device)
        else:
            is_weights = torch.ones_like(rewards)



        rewards = self.reward_transformation_fn(rewards)

        current_q_values = self.q_net(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # compute targets
        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.q_net(next_states).argmax(dim=-1)
                next_state_values = self.target_q_net(next_states) \
                    .gather(-1, best_actions.unsqueeze(-1)) \
                    .squeeze(-1)
            else:
                next_state_values = self.target_q_net(next_states).max(-1)[0]

            if self.semi_markov:
                target_q_values = rewards + (1 - dones) * torch.pow(self.gamma, dts) * next_state_values
            else:
                target_q_values = rewards + (1 - dones) * self.gamma * next_state_values


            td_error_abs = (current_q_values - target_q_values).abs()


        loss = self.loss_criterion(current_q_values, target_q_values)
        loss = (is_weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping:
            # Clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient_norm)
            metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.current_gradient_step += 1
        if self.current_gradient_step % self.update_target_every == 0:
            self.update_target()

        with torch.no_grad():
            metrics["loss"] = loss.item()
            metrics["td_error_abs"] = td_error_abs.mean().item()
            metrics["q_taken_mean"] = current_q_values.mean().item()
            metrics["target_mean"] = target_q_values.mean().item()
            metrics["lr"] = self.lr_scheduler.get_last_lr()
            if self.prioritized_replay:
                metrics["importance_weights"] = is_weights.mean().item()
                self.buffer.update(indices, td_error_abs.view(-1).detach().cpu().numpy())
                metrics["beta"] = self.buffer._beta

        return metrics


    def after_env_step(self, n_steps: int = 1):
        """
        This method is called after each environment step at the end
        @return:
        """
        super().after_env_step(n_steps)
        self.action_selector.after_env_step(n_steps)

    def get_agent_metrics_for_logging(self) -> Dict[str, Any]:
        return self.action_selector.get_metrics_for_logging()

    def log_weight_histogram(self, logger, global_step, prefix=""):
        logger.add_weight_histogram(self.q_net, global_step, prefix=prefix)

    def update_target(self):
        if self.slow_target_fraction == 1.0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                with torch.no_grad():
                    target_param.data.copy_(self.slow_target_fraction * param.data + (
                                1.0 - self.slow_target_fraction) * target_param.data)

    def save_model(self, path, step, agent_number):
        """Saves the current model of the agent."""
        model_path = path / f"ddqn_target_model_{step}_{agent_number}.pth"
        torch.save(self.target_q_net.state_dict(), model_path)
        model_path = path / f"ddqn_model_{step}_{agent_number}.pth"
        torch.save(self.q_net.state_dict(), model_path)

    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"ddqn_model_{step}_{agent_number}.pth"
        self.q_net.load_state_dict(torch.load(model_path), strict=False)
        model_path = path / f"ddqn_target_model_{step}_{agent_number}.pth"
        self.target_q_net.load_state_dict(torch.load(model_path), strict=False)
