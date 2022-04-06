from pathlib import Path
from itertools import chain

import numpy as np
import torch.distributions
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR, ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataloader import default_collate

from agents.agent import Agent
from envs.utils import get_distance_matrix
from modules.attention_graph_decoder import AttentionGraphDecoder
from modules.grcn.grcn_twin import GraphConvolutionResourceNetworkTwin
from modules.grcn.grcn_twin_af_atten import GraphConvolutionResourceNetworkTwinAfAttention
from modules.grcn.grcn_twin_after_dist import GraphConvolutionResourceNetworkTwinAfterDist
from modules.mardam.mardam import MARDAM
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs
from utils.rl.misc.reward_transformation import build_reward_transformation_fn_from_config
from utils.rl.misc.td_lambda_targets import calculate_n_step_targets
from utils.rl.replay.batch import Batch, recursive_to
from utils.rl.replay.complete_episode_replay_buffer import CompleteEpisodeReplayBuffer
from utils.rl.replay.episode_collate import EpisodeCollate, recursive_pad_sequence
from utils.rl.replay.parallel_rollout_buffer import ParallelRolloutBuffer
from utils.rl.replay.transition import Transition
from utils.torch.iterable_episode_dataset import IterableEpisodeDataset


class ActorCriticAgent(Agent):
    def __init__(self, action_space, observation_space, graph, config: DictConfig):
        super().__init__(action_space, observation_space, graph, config)
        self.distance_matrix = torch.from_numpy(get_distance_matrix(graph)).float().to(self.device)

        self.critic_coefficient = 1.0  # todo
        self.entropy_coefficient = 0.0  # todo

        self.greedy_testing = False  # todo

        self.reward_transformation_fn = build_reward_transformation_fn_from_config(config)
        # todo do not hardcode
        # model = GraphConvolutionResourceNetworkTwin(distance_matrix=self.distance_matrix,
        #                                                  observation_space=observation_space,
        #                                                  number_of_agents=config.number_of_agents,
        #                                                  config=config.model.grcn)

        # model = GraphConvolutionResourceNetworkTwinAfterDist(distance_matrix=self.distance_matrix,
        #                                                  observation_space=observation_space,
        #                                                  number_of_agents=config.number_of_agents,
        #                                                  config=config.model.grcn)

        model = GraphConvolutionResourceNetworkTwinAfAttention(distance_matrix=self.distance_matrix,
                                                               observation_space=observation_space,
                                                               number_of_agents=config.number_of_agents,
                                                               config=config.model.grcn,
                                                               use_value_head=True)

        # model = AttentionGraphDecoder(
        #         n_actions=action_space.n,
        #         n_agents=config.number_of_agents,
        #         resource_dim=observation_space.shape[1],
        #         distance_matrix=self.distance_matrix,
        #         config=config.model)

        # model = MARDAM(action_space=action_space, observation_space=observation_space, config=config.model)#todo

        self.batch_size = config.batch_size
        self.model = model
        self.model.to(self.device)
        # self.buffer = CompleteEpisodeReplayBuffer(config, transition_type=Transition)
        # self.train_data_set = IterableEpisodeDataset(self.buffer,
        #                                              max_sequence_length=config.max_sequence_length,
        #                                              seed=config.seed)
        #
        # self.train_data_loader = DataLoader(self.train_data_set,
        #                                     batch_size=self.batch_size,
        #                                     num_workers=0,
        #                                     pin_memory=False,
        #                                     collate_fn=EpisodeCollate(1 if config.shared_agent else config.number_of_agents))
        #
        # self.train_iterator = iter(self.train_data_loader)

        self.buffer = ParallelRolloutBuffer()

        self.gamma = config.gamma
        self.max_gradient_norm = config.max_gradient_norm
        self.semi_markov = config.semi_markov

        # self.optimizer: torch.optim.Optimizer = ignore_unmatched_kwargs(getattr(torch.optim,
        #                                                                         config.model_optimizer.optimizer)) \
        #     (self.model.parameters(), **config.model_optimizer)
        self.optimizer: torch.optim.Optimizer = ignore_unmatched_kwargs(getattr(torch.optim,
                                                                                config.model_optimizer.optimizer)) \
        ([{"params": self.model.non_value_params, "lr": 0.00001},
          {"params": self.model.value_params, "lr": 0.001}])

        lr_schedule_lambda = lambda epoch: max(0.99 ** epoch,
                                               0.0001 / config.model_optimizer.lr)  # todo do not hardcode
        lr_schedule_lambda = lambda epoch: 1.0
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=[lr_schedule_lambda, lr_schedule_lambda],
                                                              last_epoch=-1)

    def act(self, state, **kwargs) -> (int, None):
        train_mode_before = self.model.training
        self.model.train(mode=False)

        with torch.no_grad():
            model_input = recursive_pad_sequence([recursive_to(default_collate([state]), self.device)],
                                                 batch_first=True)
            # action = self.model.policy(*model_input)
            logits, values = self.model(model_input)

            if self.test and self.greedy_testing:
                action = logits.argmax().item()
            else:
                action = Categorical(logits=logits).sample().item()

            self.model.train(mode=train_mode_before)

            return action, None

    def multi_act(self, state):
        train_mode_before = self.model.training
        self.model.train(mode=False)

        with torch.no_grad():
            model_input = recursive_pad_sequence([recursive_to(default_collate(state), self.device)], batch_first=False)
            # action = self.model.policy(*model_input)
            logits, values = self.model(model_input)

            if self.test and self.greedy_testing:
                action = logits.argmax(dim=-1).cpu().numpy()
            else:
                action = Categorical(logits=logits).sample().cpu().numpy()

            action = action.squeeze(axis=-1)

            self.model.train(mode=train_mode_before)

            return action, None

    def learn(self) -> {}:
        if len(self.buffer) == 0:
            return {}

        # batch shape: BxTxN
        # batch = next(self.train_iterator)
        batch = self.buffer.get_batch()
        self.buffer.clear()
        batch = batch.to(self.device)

        mask = batch["mask"].float().squeeze(-1)
        states = batch["states"]
        next_states = batch["next_states"]
        actions: torch.LongTensor = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"].float()
        dts = batch["infos"]
        #  agents that can not make a decision will calcualte the q-value for the same action from the new state (action_mask takes care of this)

        rewards = self.reward_transformation_fn(rewards)

        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # returns = self.calc_returns(rewards, dones, dts, mask)

        if True: #todo
            with torch.no_grad():
                _, next_state_values = self.model(next_states)
        else:
            next_state_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, 0]).unsqueeze(-1)],dim=-1)


        returns = calculate_n_step_targets(dones=dones,
                                           dts=dts,
                                           mask=mask,
                                           rewards=rewards,
                                           target_max_q_values=next_state_values,
                                           gamma=self.gamma,
                                           n_steps=1,
                                           semi_markov=self.semi_markov)

        advantage = (returns - values).detach()

        actor_loss = ((-log_probs * advantage) * mask).sum() / mask.sum()
        critic_loss = (((returns - values) * mask) ** 2).sum() / mask.sum()

        entropy = (dist.entropy() * mask).sum().item() / mask.sum()
        entropy_loss = -entropy

        loss = actor_loss + self.critic_coefficient * critic_loss + self.entropy_coefficient * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(chain.from_iterable(group["params"] for group in self.optimizer.param_groups), self.max_gradient_norm)

        self.optimizer.step()
        self.lr_scheduler.step()

        return {"loss": loss.item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy.item(),
                "pi_max": (dist.probs.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                "grad_norm": grad_norm.item(),
                "lr": self.lr_scheduler.get_last_lr()
                }

    def set_test(self, test: bool):
        super().set_test(test)
        self.model.train(not test)

    def save_model(self, path: Path, step: int, agent_number):
        model_path = path / f"ac_model_{step}_{agent_number}.pth"
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, path: Path, step: int, agent_number):
        """Loads the pretrained model."""
        model_path = path / f"ac_model_{step}_{agent_number}.pth"
        self.model.load_state_dict(torch.load(model_path))
