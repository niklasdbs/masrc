import numpy as np
import logging

import torch

from typing import List

from agents.agent import Agent
from config.settings import DEVICE, NUM_AGENTS

from torch import nn
from torch.nn.functional import softmax

from agents.ddqn.prioritized_replay_memory import ReplayMemory


class ComNet(Agent):

    def __init__(self, action_space, observation_space, params):
        super().__init__(action_space, observation_space)
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_agents = NUM_AGENTS

        learning_rate = params["learning_rate"]
        self.minibatch_size = params["minibatch_size"]
        self.target_update_interval = params["target_update_interval"]

        self.epsilon = params["epsilon"]["start"]
        self.epsilon_min = params["epsilon"]["min"]
        self.epsilon_decay = params["epsilon"]["decay"]

        self.warmup_phase = params["warmup_phase"]
        self.update_step = params["update_step"]
        self.memory = ReplayMemory(size=params["memory_size"])
        self.training_count = 0

        action_dim: int = action_space.n
        state_dim: int = observation_space.shape[0] * observation_space.shape[1]
        # for now take same dim for hidden state as state
        hidden_dim = observation_space.shape[0] * observation_space.shape[1]

        self.policy_model = ComNetModule(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_model = ComNetModule(state_dim, action_dim, hidden_dim).to(DEVICE)

        parameters = self.policy_model.parameters()
        if params["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(parameters, learning_rate)
        elif params["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(parameters, learning_rate)

        if params["loss"] == "mse":
            self.loss = torch.nn.MSELoss()
        elif params["loss"] == "l1":
            self.loss = torch.nn.SmoothL1Loss()

    def act(self, state, available_agents: List[bool] = [True]):
        """Returns an action based on the state of the environment."""
        rand = np.random.random()
        if rand < self.epsilon and not self.test:
            # random action
            action = [self.action_space.sample() if i else None for i in available_agents]
        else:
            state = [torch.from_numpy(agent_state).to(DEVICE) for agent_state in state]
            state = torch.stack(state)
            q_values = self.policy_model(state.unsqueeze(0), len(available_agents))
            q_values = q_values.squeeze(0)
            action = []
            for i, available in enumerate(available_agents):
                if available:
                    action.append(q_values[i].argmax().item())
                else:
                    action.append(None)

        self._epsilon_decay()
        return action

    def _epsilon_decay(self):
        if (self.epsilon - self.epsilon_decay) >= self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        logging.debug("Epsilon %s", self.epsilon)

    def save_transition(self, next_states, actions, rewards, old_states, dones):
        """Saves a transition to the agent's memory.

        Columns of the state vector:
        0.-3. one-hot encoded:
            - free
            - occupied,
            - violation
            - fined
        4. walking time to resource
        5. current time
        6. time of arrival
        7. -1 to +2 indicator of violation time
        [8. allowed parking time]
        """
        old_state = []
        for state in old_states:
            old_state.append(torch.FloatTensor(
                state[:, :self.observation_space.shape[1]]))  # (531, x), x={8, 9}
        next_state = []
        for state in next_states:
            next_state.append(torch.FloatTensor(
                state[:, :self.observation_space.shape[1]]))  # (531, x), x={8, 9}

        action = torch.tensor(actions)  # (1,1) (int)
        reward = torch.FloatTensor([rewards])  # 1 (float)
        # done doesn't have to be converted

        self.memory.save((torch.stack(old_state), action, reward, torch.stack(next_state), dones))

        self.update()

    def update(self):
        """Updates the weights to optimize a ddqn loss function."""
        if self.warmup_phase > 0:
            self.warmup_phase -= 1
            return
        if self.training_count == 0:
            logging.info("Warmup phase ended!")
        self.training_count += 1

        # Update only each update_step steps
        if self.training_count % self.update_step != 0:
            return

        minibatch = self.memory.sample_batch(self.minibatch_size)

        states, actions, rewards, next_states, non_final_mask = get_tensors(minibatch)

        on_policy_q_values = self.policy_model(states.to(DEVICE), self.num_agents) \
            .gather(2, actions.unsqueeze(1).to(DEVICE)).squeeze()
        next_actions = self.policy_model(next_states.to(DEVICE), self.num_agents).max(dim=-1)[1]

        # Calculate the q-values for those proposed actions via the target network
        off_policy_q_values = torch.zeros(self.minibatch_size, self.num_agents, device=DEVICE)

        unmasked_q = self.target_model(next_states.to(DEVICE), self.num_agents) \
            .gather(2, next_actions.unsqueeze(1)).squeeze().detach()

        for i, mask in non_final_mask:
            off_policy_q_values[i][mask] = unmasked_q[i]
        for index in range(len(off_policy_q_values)):
            off_policy_q_values[index] = torch.add(off_policy_q_values[index], rewards[index])
        self.perform_gradient_step(on_policy_q_values, off_policy_q_values)

        # Every few steps, update target network
        if self.training_count % self.target_update_interval == 0:
            logging.debug("frozen weights are updated")
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.target_model.eval()

    def perform_gradient_step(self, on_policy_q_values, expected_off_policy_action_q_values):
        """Calculates the loss and performs a gradient step."""
        loss = self.loss(on_policy_q_values, expected_off_policy_action_q_values)
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_tensors(minibatch):
    """Returns the tensors based on the sampled minibatch of transitions."""

    states, actions, rewards, next_states, dones = tuple(zip(*minibatch))
    # for dones in all_dones:
    #     non_final_masks.append([not done for done in dones])
    # non_final_masks = torch.tensor(non_final_masks, dtype=torch.bool, device=DEVICE).detach()
    non_final_mask = torch.tensor([[not do for do in done] for done in dones], dtype=torch.bool,
                                  device=DEVICE).detach()
    # values are already tensors, we just have to stack them
    states = torch.stack(states)  # (64,#agets, 531, x), x={8, 9}
    rewards = torch.stack(rewards).squeeze()  # 64
    actions = torch.stack(actions).squeeze()  # 64
    # for s, done in zip(next_states, dones):
    #
    #     for i, state in enumerate(s):
    #         if done[i]:
    #             pass
    next_states = [s for s, done in zip(next_states, dones) if not done.any()]
    next_states = torch.stack(next_states)  # (64-x, 531, x), x={8, 9}

    return states, actions, rewards, next_states, non_final_mask


class ComNetModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        communication_dim = hidden_dim * 2
        # crate model for each agent individually?
        self.encoder_model = ComNetEncoder(state_dim, hidden_dim).to(DEVICE)
        self.decoder_model = ComNetDecoder(hidden_dim, action_dim).to(DEVICE)
        self.communication_model = CommunicationLayer(communication_dim, hidden_dim).to(DEVICE)

    def forward(self, all_states, num_agents):

        all_q_values = []
        for states in all_states:
            hidden_states0 = []
            for i in range(num_agents):
                resource_matrix = states[i]
                hidden_states0.append(self.encoder_model(resource_matrix))
            hidden_states0 = torch.stack(hidden_states0).to(DEVICE)
            com_states0 = [torch.zeros(self.hidden_dim).to(DEVICE) for _ in range(num_agents)]
            com_states0 = torch.stack(com_states0).to(DEVICE)
            hidden_states1 = []
            for i in range(num_agents):
                hidden_states1.append(
                    self.communication_model(torch.cat((hidden_states0[i], com_states0[i]), 0)))
            com_states1 = []
            for i in range(num_agents):
                com = None
                for ind, hidden in enumerate(hidden_states1):
                    if ind == i:
                        continue
                    if com is None:
                        com = hidden
                    else:
                        com += hidden
                com_states1.append(com / (num_agents - 1))
            hidden_states2 = []
            for i in range(num_agents):
                hidden_states2.append(
                    self.communication_model(torch.cat((hidden_states1[i], com_states1[i]), 0)))

            q_values = []
            for i in range(num_agents):
                q_values.append(self.decoder_model(hidden_states2[i]))
            all_q_values.append(torch.stack(q_values).to(DEVICE))

        all_q_values = torch.stack(all_q_values).to(DEVICE)

        return all_q_values


class ComNetEncoder(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        flat_state = torch.flatten(state.float(), 0).to(self.device)
        out = softmax(self.fc1(flat_state)).to(self.device)
        return out


class CommunicationLayer(nn.Module):
    def __init__(self, communication_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(communication_dim, hidden_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        out = torch.tanh(self.fc1(state)).to(self.device)
        return out


class ComNetDecoder(nn.Module):

    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        # state = torch.flatten(state.float()).to(self.device)
        out = softmax(self.fc1(state)).to(self.device)
        return out
