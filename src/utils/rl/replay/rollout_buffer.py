import torch

from utils.rl.replay.transition import Transition


class RolloutBuffer:
    """
    Used for on-policy algorithms to keep the current rollout. Needs to be reset after each policy update.
    """
    def __init__(self, config):
        self.buffer : [Transition] = []

    def __len__(self):
        return len(self.buffer)


    def add_episode(self, episode : [Transition]):
        self.buffer = episode.copy()

    def add_transition(self, transition : Transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()

    def get_tensor_for_on_policy(self, device : torch.device):
        #todo make this nicer
        states = []
        actions = []
        rewards = []
        dones = []
        infos = []
        next_states = []
        # values = []
        # log_prob = []

        for transition in self.buffer:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
            infos.append(transition.info["dt"])  # todo handle case without dt
            # log_prob.append(transition.additional_information[0])
            # values.append(transition.additional_information[1])

        #states = torch.tensor(states, device=device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        dones = torch.tensor(dones, device=device)
        infos = torch.tensor(infos, device=device)
        # log_prob = torch.stack(log_prob).view(-1)
        # values = torch.stack(values).view(-1)

        return states, actions, rewards, dones, infos, next_states #, log_prob, values







