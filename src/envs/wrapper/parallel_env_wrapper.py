from collections import defaultdict
from typing import Union

import functools
import numpy as np
from gym import spaces
from pettingzoo import ParallelEnv

from envs.potop_env import PotopEnv


class ToParallelWrapper(ParallelEnv):
    def __init__(self, env: PotopEnv):
        self.env = env
        self._agent_iterator = None #iter(self.env.agent_iter())


    @property
    def observation_spaces(self):
        try:
            return {agent: self.observation_space(agent) for agent in self.possible_agents}
        except AttributeError:
            raise AttributeError(
                    "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead")

    @property
    def action_spaces(self):
        try:
            return {agent: self.action_space(agent) for agent in self.possible_agents}
        except AttributeError:
            raise AttributeError(
                    "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
                "observation": self.env.observation_space(agent),
                "action_mask": spaces.Discrete(1),  # not yet implemented
                "needs_to_act": spaces.Discrete(1)
        })

    def action_space(self, agent):
        return self.env.action_space(agent)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observe(self, agent):
        # return the observation at the last step for a specific agent
        return {
                "observation": self.env.observe(agent),
                "action_mask": -1,  # not yet implemented #todo implement action mask
                "needs_to_act": int(self.env.agent_states[agent].current_action is None)}

    def reset(self, reset_days=False, only_do_single_episode=False) -> Union[bool, dict]:
        statement = self.env.reset(reset_days, only_do_single_episode)
        self._agent_iterator = iter(self.env.agent_iter())
        observations = {agent: self.observe(agent) for agent in self.env.agents if not self.env.dones[agent]}
        if statement:
            return observations
        else:
            return False

    def step(self, actions):
        # using np.float32 is important, otherwise we will have float64 tensors
        rewards = defaultdict(np.float32)
        discounted_rewards = defaultdict(np.float32)

        for i in range(len(actions)):
            agent = next(self._agent_iterator)

            step_advanced_env = self.env.step(actions[agent])

            if step_advanced_env and i + 1 < len(actions):
                raise Exception("provided actions for agents that should not act")

        for agent in self.env.agents:
            rewards[agent] += self.env.rewards[agent]
            discounted_rewards[agent] += self.env.discounted_rewards[agent]

        dones = dict(self.env.dones)
        infos = dict(self.env.infos)
        observations = {agent: self.observe(agent) for agent in self.env.agents}

        #make sure all agents do the empty env step (so that the env is in the correct done state)
        while self.env.agents and self.env.dones[self.env.agent_selection]:
            self.env.step(None)

        return observations, rewards, discounted_rewards, dones, infos

    def last(self, agent=None, observe=True):
        if agent is None:
            agent = self.agent_selection
            assert self.agent_states[
                       agent].current_action is None, "currently between the actions of an agent, a valid observation may not be created"

        observation = self.observe(agent) if observe else None
        observations = {agent: self.observe(agent) for agent in self.env.agents}

        return observations, \
               self.env.rewards[agent], \
               self.env.discounted_rewards[agent], \
               self.env.dones[agent], \
               self.env.infos[agent]

    def render(self, mode="human", show=False):
        return self.env.render(mode, show)

    def state(self):
        return self.env.state()

    def close(self):
        return self.env.close()

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return object.__getattribute__(self.env, name)

    def __setattr__(self, name, value):
        try:
            return object.__setattr__(self, name, value)
        except AttributeError:
            return object.__setattr__(self.env, name, value)

    def __delattr__(self, name):
        try:
            return object.__delattr__(self, name)
        except AttributeError:
            return object.__delattr__(self.env, name)
