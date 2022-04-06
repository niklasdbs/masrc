import collections
import pathlib
from multiprocessing import Pipe, Process

import logging
import numpy as np
import torch
from cv2 import cv2
from gym.vector.utils import CloudpickleWrapper
from omegaconf import DictConfig
from tqdm import tqdm

from agents import agents as agents_classes
from trainers.base_trainer import BaseTrainer
from utils.logging.logger import Logger
from utils.rl.replay.replay_buffer import Transition


# based on https://github.com/oxwhirl/pymarl/blob/master/src/runners/parallel_runner.py

class ParallelSequentialTrainer(BaseTrainer):  # todo contains a bit of dupplicate code but inhertinace from seqtrainer is circular import
    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        self.shared_agent = config.shared_agent
        assert self.shared_agent  # currently only a shared agent setup is supported

        agent_to_use = getattr(agents_classes, config.agent)

        if self.shared_agent:
            agent_model = agent_to_use(action_space=validation_env.action_space(0),
                                       observation_space=validation_env.observation_space(0),
                                       graph=validation_env.unwrapped.graph,
                                       config=config)

            if self.load_agent_model:
                agent_model.load_model(pathlib.Path(config.path_to_model), step=self.load_step, agent_number=0)

            self.policy_agents = {agent: agent_model for agent in validation_env.possible_agents}
        else:
            create_agent = lambda agent: agent_to_use(action_space=validation_env.action_space(agent),
                                                      observation_space=validation_env.observation_space(agent),
                                                      graph=validation_env.unwrapped.graph,
                                                      config=config)

            self.policy_agents = {agent: create_agent(agent) for agent in validation_env.possible_agents}

            if self.load_agent_model:
                for agent_id, agent in self.policy_agents.items():
                    agent.load_model(pathlib.Path(config.path_to_model), step=self.load_step, agent_number=agent_id)

        self.number_of_parallel_envs = config.number_of_parallel_envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.number_of_parallel_envs)])
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(train_env)))
                   for worker_conn in self.worker_conns]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("final_advanced_metrics", None))
        self.env_info = self.parent_conns[0].recv()

    def train(self):
        for agent in self.policy_agents.values():
            agent.set_test(False)

        metrics = collections.defaultdict(list)

        current_step = 0
        current_episode = 1
        last_training_step = 0
        last_logging_step = 0
        last_save_step = 0
        last_eval_step = 0
        t = tqdm(total=self.number_of_environment_steps)
        while current_step < self.number_of_environment_steps:
            current_episode_data_per_agent = [{agent: [] for agent in self.evaluation_env.possible_agents} for _ in
                                              range(self.number_of_parallel_envs)]

            for parent_conn in self.parent_conns:
                parent_conn.send(("reset", None))

            previous_observations_for_agent = [{agent: None for agent in self.evaluation_env.possible_agents} for _ in
                                               range(self.number_of_parallel_envs)]
            previous_actions_for_agent = [{agent: None for agent in self.evaluation_env.possible_agents} for _ in
                                          range(self.number_of_parallel_envs)]
            previous_additional_information_for_agent = [{agent: None for agent in self.evaluation_env.possible_agents}
                                                         for _ in range(self.number_of_parallel_envs)]

            terminated = [False for _ in range(self.number_of_parallel_envs)]
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

            for parent_conn in self.parent_conns:
                env_reset_results = parent_conn.recv()

            while True:
                agent_ids = []
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        parent_conn.send(("next_agent", None))
                        next_agent_id = parent_conn.recv()
                        agent_ids.append(next_agent_id)

                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        parent_conn.send(("last", None))

                observations = []
                rewards = []
                discounted_rewards = []
                dones = []
                infos = []
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        observation, reward, discounted_reward, done, info = parent_conn.recv()
                        observations.append(observation)
                        rewards.append(reward)
                        discounted_rewards.append(discounted_reward)
                        dones.append(done)
                        infos.append(info)

                agent = self.policy_agents[0]  # todo currently this will only work with an shared agent
                actions, additional_informations = agent.multi_act(observations)

                action_idx = 0
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        parent_conn.send(("step", actions[action_idx]))
                        action_idx += 1

                step_results_idx = 0
                for idx, parent_conn in enumerate(self.parent_conns):
                    if idx in envs_not_terminated:
                        agent_id = agent_ids[step_results_idx]
                        if previous_observations_for_agent[idx][agent_id] is not None:
                            next_observation = observations[step_results_idx]
                            transition = Transition(previous_observations_for_agent[idx][agent_id],
                                                    previous_actions_for_agent[idx][agent_id],
                                                    next_observation,
                                                    discounted_rewards[step_results_idx] if self.semi_markov else
                                                    rewards[step_results_idx],
                                                    dones[step_results_idx],
                                                    infos[step_results_idx],
                                                    previous_additional_information_for_agent[idx][agent_id])

                            if not self.replay_whole_episodes:
                                agent.buffer.add_transition(transition, idx)#todo !!!

                            current_episode_data_per_agent[idx][agent_id].append(transition)

                        previous_observations_for_agent[idx][agent_id] = observations[step_results_idx]
                        previous_actions_for_agent[idx][agent_id] = actions[step_results_idx]

                        if additional_informations:
                            previous_additional_information_for_agent[idx][agent_id] = additional_informations[
                                step_results_idx]
                        step_results_idx += 1

                step_results = []
                envs_terminated_in_current_step = []
                for idx, parent_conn in enumerate(self.parent_conns):
                    if idx in envs_not_terminated:
                        if not terminated[idx]:
                            step_result, env_terminated = parent_conn.recv()
                            step_results.append(step_result)
                            if env_terminated:
                                envs_terminated_in_current_step.append(idx)

                env_steps = sum(step_results)
                current_step += env_steps
                step_advanced_env = env_steps > 0  # step advanced at least one env
                if step_advanced_env:
                    t.update(env_steps)
                    agent.after_env_step(n_steps=env_steps)

                logging_step = current_episode if self.use_episode_for_logging else current_step

                # should_train = \
                #     (current_step % self.train_every == 0 and step_advanced_env)\
                #     if self.shared_agent else \
                #     (agent.current_step % self.train_every == 0)

                should_train = current_step - last_training_step >= self.train_every

                if should_train and current_step > self.start_learning and not self.train_at_episode_end:
                    last_training_step = current_step
                    for train_iteration in range(self.train_steps):
                        new_metrics = agent.learn()
                        if self.shared_agent:
                            [metrics["learner/" + key].append(value) for key, value in new_metrics.items()]
                        else:
                            [metrics[f"learner_{agent_id}/{key}"].append(value) for key, value in new_metrics.items()]

                step_results_idx = 0
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        agent_id = agent_ids[step_results_idx]

                        if dones[step_results_idx]:
                            if self.replay_whole_episodes:
                                agent.buffer.add_episode(current_episode_data_per_agent[idx][agent_id])

                            episode_length = len(current_episode_data_per_agent[idx][agent_id])
                            score = sum((transition.reward for transition in
                                         current_episode_data_per_agent[idx][agent_id]))
                            score_discounted = sum(
                                    (self.gamma ** transition.info["dt"] * transition.reward for transition in
                                     current_episode_data_per_agent[idx][agent_id]))
                            consumed_time = sum((transition.info["dt"] for transition in
                                                 current_episode_data_per_agent[idx][agent_id]))

                            self.writer.add_scalar(f"train/episode_length_{agent_id}",
                                                   episode_length,
                                                   logging_step,
                                                   epoch=current_episode,
                                                   current_step=current_step)
                            self.writer.add_scalar(f"train/return_{agent_id}",
                                                   score,
                                                   logging_step,
                                                   epoch=current_episode,
                                                   current_step=current_step
                                                   )
                            self.writer.add_scalar(f"train/return_discounted_{agent_id}",
                                                   score_discounted,
                                                   logging_step,
                                                   epoch=current_episode,
                                                   current_step=current_step
                                                   )
                            self.writer.add_scalar(f"train/consumed_time_{agent_id}",
                                                   consumed_time,
                                                   logging_step,
                                                   epoch=current_episode,
                                                   current_step=current_step)

                            current_episode_data_per_agent[idx][agent_id] = []
                        step_results_idx += 1

                # Update envs_not_terminated
                for idx in envs_terminated_in_current_step:
                    terminated[idx] = True

                envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
                all_terminated = all(terminated)
                if all_terminated:
                    step_metric_for_training = current_episode if self.replay_whole_episodes else current_step

                    if self.train_at_episode_end \
                            and step_metric_for_training >= self.start_learning \
                            and step_metric_for_training - last_training_step >= self.train_every:
                        last_training_step = step_metric_for_training
                        for train_iteration in range(self.train_steps):
                            new_metrics = agent.learn()
                            if self.shared_agent:
                                [metrics["learner/" + key].append(value) for key, value in new_metrics.items()]
                            else:
                                [metrics[f"learner_{agent_id}/{key}"].append(value) for key, value in
                                 new_metrics.items()]

                    if logging_step - last_logging_step >= self.log_metrics_every:
                        last_logging_step = logging_step
                        for name, values in metrics.items():
                            if len(values) > 0:
                                self.writer.add_scalar(name,
                                                       np.mean(np.array(values)),
                                                       current_step,
                                                       epoch=current_episode,
                                                       current_step=current_step)
                                metrics[name].clear()

                        for name, values in agent.get_agent_metrics_for_logging().items():
                            self.writer.add_scalar("policy_agent/" + name, np.mean(np.array(values)),
                                                   current_step,
                                                   epoch=current_episode,
                                                   current_step=current_step)

                        self.writer.write()

                    if logging_step - last_eval_step >= self.eval_every:
                        last_eval_step = logging_step
                        result = self.evaluate(logging_step)
                        if self.early_stopping(result):
                            self.save_model(logging_step)
                            self.writer.write()
                            t.close()
                            for parent_conn in self.parent_conns:
                                parent_conn.send(("close", None))
                            return

                    if logging_step - last_save_step >= self.save_model_every:
                        last_save_step = logging_step
                        self.save_model(logging_step)

                    break

            current_episode += self.number_of_parallel_envs
        t.close()
        self.save_model(current_step)

    def evaluate(self, current_step, mode="validation", env=None) -> float:
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        for agent in self.policy_agents.values():
            agent.set_test(True)
        result = 0.0

        render = self.render and mode == "test"

        with torch.no_grad():
            if not env:
                env = self.evaluation_env

            episode_data = {agent: collections.defaultdict(list) for agent in env.possible_agents}

            for episode in tqdm(range(self.eval_episodes)):
                first_reset_in_episode = True  # this is used to set the flag of only doing a

                while True:
                    env_reset_result = env.reset(reset_days=first_reset_in_episode, only_do_single_episode=True)
                    first_reset_in_episode = False

                    if env_reset_result == False:
                        break

                    if render:
                        video = self._create_cv_writer(env, episode)

                    current_episode_data_per_agent = {agent: [] for agent in env.possible_agents}

                    for agent in env.agent_iter():
                        observation, reward, discounted_reward, done, info = env.last()

                        action, _ = self.policy_agents[agent].act(observation)

                        step_advanced_env = env.step(action)

                        if render and step_advanced_env:
                            imgs = env.render(show=False)
                            for img in imgs:
                                video.write(img)

                        transition = Transition(observation,
                                                action,
                                                observation,
                                                reward,
                                                done,
                                                info)  # todo save next observation correct
                        current_episode_data_per_agent[agent].append(transition)

                    self._collect_stats_for_day(current_episode_data_per_agent, episode_data)

                    if render:
                        cv2.destroyAllWindows()
                        video.release()

                metrics = self.log_evaluation_metrics(current_step, env, episode_data, mode)

                result += np.sum(metrics["return_mean"])

        for agent in self.policy_agents.values():
            agent.set_test(False)
        self.writer.write()

        return result / self.eval_episodes

    def save_model(self, current_step: int):
        for agent_id, agent in self.policy_agents.items():
            agent.save_model(self.save_model_path, current_step, agent_id)


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.fn()
    agent_iterator = None
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            step_result = env.step(actions)
            terminated = not env.agents
            remote.send((step_result, terminated))
        elif cmd == "reset":
            env_reset_result = env.reset()
            agent_iterator = iter(env.agent_iter())
            remote.send(env_reset_result)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "final_advanced_metrics":
            remote.send(env.final_advanced_metrics)
        elif cmd == "next_agent":
            remote.send(next(agent_iterator))
        elif cmd == "last":
            remote.send(env.last())
        else:
            raise NotImplementedError
