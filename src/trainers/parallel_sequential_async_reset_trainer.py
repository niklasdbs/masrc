import collections

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from trainers import ParallelSequentialTrainer
from utils.logging.logger import Logger
from utils.rl.replay.transition import Transition


class ParallelSequentialAsyncResetTrainer(ParallelSequentialTrainer):
    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        super().__init__(train_env, validation_env, writer, config)
        assert not self.train_at_episode_end

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

        for parent_conn in self.parent_conns:
            env_reset_results = parent_conn.recv()

        while current_step < self.number_of_environment_steps:
            agent_ids = []
            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("next_agent", None))
                next_agent_id = parent_conn.recv()
                agent_ids.append(next_agent_id)

            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("last", None))

            observations = []
            rewards = []
            discounted_rewards = []
            dones = []
            infos = []
            for idx, parent_conn in enumerate(self.parent_conns):
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
                parent_conn.send(("step", actions[action_idx]))
                action_idx += 1

            step_results_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
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
                step_result, env_terminated = parent_conn.recv()
                step_results.append(step_result)
                if env_terminated:
                    envs_terminated_in_current_step.append(idx)
                    current_episode += 1

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
                    previous_observations_for_agent[idx][agent_id] = None
                    previous_actions_for_agent[idx][agent_id] = None
                    previous_additional_information_for_agent[idx][agent_id] = None

                step_results_idx += 1

            # Update envs_not_terminated
            for idx in envs_terminated_in_current_step:
                self.parent_conns[idx].send(("reset", None))

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

            for idx in envs_terminated_in_current_step:
                env_reset_results = self.parent_conns[idx].recv()

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


        t.close()
        self.save_model(current_step)