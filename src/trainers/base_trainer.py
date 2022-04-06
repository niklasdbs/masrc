from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import numpy as np
import pathlib
from omegaconf import DictConfig

from utils.logging.logger import Logger
from utils.torch.early_stopping import EarlyStopping


class BaseTrainer(ABC):
    def __init__(self, train_env, validation_env, writer: Logger, config: DictConfig):
        self.train_env = train_env
        self.number_of_environment_steps = int(config.number_of_environment_steps)
        self.train_every = config.train_every
        self.train_steps = config.train_steps
        self.train_at_episode_end = config.train_at_episode_end
        self.replay_whole_episodes = config.replay_whole_episodes
        self.start_learning = config.start_learning
        self.use_episode_for_logging = config.use_episode_for_logging
        self.early_stopping = (lambda _ : False) \
            if not config.early_stopping else \
            EarlyStopping(config.early_stopping_patience, config.early_stopping_delta)

        self.load_agent_model = config.load_agent_model
        self.load_step : int = config.load_step

        self.gamma = config.gamma
        self.semi_markov = config.semi_markov
        self.log_metrics_every = config.log_metrics_every
        self.writer: Logger = writer

        self.eval_every = config.eval_every
        self.eval_episodes = config.evaluation_episodes
        self.render = config.render
        self.evaluation_env = validation_env

        self.save_model_every = config.save_every

        save_model_dir = pathlib.Path(config.save_model_dir).expanduser()
        save_model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_path = save_model_dir

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, current_step, mode="validation", env=None) -> float:
        pass

    @abstractmethod
    def save_model(self, current_step: int):
        pass

    def _create_cv_writer(self, env, episode):
        # create OpenCV video writer
        video = cv2.VideoWriter(f'../videos/video_{env.data_split}_{episode}_{env.current_day}.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                1,
                                (env.renderer.render_width, env.renderer.render_height),
                                isColor=True)
        return video

    def _collect_stats_for_day(self, current_episode_data_per_agent, episode_data):
        for agent in current_episode_data_per_agent.keys():
            episode_data[agent]["episode_length"].append(len(current_episode_data_per_agent[agent]))
            episode_data[agent]["return"].append(sum((transition.reward for transition in
                                                      current_episode_data_per_agent[agent])))
            episode_data[agent]["return_discounted"].append(
                    sum((self.gamma ** transition.info["dt"] * transition.reward for transition in
                         current_episode_data_per_agent[agent])))

    def log_evaluation_metrics(self, current_step, env, episode_data, mode):
        self.log_advanced_metrics(current_step, env, mode)
        metrics = defaultdict(list)
        for agent in episode_data.keys():
            for key, values in episode_data[agent].items():
                metrics[key + "_mean"].append(np.mean(values))
                metrics[key + "_median"].append(np.median(values))
                metrics[key + "_sum"].append(np.sum(values))
                metrics[key + "_min"].append(np.min(values))
                metrics[key + "_max"].append(np.max(values))

                self.writer.add_scalar(f"{mode}/{key}_mean_{agent}", np.mean(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_median_{agent}", np.median(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_sum_{agent}", np.sum(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_min_{agent}", np.min(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_max_{agent}", np.max(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_std_{agent}", np.std(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_var_{agent}", np.var(values), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_first_quantile_{agent}", np.quantile(values, 0.25), current_step, epoch=current_step, current_step=current_step)
                self.writer.add_scalar(f"{mode}/{key}_third_quantile_{agent}", np.quantile(values, 0.75), current_step, epoch=current_step, current_step=current_step)
        for key, values in metrics.items():
            self.writer.add_scalar(f"{mode}/{key}_mean", np.mean(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_median", np.median(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_sum", np.sum(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_min", np.min(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_max", np.max(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_std", np.std(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_var", np.var(values), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_first_quantile", np.quantile(values, 0.25), current_step, epoch=current_step, current_step=current_step)
            self.writer.add_scalar(f"{mode}/{key}_third_quantile", np.quantile(values, 0.75), current_step, epoch=current_step, current_step=current_step)
        return metrics

    def log_advanced_metrics(self, current_step, env, mode):
        advanced_metrics = env.final_advanced_metrics
        for key, values in advanced_metrics.items():
            self.writer.add_scalar(f"{mode}_advanced_metrics/{key}", values, global_step=current_step, epoch=current_step, current_step=current_step)

        return advanced_metrics
