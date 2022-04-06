import logging
import os
import random
import sys
from functools import partialmethod, partial

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

import trainers
from datasets.datasets import DataSplit
from datasets.event_log_loader import EventLogLoader
from envs.potop_env import PotopEnv
from envs.utils import load_graph_from_file, precompute_shortest_paths
from envs.wrapper.parallel_env_wrapper import ToParallelWrapper
from graph.graph_helpers import create_graph
from trainers.base_trainer import BaseTrainer
from utils.logging.logger import Logger, JSONOutput, WANDBLogger


def simulate(config: DictConfig) -> float:
    """Performs a simulation of the environment."""
    logging.info('Starting simulation.')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    event_log, graph, shortest_path_lookup = load_data_for_env(config)

    env_partial = config.trainer in ["ParallelSequentialTrainer", "ParallelSequentialAsyncResetTrainer"]#todo
    if env_partial:
        train_env = partial(_initialize_environment, DataSplit.TRAINING, event_log, graph, shortest_path_lookup, config)
    else:
        train_env = _initialize_environment(DataSplit.TRAINING, event_log, graph, shortest_path_lookup, config)
    val_env = _initialize_environment(DataSplit.VALIDATION, event_log, graph, shortest_path_lookup, config)
    test_env = _initialize_environment(DataSplit.TEST, event_log, graph, shortest_path_lookup, config)


    output_loggers = [
        #TensorboardOutput(log_dir=".", comment=f""),
        JSONOutput(log_dir=os.getcwd())
    ]

    if not config.experiment_name == "debug":
        output_loggers.append(WANDBLogger(config=config))

    writer = Logger(output_loggers)
    trainer : BaseTrainer = getattr(trainers, config.trainer)(train_env=train_env,
                                                              validation_env=val_env,
                                                              writer=writer,
                                                              config=config)


    trainer.train()

    test_result = trainer.evaluate(0, mode="test", env=test_env)
    test_result = trainer.evaluate(0, mode="validation", env=val_env)

    logging.info("Simulation finished.")

    if not env_partial:
        train_env.close()
    test_env.close()
    val_env.close()

    writer.close()

    return test_result

def _set_seeds(seed: int):
    logging.info('Setting seed: "%s"', seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _initialize_environment(datasplit: DataSplit, event_log, graph, shortest_path_lookup, config):
    logging.info('Initializing environment.')

    env = PotopEnv(
        event_log=event_log,
        graph=graph,
        shortest_path_lookup=shortest_path_lookup,
        data_split=datasplit,
        config=config)

    if config.parallel_env:
        env = ToParallelWrapper(env)

    return env


def load_data_for_env(config: DictConfig):
    graph_file_name = os.path.join(to_absolute_path(config.path_to_graphs), f"{''.join(config.area).lower()}.gpickle")
    if not os.path.exists(graph_file_name):
        # Creates a new graph
        os.makedirs(os.path.dirname(graph_file_name), exist_ok=True)
        create_graph(config, filename=graph_file_name)
    graph = load_graph_from_file(graph_file_name)
    # Lookup table for the shortest path between two nodes.
    shortest_path_lookup = precompute_shortest_paths(graph)

    event_log = EventLogLoader(graph, config).load()
    return event_log, graph, shortest_path_lookup



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


@hydra.main(config_path="config", config_name="config")
def run(config : DictConfig):
    #logging is configured by hydra

    #exceptions will be loggegd to hydra log file
    sys.excepthook = handle_exception

    logging.debug(torch.cuda.is_available())

    #tdqm should not spam error logs
    tqdm.__init__ = partialmethod(tqdm.__init__, file=sys.stdout)

    _set_seeds(config.seed)

    return simulate(config)


if __name__ == '__main__':
    run()