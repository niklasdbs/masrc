import glob
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

import main
from datasets.datasets import DataSplit
from trainers.sequential_trainer import SequentialTrainer
from utils.logging.logger import Logger

#run = "multirun/2021-11-29/17-56-12/+experiment=mardam/mardam,number_of_agents=1"
#4 = 2021-11-29/20-44-33
#6 = 2021-11-30/05-49-52

#search = "multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource=True,agent/model@model=ddqn/grcn,number_of_agents=10/"
#search = "multirun/*/*/+experiment=greedy/greedy_with_wait,number_of_agents=10/"
search = "multirun/**/*2quvheds/"

run = sorted(glob.glob(search, recursive=True))[-1]

file_path = Path(run)/"../../"
all_saved_models = [(int(file.name.split("_")[-2]  ),int(file.name.split("_")[-1].replace(".pth",""))) for file in Path(file_path/"models").glob('*.pth')]
last_saved_step = max(all_saved_models)[0]


cfg = OmegaConf.load(file_path / ".hydra/config.yaml")
cfg["path_to_model"] = str(Path(file_path/"models").absolute())
cfg["load_step"] = last_saved_step
cfg["load_agent_model"] = True
cfg.render = True
cfg["render_resolution"] = DictConfig({"w": 1800, "h" : 1800, "dpi" : 100.0})
cfg["render_steps_of_actions"] = True
cfg["calculate_advanced_statistics"] = ["TEST"]
cfg["create_observation_between_steps"] = False
cfg["random_start_point"] = False

main._set_seeds(cfg.seed)

output_loggers = []

writer = Logger(output_loggers)
event_log, graph, shortest_path_lookup = main.load_data_for_env(cfg)
test_env = main._initialize_environment(DataSplit.TEST, event_log, graph, shortest_path_lookup, cfg)

trainer = SequentialTrainer(train_env=test_env, validation_env=test_env, writer=writer, config=cfg)
test_result = trainer.evaluate(0, mode="test", env=test_env)
