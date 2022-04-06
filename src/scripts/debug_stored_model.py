import glob
import json
from pathlib import Path

from omegaconf import OmegaConf

import main
from datasets.datasets import DataSplit
from trainers.sequential_trainer import SequentialTrainer
from utils.logging.logger import Logger, JSONOutput

# metric_to_compare = "validation_advanced_metrics/violation_catched_quota"

search = "multirun/**/*1o3x52j1/"

run = sorted(glob.glob(search, recursive=True))[-1]

file_path = Path(run)/"../../"
# with open(file_path / "scalars.json", mode="r") as f:
#     lines = f.readlines()
#     json_lines = [json.loads(line) for line in lines]
#     best_scalars = [(json_line["max_step"],json_line[metric_to_compare]) for json_line in json_lines if metric_to_compare in json_line.keys()]
#     best_scalars.sort(key=lambda x: x[1], reverse=True)
#     best_step, best_scalar = best_scalars[0]

all_saved_models = [(int(file.name.split("_")[-2]),int(file.name.split("_")[-1].replace(".pth",""))) for file in Path(file_path/"models").glob('*.pth')]
last_saved_step = max(all_saved_models)[0]
step_to_use = last_saved_step
#step_to_use = 4000

cfg = OmegaConf.load(file_path / ".hydra/config.yaml")
cfg["path_to_model"] = str(Path(file_path/"models").absolute())
cfg["load_step"] = step_to_use
cfg["load_agent_model"] = True
cfg.render = False
cfg["calculate_advanced_statistics"] = ["VALIDATION"]
cfg["device"] = "cpu"
#cfg["create_observation_between_steps"] = False
#cfg["random_start_point"] = False

main._set_seeds(cfg.seed)

output_loggers = [
    #JSONOutput(log_dir=file_path)
]

writer = Logger(output_loggers)
event_log, graph, shortest_path_lookup = main.load_data_for_env(cfg)
test_env = main._initialize_environment(DataSplit.TRAINING, event_log, graph, shortest_path_lookup, cfg)

#test_env._render_steps_of_action = True

trainer = SequentialTrainer(train_env=test_env, validation_env=test_env, writer=writer, config=cfg)
test_result = trainer.evaluate(-1, mode="test", env=test_env)
