import glob
import json
from pathlib import Path

from omegaconf import OmegaConf

import main
from datasets.datasets import DataSplit
from trainers.sequential_trainer import SequentialTrainer
from utils.logging.logger import Logger, JSONOutput

#this will append test results to the scalar.json of the run!

metric_to_compare = "validation_advanced_metrics/violation_catched_quota"

#run = "multirun/2021-11-29/17-56-12/+experiment=mardam/mardam,number_of_agents=1"
#4 = 2021-11-29/20-44-33
#6 = 2021-11-30/05-49-52
#run_search = "../multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource=True,agent/model@model=ddqn/grcn,number_of_agents=1/"
#run_search = "multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource=False,agent/model@model=ddqn/grcn,number_of_agents=1/"
agent_range = range(1, 16)

for model in ["grcn"]:#, "large"
    for add_other_agent_targets in [ "False"]:#"True",
        for agent_number in agent_range:
            runs = glob.glob(
                f"multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource={add_other_agent_targets},agent/model@model=ddqn/{model},number_of_agents={agent_number}/")

            if len(runs) == 0:
                print(f"NOT FOUND {model} n_agents:{agent_number}")
                print("")
                continue

            run = sorted(runs)[-1]

            file_path = Path(run)

            with open(file_path / "scalars.json", mode="r") as f:
                lines = f.readlines()
                json_lines = [json.loads(line) for line in lines]
                best_scalars = [(json_line["max_step"],json_line[metric_to_compare]) for json_line in json_lines if metric_to_compare in json_line.keys()]
                best_scalars.sort(key=lambda x: x[1], reverse=True)
                best_step, best_scalar = best_scalars[0]

            # all_saved_models = [(int(file.name.split("_")[-2]),int(file.name.split("_")[-1].replace(".pth",""))) for file in Path(file_path/"models").glob('*.pth')]
            # last_saved_step = max(all_saved_models)[0]
            # best_step = 825000

            cfg = OmegaConf.load(file_path / ".hydra/config.yaml")
            cfg["path_to_model"] = str(Path(file_path/"models").absolute())
            cfg["load_step"] = best_step
            cfg["load_agent_model"] = True
            cfg.render = False
            cfg["calculate_advanced_statistics"] = ["TEST"]
            cfg["create_observation_between_steps"] = False
            cfg["random_start_point"] = False

            main._set_seeds(cfg.seed)

            output_loggers = [
                JSONOutput(log_dir=file_path)
            ]

            writer = Logger(output_loggers)
            event_log, graph, shortest_path_lookup = main.load_data_for_env(cfg)
            test_env = main._initialize_environment(DataSplit.TEST, event_log, graph, shortest_path_lookup, cfg)

            trainer = SequentialTrainer(train_env=test_env, validation_env=test_env, writer=writer, config=cfg)
            test_result = trainer.evaluate(best_step, mode="test", env=test_env)
