import itertools

import glob
import sys
from pathlib import Path
from typing import Any, Optional, Union

import wandb
from omegaconf import OmegaConf
from main import load_data_for_env, _initialize_environment, _set_seeds

from datasets.datasets import DataSplit
from scripts.run_finder_wandb import filter_runs, find_best_run_in_runs, find_bests_of_its_kind, scan_history_for_metric
from trainers import SequentialTrainer
from utils.logging.logger import JSONOutput, Logger

relevant_metric = "validation_advanced_metrics/fined_resources"  # violation_catched_quota fined_resources

areas = ["docklands", "queensberry", "downtown"]
number_of_agents = [1, 2,4,8] #2,4,8
kinds = ["grcn_twin", "shared_ind", "mardam_no_oaf"]

api = wandb.Api(timeout=60)


class WANDBUpdateLogger:
    def __init__(self, run):
        self.run = run
        wandb.init(project="mtop", resume="must", id=run.id)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, epoch: int = -1, current_step: int = -1):
        wandb.log({tag: scalar_value, "step": global_step, "epoch": epoch, "current_step": current_step}, commit=False)

    def add_video(self,
                  tag: str,
                  vid_tensor: Any,
                  global_step: int,
                  fps: Optional[Union[int, float]] = 4):
        pass

    def add_weight_histogram(self, model, global_step: int, prefix: str = ""):
        pass

    def write(self):
        wandb.log({"save": "save"}, commit=True)

    def close(self):
        wandb.log({"save": "final"}, commit=True)
        wandb.finish()


def find_path_for_run(run):
    search = f"multirun/**/*{run.id}/"
    runs = sorted(glob.glob(search, recursive=True))
    if len(runs) == 0:
        print(f"WARNING: could not find run {run.name} with id {run.id}", file=sys.stderr)
        return None

    run = runs[-1]

    file_path = Path(run) / "../../"
    if not file_path.exists():
        print(f"WARNING: could not find run {run.name} with id {run.id}", file=sys.stderr)
        return None

    return file_path


def evaluate_run(run):
    if "test_eval_final" in run.config:
        print(f"WARNING: already evaluated skip run {run.name} with id {run.id}", file=sys.stderr)
        return

    if run.state == 'running' and False:#todo
        print(f"WARNING: still running skip run {run.name} with id {run.id}", file=sys.stderr)
        return

    path = find_path_for_run(run)

    if path is None:
        return

    _, full_row_best_run = scan_history_for_metric(run, relevant_metric, only_fetch_relevant_metric=False)

    best_step = full_row_best_run["step"]

    if best_step <= 0:
        all_saved_models = [(int(file.name.split("_")[-2]), int(file.name.split("_")[-1].replace(".pth", ""))) for file
                            in Path(path / "models").glob('*.pth')]
        last_saved_step = max(all_saved_models)[0]
        best_step = last_saved_step

    cfg = OmegaConf.load(path / ".hydra/config.yaml")
    cfg["path_to_model"] = str(Path(path / "models").absolute())
    cfg["load_step"] = best_step
    cfg["load_agent_model"] = True
    _set_seeds(cfg.seed)

    wandb_logger = WANDBUpdateLogger(run)
    json_logger = JSONOutput(log_dir=path)
    output_loggers = [
            json_logger,
            wandb_logger
    ]

    writer = Logger(output_loggers)
    event_log, graph, shortest_path_lookup = load_data_for_env(cfg)
    test_env = _initialize_environment(DataSplit.TEST, event_log, graph, shortest_path_lookup, cfg)

    trainer = SequentialTrainer(train_env=test_env, validation_env=test_env, writer=writer, config=cfg)
    test_result = trainer.evaluate(best_step, mode="test_final", env=test_env)

    wandb_logger.close()
    json_logger.close()
    test_env.close()
    run_for_update = api.run("/".join(run.path))
    run_for_update.config["test_eval_final"] = True
    run_for_update.update()

def execute():
    for area, n_agents in itertools.product(areas, number_of_agents):
        runs = filter_runs(api=api, area=area, number_of_agents=n_agents)
        _, _, _, runs_and_best_row_sorted = find_best_run_in_runs(runs, relevant_metric)

        bests_by_its_kind = find_bests_of_its_kind(runs_and_best_row_sorted)

        for kind in kinds:
            if kind not in bests_by_its_kind:
                print(f"WARNING: {kind} not found in runs for area: {area}, n_agents: {n_agents}", file=sys.stderr)
                continue

            best_metric, row_with_best_metric, run = bests_by_its_kind[kind][0]
            # run = api.run("nst/mtop/17ts4d07")#todo for debug

            try:
                evaluate_run(run)
            except Exception as e:
                print(f"Exception run {run.name} with id {run.id}: {e}", file=sys.stderr)

def execute_single(run_path : str):
    run = api.run(run_path)
    evaluate_run(run)

if __name__ == '__main__':
    execute()

#ablations
#shared reward docklands 2 nst/mtop/12ze8969
#no agent emb nst/mtop/28bxr93o
#no action emb nst/mtop/5k5tte0s
#no dist nst/mtop/20i15db3
#no xy nst/mtop/389x3k7q
#no agent id nst/mtop/2c4l3lpv
#add tar nst/mtop/f0jswclq
#shared reward 4 nst/mtop/3iif217d

    # ablations = ["nst/mtop/12ze8969",
    #              "nst/mtop/28bxr93o" ,
    #              "nst/mtop/5k5tte0s",
    #              "nst/mtop/20i15db3",
    #              "nst/mtop/389x3k7q",
    #              "nst/mtop/2c4l3lpv",
    #              "nst/mtop/f0jswclq",
    #              "nst/mtop/3iif217d"
    #              ]
    #
    # for r_path in ablations:
    #     execute_single(run_path=r_path)