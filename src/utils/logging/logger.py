import json
import pathlib
import wandb
from typing import Union, Optional, Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, outputs: []):
        self._outputs = outputs

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, epoch: int = -1, current_step: int = -1):
        for output in self._outputs:
            output.add_scalar(tag, scalar_value, global_step, epoch, current_step)

    def add_video(self,
                  tag: str,
                  vid_tensor: Any,
                  global_step: int,
                  fps: Optional[Union[int, float]] = 4):
        for output in self._outputs:
            output.add_scalar(tag, vid_tensor, global_step, fps)

    def add_weight_histogram(self, model, global_step: int, prefix: str = ""):
        for output in self._outputs:
            output.add_weight_histogram(model, global_step, prefix)

    def write(self):
        for output in self._outputs:
            output.write()

    def close(self):
        for output in self._outputs:
            output.close()


class TensorboardOutput:

    def __init__(self, log_dir, comment=""):
        self._writer = SummaryWriter(log_dir, comment)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, epoch: int = -1, current_step: int = -1):
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_video(self,
                  tag: str,
                  vid_tensor: Any,
                  global_step: int,
                  fps: Optional[Union[int, float]] = 4):
        self._writer.add_video(tag, vid_tensor, global_step, fps=fps)

    def add_weight_histogram(self, model, global_step: int, prefix: str = ""):
        for name, weight in model.named_parameters():
            if len(prefix) > 0:
                name = f"{prefix}/{name}"
            self._writer.add_histogram(name, weight, global_step)
            self._writer.add_histogram(f'{name}.grad', weight.grad, global_step)

    def write(self):
        pass

    def close(self):
        self._writer.close()


class JSONOutput:

    def __init__(self, log_dir, excluded_keys=None):
        if excluded_keys is None:
            excluded_keys = []
        self._log_dir = pathlib.Path(log_dir).expanduser()
        self._scalars = []
        self.excluded_keys = excluded_keys  # todo currently excluded keys are checked via contains (that can be slow)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, epoch: int = -1, current_step: int = -1):
        self._scalars.append((global_step, tag, scalar_value))

    def add_video(self,
                  tag: str,
                  vid_tensor: Any,
                  global_step: int,
                  fps: Optional[Union[int, float]] = 4):
        pass  # do not log a video file to json

    def _filter_excluded(self, tag: str) -> bool:
        return any(tag.__contains__(key) for key in self.excluded_keys)

    def add_weight_histogram(self, model, global_step: int, prefix: str = ""):
        pass  # do not log weight histograms to json

    def write(self):
        if len(self._scalars) == 0:
            return

        values = {tag: float(scalar) for _, tag, scalar in self._scalars if not self._filter_excluded(tag)}

        max_step = max(step for step, _, _ in self._scalars)

        with (self._log_dir / 'scalars.json').open('a') as json_file:
            json.dump({"max_step": max_step, **values}, fp=json_file)
            json_file.write("\n")

        self._scalars.clear()

    def close(self):
        pass


class WANDBLogger:
    def __init__(self, config: DictConfig):
        run = wandb.init(entity="nst",
                         project="mtop",
                         group=config.experiment_name,
                         tags=[
                                 f"{config.number_of_agents}_agents",
                                 config.observation,
                                 config.experiment_name if config.get("model", None) is None else config.model.name,
                                 config.area_name,
                                 config.trainer,
                                 config.agent,
                                 config.experiment_name
                         ],
                         config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True))

        run.log_code(to_absolute_path("modules/"))

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
