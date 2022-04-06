import glob
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import main
from datasets.datasets import DataSplit


@hydra.main(config_path="../../config", config_name="config")
def run(cfg : DictConfig):
    cfg.render = True
    cfg["render_resolution"] = DictConfig({"w": 1800, "h" : 1800, "dpi" : 100.0})
    cfg["render_steps_of_actions"] = True
    cfg["calculate_advanced_statistics"] = ["TEST"]
    cfg["create_observation_between_steps"] = True
    cfg["random_start_point"] = False

    main._set_seeds(cfg.seed)
    event_log, graph, shortest_path_lookup = main.load_data_for_env(cfg)
    test_env = main._initialize_environment(DataSplit.TRAINING, event_log, graph, shortest_path_lookup, cfg)

    test_env._render_steps_of_action = True

    test_env.reset()
    for _ in range(50):
        test_env.step(0)

    imgs = test_env.render(show=False)
    # color = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
    # plt.imshow(color, aspect="auto")
    # plt.show()

    cv2.imwrite(to_absolute_path("../images/graph.png"), img=imgs[-1])

    a= 3


if __name__ == '__main__':
    run()