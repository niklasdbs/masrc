import pandas as pd
from hydra.utils import to_absolute_path

from omegaconf import DictConfig


class EventLogLoader:

    def __init__(self, graph, config : DictConfig):
        self.path_to_event_log = to_absolute_path(config.path_to_event_log)

        spots = []

        for _, _, data in graph.edges(data=True):
            if "spots" in data:
                for spot in data["spots"]:
                    spots.append(spot["id"])

        self.spots = set(spots)
        self.small_event_log = config.small_event_log


    def load(self) -> pd.DataFrame:
        event_log = pd.read_pickle(self.path_to_event_log, compression="gzip")
        event_log = event_log[event_log["StreetMarker"].isin(self.spots)]

        if self.small_event_log:
            event_log = event_log[event_log["Time"].dt.isocalendar().week <= 2]

        return event_log

