from dataclasses import dataclass
from typing import Any

from envs.enums import ParkingStatus


@dataclass
class Resource:
    ident: int
    status: ParkingStatus
    source_node: Any
    target_node: Any
    position_on_edge : int
    arrival_time: int
    max_parking_duration_seconds: int
    time_last_violation : int #this field is only valid if the parkingstatus is inviolation or fined
    x : float
    y : float
