from dataclasses import dataclass
from typing import Any


@dataclass
class CLDETransition:
    local_observations : Any
    global_observation: Any
    actions : Any
    action_mask : Any
    rewards : Any
    dones : Any
    infos : Any
