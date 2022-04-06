from dataclasses import dataclass
from typing import Any

@dataclass
class Transition:
    state : Any
    action : Any
    next_state : Any
    reward : float
    done : Any
    info : Any
    additional_information : Any = None