from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Union, Optional

from envs.enums import EventType


@dataclass(order=False)
@total_ordering
class AgentEvent:
    time : int
    position_node : Any
    position_node_source : Any
    position_on_edge: Optional[int]
    agent : int
    completes_action : bool = False  #indicates that an agent needs to choose an action after this event

    def __lt__(self, other):
        if isinstance(other, ResourceEvent) and self.time == other.time:
            return True # ensure that agent comes before resource if they happen at the same time

        return self.time < other.time


@dataclass(order=False, frozen=True)
@total_ordering
class ResourceEvent:
    time : int
    max_seconds : int
    resource_id : int
    event_type : EventType

    def __lt__(self, other):
        if isinstance(other, AgentEvent) and self.time == other.time:
            return False #ensure that agent comes before resource if they happen at the same time

        return self.time < other.time

