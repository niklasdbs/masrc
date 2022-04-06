from dataclasses import dataclass
from typing import Any, Union, List, Optional


@dataclass
class AgentState:
    agent_id : int
    position_node: Any
    #(position_node_source, position_node) identifies the edge that the agent is currently located
    position_node_source: Any
    #the position on the edge denoted in meters from the start of the edge
    position_on_edge: Optional[int] = None
    current_action: Optional[int] = None
    current_route: Optional[List[Any]] = None
    current_action_complete = False
    time_last_action: int = 0
    reward_since_last_action: float = 0.0
    discounted_reward_since_last_action: float = 0.0

    # todo add agent statistics for the action here

    def reset_after_current_action_completed(self):
        self.time_last_action = 0
        self.current_action_complete = False
        self.current_action = None
        self.current_route = None
        self.reward_since_last_action = 0.0
        self.discounted_reward_since_last_action = 0.0
