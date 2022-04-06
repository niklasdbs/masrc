
class top_agent_selector():
    def __init__(self, initial_order):
        self.initial_order = initial_order
        self._current_agent = 0
        self.selected_agent = 0

        self._next_agents = initial_order[:]



    def set_agent_needs_to_select_new_action(self, agent):
        assert agent not in self._next_agents
        assert agent >= 0
        self._next_agents.append(agent)


    def set_next_agents(self, agents):
        self._next_agents = agents

    def next(self):
        self._current_agent = self.selected_agent = self._next_agents.pop(0)
        return self.selected_agent

    def is_last(self):
        return len(self._next_agents) == 0

    def is_first(self):#todo
        return  (len(self._next_agents)> 0 and self.selected_agent == self._next_agents[0]) or \
                (len(self._next_agents) == 0 and self.selected_agent == self.initial_order[0])
