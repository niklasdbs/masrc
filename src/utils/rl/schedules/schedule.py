from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def step(self, n_steps: int = 1):
        pass
