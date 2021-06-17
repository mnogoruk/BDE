from abc import ABC, abstractmethod


class AbstractODE(ABC):

    def __init__(self, func, init_x, init_y):
        self.func = func
        self.init_x = init_x
        self.init_y = init_y
        self.solution = None

    @abstractmethod
    def set_step(self, h):
        pass

    @abstractmethod
    def set_step_count(self, n):
        pass

    @abstractmethod
    def find_solution(self):
        pass
