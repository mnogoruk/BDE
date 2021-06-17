import numpy as np

from RungeKutta import RungeKutta
from abstract import AbstractODE


class Adams(AbstractODE):
    INITIAL_BDE = RungeKutta
    H = 0.1
    N = 100

    def set_initial_bde(self, bde_class):
        self.INITIAL_BDE = bde_class

    def set_step(self, h):
        self.H = h

    def set_step_count(self, n):
        self.N = n

    def find_solution(self):
        f = self.func
        init_x = self.init_x
        init_y = self.init_y
        h = self.H
        n = self.N

        init_step_count = 4

        # better to use fabric template. But this is python))
        init_bde = self.INITIAL_BDE(f, init_x, init_y)
        init_bde.set_step(h)
        init_bde.set_step_count(init_step_count)

        _, init_solution_y = init_bde.find_solution()
        y = np.zeros(n)
        x = np.arange(init_x, init_x + n * h, h)
        for i in range(init_step_count):
            y[i] = init_solution_y[i]
        for i in range(init_step_count-1, n - 1):
            qi = h * f(x[i], y[i])
            delta_1_qi_1 = h * (f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
            delta_2_qi_2 = h * (f(x[i], y[i]) - 2 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))
            delta_3_qi_3 = h * (
                        f(x[i], y[i]) - 3 * f(x[i - 1], y[i - 1]) + 3 * f(x[i - 2], y[i - 2]) - f(x[i - 3], y[i - 3]))
            y[i + 1] = y[i] + qi + 1/2 * delta_1_qi_1 + 5/12 * delta_2_qi_2 + 3/8 * delta_3_qi_3
        self.solution = [x, y]
        return self.solution
