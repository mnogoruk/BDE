import numpy as np

from RungeKutta import RungeKutta
from abstract import AbstractODE


class ForecastingAndCorrection(AbstractODE):
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
        init_step_count = 4

        # better to use fabric template. But this is python))
        init_bde = self.INITIAL_BDE(self.func, self.init_x, self.init_y)
        init_bde.set_step(self.H)
        init_bde.set_step_count(init_step_count)

        _, init_solution_y = init_bde.find_solution()

        y = np.zeros(self.N)
        x = np.arange(self.init_x, self.init_x + self.N * self.H, self.H)

        for i in range(init_step_count):
            y[i] = init_solution_y[i]

        for i in range(init_step_count-1, self.N - 1):
            forecasted_y = self._forecasting(x, y, i)
            y[i + 1] = self._correction(x, y, i, forecasted_y)

        self.solution = [x, y]
        return self.solution

    def _forecasting(self, x, y, i):
        f = self.func
        return y[i] + self.H / 24 * (
                55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3]))

    def _correction(self, x, y, i, forecasted_y):
        f = self.func
        return y[i] + self.H / 24 * (
                    9 * f(x[i+1], forecasted_y) + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))
