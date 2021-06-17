from abstract import AbstractODE
import numpy as np


class RungeKutta(AbstractODE):
    H = 0.1
    N = 100

    def set_step(self, h):
        self.H = h

    def set_step_count(self, n):
        self.N = n

    def find_solution(self):
        x = np.arange(self.init_x, self.init_x + self.N * self.H, self.H)
        y = np.zeros(self.N)

        y[0] = self.init_y

        f = self.func
        h = self.H
        for i in range(self.N - 1):
            k0 = f(x[i], y[i])
            k1 = f(x[i] + h / 2, y[i] + h / 2 * k0)
            k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
            k3 = f(x[i] + h, y[i] + h * k2)
            y[i + 1] = y[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

        self.solution = [x, y]
        return self.solution
