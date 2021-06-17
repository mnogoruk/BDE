import math

import numpy as np

from Admas import Adams
from ForecastingAndCorrection import ForecastingAndCorrection
from RungeKutta import RungeKutta
import matplotlib.pyplot as plt


def runge_kutta(func, init_x, init_y):
    bde = RungeKutta(func, init_x, init_y)
    bde.set_step(0.01)
    x, y = bde.find_solution()

    plt.plot(x, y)
    plt.show()


def adamas(func, init_x, init_y):
    bde = Adams(func, init_x, init_y)
    bde.set_step(0.01)
    x, y = bde.find_solution()

    plt.plot(x, y)
    plt.show()


def forecasting_and_correction(func, init_x, init_y):
    bde = ForecastingAndCorrection(func, init_x, init_y)
    bde.set_step(0.01)
    x, y = bde.find_solution()

    plt.plot(x, y)
    plt.show()


def main():
    def func(x, y):
        return x * y

    init_x = 0
    init_y = 1

    runge_kutta(func, init_x, init_y)
    adamas(func, init_x, init_y)
    forecasting_and_correction(func, init_x, init_y)

    def answer(xi):
        return math.e ** (xi ** 2 / 2)

    x = np.arange(0, 1, 0.01)
    y = [answer(xi) for xi in x]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
