import math

import numpy as np

from dezero import Function


class Sin(Function):
    def forward(self, x: np.array):
        y = np.sin(x)
        return y

    def backward(self, gy: np.array):
        x = self.inputs[0].data
        gx = np.cos(x) * gy
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y
