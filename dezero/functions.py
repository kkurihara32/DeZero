import numpy as np

from dezero.core import Function, Variable


class Sin(Function):
    def forward(self, x: np.ndarray):
        y = np.sin(x)
        return y

    def backward(self, gy: Variable):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray):
        y = np.cos(x)
        return y

    def backward(self, gy: Variable):
        x, = self.inputs
        gx = gy * (-sin(x))
        return gx


def cos(x):
    return Cos()(x)
