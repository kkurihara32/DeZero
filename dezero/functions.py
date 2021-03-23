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


class Tanh(Function):
    def forward(self, x: np.ndarray):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


def cos(x):
    return Cos()(x)
