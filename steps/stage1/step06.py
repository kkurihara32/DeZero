from abc import ABCMeta, abstractmethod

import numpy as np

from steps.stage1.step01 import Variable


class Variable(object):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None


class Function(object, metaclass=ABCMeta):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self._input = input
        return output

    @abstractmethod
    def forward(self, x: np.array):
         raise NotImplementedError

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x: np.array):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self._input.data
        gx = 2 * x * gy
        return gx


class EXP(Function):
    def forward(self, x: np.array):
        return np.exp(x)

    def backward(self, gy):
        x = self._input.data
        gx = np.exp(x) * gy
        return gx
