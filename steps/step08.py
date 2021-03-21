from abc import ABCMeta, abstractmethod

import numpy as np


class Variable(object):
    def __init__(self, data: np.ndarray):
        self._data = data
        self._grad = None
        self._creator = None

    def set_creator(self, func):
        self._creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def creator(self):
        return self._creator

    @grad.setter
    def grad(self, value):
        self._grad = value


class Function(object, metaclass=ABCMeta):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 出力変数に生みの親の関数を覚えさせる
        self._input = input
        self._output = output
        return output

    @abstractmethod
    def forward(self, x: np.array):
        raise NotImplementedError

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output


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