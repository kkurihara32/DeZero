from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import numpy as np


class Variable(object):
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported type".format(type(data)))
        self._data = data
        self._grad = None
        self._creator = None

    def set_creator(self, func):
        self._creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self._inputs = inputs
        self._outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self, gys: List[np.array]):
        raise NotImplementedError

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 + x1
        return y

    def backward(self, gys: List[np.array]):
        pass


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    return Add()(x0, x1)
