from typing import Tuple

import numpy as np

from dezero.core import Function, Variable, as_variable
from dezero import utils


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


class Reshape(Function):
    def __init__(self, shape):
        self._shape = shape

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.reshape(self._shape)
        return y

    def backward(self, gy: Variable):
        return reshape(gy, self.x_shape)


def reshape(x, shape: Tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x: np.ndarray):
        y = np.transpose(x)
        return y

    def backward(self, gy: Variable):
        gx = transpose(gy)
        return gx


def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self._axis = axis
        self._keepdims = keepdims

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.sum(axis=self._axis, keepdims=self._keepdims)
        return y

    def backward(self, gy: Variable):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self._axis, self._keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

    @property
    def axis(self):
        return self._axis

    @property
    def keepdims(self):
        return self._keepdims


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self._shape = shape
        print("b:{}".format(self._shape))

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self._shape)
        return y

    def backward(self, gy: Variable):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self._shape = shape

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = utils.sum_to(x, self._shape)
        return y

    def backward(self, gy: Variable):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
