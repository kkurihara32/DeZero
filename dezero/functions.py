from typing import Tuple

import numpy as np

import dezero
from dezero.core import Function, Variable, as_variable
from dezero import utils, cuda


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


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


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


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray):
        y = np.dot(x, W)
        return y

    def backward(self, gy: Variable):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W) -> Variable:
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy: Variable):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray):
        y = np.dot(x, W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)

        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


# class ReLU(Function):
#     def forward(self, x):
#         xp = cuda.get_array_module(x)
#         y = xp.maximum(x, 0.0)
#         return y
#
#     def backward(self, gy):
#         x, = self.inputs
#         mask = x.data > 0
#         gx = gy * mask
#         return gx
#
#
# def relu(x):
#     return ReLU()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)
