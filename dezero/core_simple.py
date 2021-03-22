import contextlib
import weakref
from abc import abstractmethod, ABCMeta

import numpy as np


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable(object):
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported type".format(type(data)))
        self._data = data
        self._grad = None
        self._creator = None
        self._generation = 0
        self._name = name

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        if self._data is None:
            return "variable(None)"
        p = str(self._data).replace("\n", "\n" + " " + " " * 8)
        return "variable(" + p + ")"

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def creator(self):
        return self._creator

    @property
    def generation(self):
        return self._generation

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    @grad.setter
    def grad(self, value):
        self._grad = value

    @generation.setter
    def generation(self, value):
        self._generation = value

    @name.setter
    def name(self, value):
        self._name = value

    def set_creator(self, func):
        self._creator = func
        self._generation = func.generation + 1

    def clear_grad(self):
        self._grad = None

    def backward(self, retain_grad: bool = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = list()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda a: a.generation)

        funcs.append(self._creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function(object, metaclass=ABCMeta):
    def __call__(self, *inputs: Variable):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self._inputs = None
        self._generation = None
        self._outputs = None

        if Config.enable_backprop:
            self._generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self._inputs = inputs
            self._outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, value):
        self._generation = value

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x: np.array):
        y = x ** 2
        return y

    def backward(self, gy: np.ndarray):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x0):
    return Square()(x0)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return x

    def backward(self, gy: np.ndarray):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        return x0 / x1

    def backward(self, gy: np.ndarray):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data

        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1**2))

        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self._c = c

    def forward(self, x: np.ndarray):
        return x ** self._c

    def backward(self, gy: np.ndarray):
        x = self.inputs[0].data
        c = self._c
        gx = c * x ** (c - 1) * gy

        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
