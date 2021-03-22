from abc import ABCMeta, abstractmethod
import contextlib
import weakref

import numpy as np


class Variable(object):
    def __init__(self, data: np.ndarray, name: str=None):
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
        p = str(self._data).replace("\n", "\n" + " " + " "*8)
        return "variable(" + p + ")"

    def set_creator(self, func):
        self._creator = func
        self._generation = func.generation + 1

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
                gxs = (gxs, )

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

    def clear_grad(self):
        self._grad = None

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


class Function(object, metaclass=ABCMeta):
    def __call__(self, *inputs: Variable):
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

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

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


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray):
        return gy, gy


class Square(Function):
    def forward(self, x: np.array):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def add(x0, x1):
    return Add()(x0, x1)


def square(x0):
    return Square()(x0)


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value =getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)

