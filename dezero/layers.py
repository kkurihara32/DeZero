import weakref

import numpy as np

import dezero.functions as F
from dezero.core import Parameter


class Layer(object):
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self._inputs = [weakref.ref(x) for x in inputs]
        self._outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()


class Linear(Layer):
    def __init__(self, out_size, no_bias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self._in_size = in_size
        self._out_size = out_size
        self._dtype = dtype

        self._W = Parameter(None, name="W")
        if self._in_size is not None:
            self._init_W()

        if no_bias:
            self._b = None
        else:
            self._b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self):
        I, O = self._in_size, self._out_size
        W_data = np.random.randn(I, O).astype(self._dtype) * np.sqrt(1 / I)
        self._W.data = W_data

    def forward(self, x):
        if self._W.data is None:
            self._in_size = x.shape[1]
            self._init_W()
        y = F.linear(x, self._W, self._b)
        return y
