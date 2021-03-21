from abc import ABCMeta, abstractmethod

import numpy as np

from steps.step01 import Variable


class Function(object, metaclass=ABCMeta):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)

        return output

    @abstractmethod
    def forward(self, x: np.array):
         raise NotImplementedError


class Square(Function):
    def forward(self, x: np.array):
        return x ** 2
