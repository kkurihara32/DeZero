import numpy as np

from steps.step02 import Function


class EXP(Function):
    def forward(self, x: np.array):
        return np.exp(x)
