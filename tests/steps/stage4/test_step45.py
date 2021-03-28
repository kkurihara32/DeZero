import unittest

import numpy as np

from dezero import Variable, Parameter, Model
from dezero.models import MLP
from dezero.utils import sum_to
import dezero.functions as F
import dezero.layers as L
from dezero import Layer


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, inputs):
        y = F.sigmoid(self.l1(inputs))
        y = self.l2(y)
        return y


class TestStep45(unittest.TestCase):
    def test_check(self):

        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        model = MLP((10, 20, 30, 40, 1))

        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y, y_pred)

            model.clear_grads()
            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad.data

            if i % 1000 == 0:
                print(loss)


if __name__ == '__main__':
    unittest.main()


