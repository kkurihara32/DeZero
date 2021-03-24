import unittest

import numpy as np

from dezero import Variable
from dezero.utils import sum_to
import dezero.functions as F


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


class TestStep41(unittest.TestCase):
    def test_check(self):
        np.random.seed(0)
        x = np.random.randn(100, 1)
        y = 5 + 2 * x + np.random.rand(100, 1)

        x = Variable(x)
        y = Variable(y)

        W = Variable(np.zeros((1, 1)))
        b = Variable(np.zeros(1))

        def predict(x):
            y = F.matmul(x, W) + b
            return y

        print("-"*20)

        lr = 0.1
        iters = 100
        for i in range(iters):
            y_pred = predict(x)
            loss = mean_squared_error(y, y_pred)

            W.clear_grad()
            b.clear_grad()
            loss.backward()

            W.data -= lr * W.grad.data
            b.data -= lr * b.grad.data
            print(W, b, loss)


if __name__ == '__main__':
    unittest.main()


