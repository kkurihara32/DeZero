import unittest

import numpy as np

from dezero import Variable
from steps.stage3.step28 import rosenbrock


class TestStep28(unittest.TestCase):
    def test_check(self):
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))
        lr = 0.001
        iters = 10000

        for i in range(iters):
            print(x0, x1)

            y = rosenbrock(x0, x1)

            x0.clear_grad()
            x1.clear_grad()
            y.backward()

            x0._data -= lr * x0.grad
            x1._data -= lr * x1.grad


if __name__ == '__main__':
    unittest.main()