import unittest

import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


class TestStep38(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward()
        print(x.grad)
        x = Variable(np.random.randn(1, 2, 3))
        y = x.reshape(2, 3)
        y2 = x.reshape((2, 3))
        print(y)
        print(y2)
        print("-"*20)
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        y.backward(retain_grad=True)
        print(y.grad)
        print(x.grad)
        print("-"*20)
        x = Variable(np.random.rand(2, 3))
        y = x.transpose()
        y2 = x.T
        print(y)
        print(y2)


if __name__ == '__main__':
    unittest.main()


