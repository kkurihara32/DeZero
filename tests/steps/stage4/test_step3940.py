import unittest

import numpy as np

from dezero import Variable
from dezero.utils import sum_to
import dezero.functions as F


class TestStep3940(unittest.TestCase):
    def test_check(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = sum_to(x, (1, 3))
        print(y)

        y = sum_to(x, (2, 1))
        print(y)
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        print(y)
        print("-"*20)
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        print("-" * 20)
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 - x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        print("-" * 20)
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        print("-" * 20)
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 / x1
        y.backward()
        print(x0.grad)
        print(x1.grad)




if __name__ == '__main__':
    unittest.main()


