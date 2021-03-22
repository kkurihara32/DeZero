import unittest

import numpy as np

from steps.step18 import Variable, square, add, Config, using_config, no_grad
from memory_profiler import profile


@profile
def test_1():
    with using_config("enable_backprop", True):
        x = Variable(np.ones((1000, 1000, 100)))
        y = square(square(square(x)))
        print("end1")
        y.backward()

    return 0


@profile
def test_2():
    with no_grad():
        x = Variable(np.ones((1000, 1000, 100)))
        y = square(square(square(x)))
    print("end2")

    return 0


class TestStep18(unittest.TestCase):
    def test_check(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()

        self.assertTrue(y.grad is None)
        self.assertTrue(t.grad is None)
        self.assertTrue(x1.grad == 1)
        self.assertTrue(x0.grad == 2)
        test_1()
        test_2()


if __name__ == '__main__':
    unittest.main()
