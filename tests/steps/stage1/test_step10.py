import unittest

import numpy as np

from steps.stage1.step09 import Variable, square


def _numerical_diff(f, x: Variable, eps=1e-4):
    x1 = Variable(x.data + eps)
    x2 = Variable(x.data - eps)
    y1 = f(x1)
    y2 = f(x2)
    return (y1.data - y2.data) / (2 * eps)


class TestStep10(unittest.TestCase):
    def test_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = _numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


if __name__ == '__main__':
    unittest.main()
