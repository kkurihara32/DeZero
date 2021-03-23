import unittest

import numpy as np

from steps.stage1.step09 import Variable, square, exp


class TestStep09(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(0.5))
        y = square(exp(square(x)))
        y.backward()
        print(x.grad)
        self.assertTrue(3.29 < x.grad < 3.3)

    def test_exception(self):
        x = Variable(np.array(1.0))
        x = Variable(None)

        with self.assertRaises(TypeError):
            Variable(1.0)


if __name__ == '__main__':
    unittest.main()
