import unittest

import numpy as np

from steps.step20 import Variable, square, add, Config, using_config, no_grad


class TestStep19(unittest.TestCase):
    def test_check(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = a * b + c
        y.backward()
        self.assertEqual(y.data, 7)
        self.assertEqual(a.grad, 2)
        self.assertEqual(b.grad, 3)
        self.assertEqual(y.grad, None)


if __name__ == '__main__':
    unittest.main()
