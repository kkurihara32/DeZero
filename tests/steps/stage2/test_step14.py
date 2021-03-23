import unittest

import numpy as np

from steps.stage2.step14 import Variable, add


class TestStep14(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(3))
        y = add(x, x)
        y.backward()
        expected = 2
        actual = x.grad
        self.assertEqual(expected, actual)

        x.clear_grad()
        y = add(add(x, x), x)
        y.backward()
        expected = 3
        actual = x.grad
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
