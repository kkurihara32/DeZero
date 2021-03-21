import unittest

import numpy as np

from steps.step11 import Variable, Add


class TestStep11(unittest.TestCase):
    def test_check(self):
        xs = [Variable(np.array(2.0)), Variable(np.array(3.0))]
        f = Add()
        ys = f(xs)
        y = ys[0]
        expected = 5
        actual = y.data
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()