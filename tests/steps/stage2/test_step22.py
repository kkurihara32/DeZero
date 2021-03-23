import unittest

import numpy as np

from steps.stage2.step22 import Variable


class TestStep22(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        print(y)
        self.assertEqual(y.data, 5.0)
        y = x + 4
        self.assertEqual(y.data, 6)
        y = 10 + 3 * x + 1
        self.assertEqual(y.data, 17)

        y = x - 1
        self.assertEqual(y.data, 1)
        y = 1 - x
        self.assertEqual(y.data, -1)
        y = x / 4
        self.assertEqual(y.data, 0.5)
        y = 4 / x
        self.assertEqual(y.data, 2)
        y = x ** 3
        self.assertEqual(y.data, 8)


if __name__ == '__main__':
    unittest.main()
