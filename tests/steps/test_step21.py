import unittest

import numpy as np

from steps.step21 import Variable, square, add, Config, using_config, no_grad


class TestStep21(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        print(y)
        self.assertEqual(y.data, 5.0)
        y = x + 4
        self.assertEqual(y.data, 6)
        y = 10 + 3 * x + 1
        self.assertEqual(y.data, 17)


if __name__ == '__main__':
    unittest.main()
