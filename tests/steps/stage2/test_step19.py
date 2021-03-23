import unittest

import numpy as np

from steps.stage2.step19 import Variable


class TestStep19(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        print(len(x))
        print(x)


if __name__ == '__main__':
    unittest.main()
