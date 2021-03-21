import unittest

import numpy as np

from steps.step01 import Variable
from steps.step02 import Square
from steps.step03 import EXP


class TestStep01(unittest.TestCase):
    def test_check(self):
        A = Square()
        B = EXP()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        c = C(b)
        self.assertTrue(1.64 < c.data < 1.65)


if __name__ == '__main__':
    unittest.main()