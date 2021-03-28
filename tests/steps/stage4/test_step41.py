import unittest

import numpy as np

from dezero import Variable
from dezero.utils import sum_to
import dezero.functions as F


class TestStep41(unittest.TestCase):
    def test_check(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = F.matmul(x, W)
        y.backward()



if __name__ == '__main__':
    unittest.main()


