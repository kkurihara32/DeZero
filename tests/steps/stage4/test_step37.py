import unittest

import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


class TestStep37(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sin(x)
        print(y)
        x0 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        x1 = Variable(np.array([[7, 8, 9], [10, 11, 12]]))
        y = x0 + x1
        print(y)


if __name__ == '__main__':
    unittest.main()


