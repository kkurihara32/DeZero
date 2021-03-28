import unittest

import numpy as np

from dezero import Variable
from dezero.models import MLP
import dezero.functions as F
from dezero import optimizers


class TestStep47(unittest.TestCase):
    def test_check(self):
        model = MLP((10, 3))
        x = Variable(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
        y = model(x)
        print(y)

        x = Variable(
            np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
        t = np.array([0, 2, 0, 1])
        y = model(x)

        loss = F.softmax_cross_entropy(y, t)
        print(loss)


if __name__ == '__main__':
    unittest.main()


