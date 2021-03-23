import unittest

import numpy as np

from dezero import Variable
from steps.stage3.step29 import f, gx2


class TestStep29(unittest.TestCase):
    def test_check(self):
        x0 = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            print(x0)
            y = f(x0)
            x0.clear_grad()
            y.backward()

            x0._data -= x0.grad / gx2(x0.data)


if __name__ == '__main__':
    unittest.main()