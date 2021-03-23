import unittest

import numpy as np

from dezero import Variable
from steps.stage3.step33 import f


class TestStep33(unittest.TestCase):
    def test_check(self):
        # x = Variable(np.array(2.0))
        # y = f(x)
        # y.backward(create_graph=True)
        # print(x.grad)
        # gx = x.grad
        # x.clear_grad()
        # gx.backward()
        # print(x.grad)

        x = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            print(i, x)

            y = f(x)
            x.clear_grad()
            y.backward(create_graph=True)

            gx = x.grad
            x.clear_grad()
            gx.backward()
            gx2 = x.grad

            x.data -= gx.data / gx2.data


if __name__ == '__main__':
    unittest.main()
