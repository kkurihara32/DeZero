import unittest

import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


class TestStep33(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = "x"
        y.name = "y"
        y.backward(create_graph=True)

        iters = 7

        for i in range(iters):
            print("wawa")
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)

        gx = x.grad
        gx.name = "gx" + str(iters+1)
        plot_dot_graph(gx, verbose=False, to_file="tang_{}_.png".format(iters+1))


if __name__ == '__main__':
    unittest.main()
