import unittest

import numpy as np

from dezero.utils import plot_dot_graph


from dezero import Variable


def _goldstein(x, y):
    z = (1 + ((x+y+1)**2) * (19- 14*x + 3*x**2-14*y+6*x*y+3*y**2)) * (30 + (2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return z


class TestStep26(unittest.TestCase):
    def test_check(self):
        # x = Variable(np.random.randn(2, 3))
        # x.name = "x"
        # print(_dot_var(x))
        # print(_dot_var(x, verbose=True))
        # x0 = Variable(np.array(1.0))
        # x1 = Variable(np.array(1.0))
        # y = x0 + x1
        # y.name = "y"
        # txt = _dot_func(y.creator)
        # print(txt)
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = _goldstein(x, y)
        z.backward()
        x.name = "x"
        y.name = "y"
        z.name = "z"
        plot_dot_graph(z, verbose=False, to_file="goldstein.png")


if __name__ == '__main__':
    unittest.main()


