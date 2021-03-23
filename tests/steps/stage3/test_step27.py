import unittest

import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph
from steps.stage3.step27 import my_sin, sin


class TestStep26(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        y.backward()

        self.assertTrue(0.707 < y.data < 0.708)
        self.assertTrue(0.707 < x.grad < 0.708)
        x = Variable(np.array(np.pi/4))
        y = my_sin(x)
        y.backward()

        self.assertTrue(0.707 < y.data < 0.708)
        self.assertTrue(0.707 < x.grad < 0.708)
        plot_dot_graph(y, verbose=False, to_file="../my_sin.png")



if __name__ == '__main__':
    unittest.main()


