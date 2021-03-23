import unittest

import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


class TestStep36(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()

        z = gx ** 3 + y
        z.backward()
        print(x.grad)



if __name__ == '__main__':
    unittest.main()
