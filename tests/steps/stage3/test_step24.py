import unittest

import numpy as np

from steps.stage3.step24 import sphere, matyas, goldstein
from dezero import Variable


class TestStep24(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(z.data, 2)
        self.assertEqual(x.grad, 2)
        self.assertEqual(y.grad, 2)
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        self.assertTrue(0.04 < x.grad < 0.041)
        self.assertTrue(0.04 < y.grad < 0.041)
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        print(x.grad, y.grad)

if __name__ == '__main__':
    unittest.main()