import unittest

import numpy as np

from steps.step06 import Variable, Square, EXP


class TestStep06(unittest.TestCase):
    def test_check(self):
        A = Square()
        B = EXP()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        y.grad = np.array(1.0)
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)
        self.assertTrue(3.29 < x.grad < 3.3)


if __name__ == '__main__':
    unittest.main()