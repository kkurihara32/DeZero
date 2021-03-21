import unittest

import numpy as np
from steps.step07 import Variable, Function, Square, EXP


class TestStep07(unittest.TestCase):
    def test_check(self):
        # x = Variable(np.array(0.5))
        # A = Square()
        # B = EXP()
        # C = Square()
        #
        # a = A(x)
        # b = B(a)
        # c = C(b)
        #
        # self.assertTrue(c.creator == C)
        # self.assertTrue(c.creator.input == b)
        # self.assertTrue(c.creator.input.creator == B)
        # self.assertTrue(c.creator.input.creator.input == a)
        # self.assertTrue(c.creator.input.creator.input.creator == A)
        # self.assertTrue(c.creator.input.creator.input.creator.input == x)
        A = Square()
        B = EXP()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        y.grad = np.array(1.0)
        y.backward()
        print(x.grad)
        self.assertTrue(x.grad is not None)
        self.assertTrue(3.29 < x.grad < 3.3)


if __name__ == '__main__':
    unittest.main()
