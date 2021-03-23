import unittest

import numpy as np

from steps.stage2.step17 import Variable,square
from memory_profiler import profile


@profile
def test():
        x = Variable(np.random.randn(10000000))
        y = square(square(square(x)))
        print(y)


class TestStep17(unittest.TestCase):
    def test_check(self):
        # x = Variable(np.array(2.0))
        # a = square(x)
        # y = add(square(a), square(a))
        # y.backward()
        #
        # expected = 32
        # actual = y.data
        #
        # self.assertEqual(expected, actual)
        #
        # expected = 64
        # actual = x.grad
        #
        # self.assertEqual(expected, actual)

        for i in range(15):
            test()


if __name__ == '__main__':
    unittest.main()
