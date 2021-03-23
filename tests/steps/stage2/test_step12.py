import unittest

import numpy as np

from steps.stage2.step13 import Variable, add, square


class TestStep13(unittest.TestCase):
    def test_check(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        z = add(square(x0), square(x1))
        z.backward()
        print(z.data)
        print(x0.grad)
        print(x1.grad)


if __name__ == '__main__':
    unittest.main()
