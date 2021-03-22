import unittest

import numpy as np

from steps.step12 import Variable, , add


class TestStep12(unittest.TestCase):
    def test_check(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        print(y.data)
        print(type(y.data))

if __name__ == '__main__':
    unittest.main()
