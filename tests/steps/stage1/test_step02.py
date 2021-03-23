import unittest

import numpy as np

from steps.stage1.step01 import Variable
from steps.stage1.step02 import Square


class TestStep01(unittest.TestCase):
    def test_check(self):
        x = Variable(np.array(10))
        f = Square()
        y = f(x)

        self.assertIsInstance(y, Variable)
        self.assertEqual(100, y.data)


if __name__ == '__main__':
    unittest.main()
