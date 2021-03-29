import unittest

import numpy as np

from dezero import test_mode
import dezero.functions as F


class TestStep54(unittest.TestCase):
    def test_check(self):
        x = np.ones(5)
        print("x: {}".format(x))

        y = F.dropout(x)
        print(y)

        with test_mode():
            y = F.dropout(x)
            print(y)


if __name__ == '__main__':
    unittest.main()