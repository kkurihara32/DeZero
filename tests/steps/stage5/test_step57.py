import unittest

import numpy as np

from steps.stage5.step55 import get_conv_outsize
from dezero import test_mode
import dezero.functions as F


class TestStep57(unittest.TestCase):
    def test_check(self):
        x1 = np.random.rand(1, 3, 7, 7)
        col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
        print(col1.shape)

        x2 = np.random.rand(10, 3, 7, 7)
        kernel_size = (5, 5)
        stride = (1, 1)
        pad = (0, 0)
        col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
        print(col2.shape)


if __name__ == '__main__':
    unittest.main()