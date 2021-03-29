import unittest

import numpy as np

from steps.stage5.step55 import get_conv_outsize
from dezero import test_mode
import dezero.functions as F



class TestStep55(unittest.TestCase):
    def test_check(self):
        H, W = 4, 4
        KH, KW = 3, 3
        SH, SW = 1, 1
        PH, PW = 1, 1

        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)
        print(OH, OW)


if __name__ == '__main__':
    unittest.main()