import numpy as np


class Variable(object):
    def __init__(self, data: np.ndarray):
        self.data = data
