import math
import random

import numpy as np


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._data_size = len(dataset)
        self._max_iter = math.ceil(self._data_size / self._batch_size)

        self.reset()

    def reset(self):
        self._iteration = 0
        if self._shuffle:
            self._index = np.random.permutation(len(self._dataset))
        else:
            self._index = np.arange(len(self._dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self._iteration >= self._max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self._iteration, self._batch_size
        batch_index = self._index[i * batch_size: (i + 1) * batch_size]
        batch = [self._dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self._iteration += 1
        return x, t

    def next(self):
        return self.__next__()
