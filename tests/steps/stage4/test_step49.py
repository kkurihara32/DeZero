import unittest
import math

import numpy as np

import dezero
from dezero.models import MLP
import dezero.functions as F
from dezero import optimizers


def f(x):
    return x / 2.0


class TestStep49(unittest.TestCase):
    def test_check(self):
        train_set = dezero.datasets.Spiral(train=True)
        print(train_set[0])
        print(len(train_set))
        print("-"*30)

        train_set = dezero.datasets.Spiral(transform=f)

        batch_index = [0, 1, 2]
        batch = [train_set[i] for i in batch_index]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        print(x.shape)
        print(t.shape)
        print("-"*30)

        max_epoch = 300
        batch_size = 30
        hidden_size = 10
        lr = 1.0

        train_set= dezero.datasets.Spiral(train=True)
        model = MLP((hidden_size, 3))
        optimizer = optimizers.SGD(lr).setup(target=model)

        data_size = len(x)
        max_iter = math.ceil(data_size / batch_size)  # 小数点の切り上げ

        for epoch in range(max_epoch):
            index = np.random.permutation(data_size)
            sum_loss = 0

            for i in range(max_iter):
                batch_index = index[i * batch_size: (i + 1) * batch_size]
                batch = [train_set[i] for i in batch_index]
                batch_x = np.array([example[0] for example in batch])
                batch_t = np.array([example[1] for example in batch])

                y = model(batch_x)
                loss = F.softmax_cross_entropy(y, batch_t)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(batch_t)

            avg_loss = sum_loss / data_size
            print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))


if __name__ == '__main__':
    unittest.main()


