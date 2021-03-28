import unittest
import math

import numpy as np

import dezero
from dezero import optimizers
from dezero.datasets import Spiral
from dezero.models import MLP
from dezero import DataLoader
import dezero.functions as F


class TestStep49(unittest.TestCase):
    def test_check(self):
        batch_size = 10
        max_epoch = 1

        train_set = Spiral(train=True)
        test_set = Spiral(train=False)
        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        for epoch in range(max_epoch):
            for x, t in train_loader:
                print(x.shape, t.shape)
                break

            for x, t in test_loader:
                print(x.shape, t.shape)
                break

        y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
        t = np.array([1, 2, 0])
        acc = F.accuracy(y, t)
        print(acc)

        print("-" * 30)

        max_epoch = 300
        batch_size = 30
        hidden_size = 10
        lr = 1.0

        train_set = dezero.datasets.Spiral(train=True)
        model = MLP((hidden_size, 3))
        optimizer = optimizers.SGD(lr).setup(target=model)

        data_size = len(x)
        max_iter = math.ceil(data_size / batch_size)  # 小数点の切り上げ

        for epoch in range(max_epoch):
            index = np.random.permutation(data_size)
            sum_loss = 0
            sum_acc = 0

            for x, t in train_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                model.clear_grads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

            avg_loss = sum_loss / data_size
            print('epoch: {}'.format(epoch + 1))
            print('train loss: {:.4f}, accuracy: {:.4f}'.format(
                sum_loss / len(train_set), sum_acc / len(train_set)))

            sum_loss, sum_acc = 0, 0

            with dezero.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = F.softmax_cross_entropy(y, t)
                    acc = F.accuracy(y, t)
                    sum_loss += float(loss.data) * len(t)
                    sum_acc += float(acc.data) * len(t)

            print('test loss: {:.4f}, accuracy: {:.4f}'.format(
                sum_loss / len(test_set), sum_acc / len(test_set)))






if __name__ == '__main__':
    unittest.main()