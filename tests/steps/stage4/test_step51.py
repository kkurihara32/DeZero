import unittest
import math

import numpy as np

import dezero
from mnist import load_mnist
from dezero import optimizers
from dezero.models import MLP
from dezero import DataLoader
import dezero.functions as F


class TestStep51(unittest.TestCase):
    def test_check(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
        print(x_train.shape)
        print(t_train.shape)
        print(x_test[0].shape)
        print(t_test[0])
        print("-"*30)

        train_set = list()
        test_set = list()

        for x, t in zip(x_train, t_train):
            train_set.append((x, t))

        for x, t in zip(x_test, t_test):
            test_set.append((x, t))

        x, t = train_set[0]
        print(type(x), x.shape)
        print(t)

        print("-"*30)

        max_epoch = 5
        batch_size = 100
        hidden_size = 1000

        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        # model = MLP((hidden_size, 10))
        model = MLP((hidden_size, 10), activation=F.relu)
        # optimizer = optimizers.SGD().setup(target=model)
        optimizer = optimizers.Adam().setup(target=model)

        for epoch in range(max_epoch):
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