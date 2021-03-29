import unittest
import os

import numpy as np

import dezero
from mnist import load_mnist
from dezero import optimizers
from dezero.models import MLP
from dezero import DataLoader
import dezero.functions as F


class TestStep01(unittest.TestCase):
    def test_check(self):
        x = np.array([1, 2, 3])
        np.save("test.npy", x)

        x = np.load("test.npy")
        print(x)

        print("-"*30)

        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])

        np.savez("test.npz", a=x1, b=x2)
        arrays = np.load("test.npz")

        print(arrays["a"])
        print(arrays["b"])

        print("-" * 30)

        data = {"x1": x1, "x2": x2}
        np.savez("test2.npz", **data)

        arrays_2 = np.load("test2.npz")
        print(arrays_2["x1"])
        print(arrays_2["x2"])

        print("-"*30)
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                          normalize=True)

        train_set = list()
        test_set = list()

        for x, t in zip(x_train, t_train):
            train_set.append((x, t))

        for x, t in zip(x_test, t_test):
            test_set.append((x, t))

        max_epoch = 2
        batch_size = 100
        hidden_size = 1000

        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        # model = MLP((hidden_size, 10))
        model = MLP((hidden_size, 10), activation=F.relu)
        # optimizer = optimizers.SGD().setup(target=model)
        optimizer = optimizers.Adam().setup(target=model)

        if os.path.exists("my_mlp.npz"):
            model.load_weights("my_mlp.npz")

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

        model.save_weights("my_mlp.npz")


if __name__ == '__main__':
    unittest.main()