import unittest

import numpy as np

from dezero import Variable,Parameter
from dezero.utils import sum_to
import dezero.functions as F
import dezero.layers as L


class TestStep44(unittest.TestCase):
    def test_check(self):
        layer = L.Layer()
        layer.p1 = Parameter(np.array(1))
        layer.p2 = Parameter(np.array(2))
        layer.p3 = Variable(np.array(3))
        layer.p4 = "test"

        print(layer._params)
        print("-"*20)
        for name in layer._params:
            print(name, layer.__dict__[name])
        print("-"*20)

        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        l1 = L.Linear(10)
        l2 = L.Linear(1)


        def predict(x):
            y = l1(x)
            y = F.sigmoid(y)
            y = l2(y)
            return y

        lr = 0.2
        iters = 10000

        for i in range(iters):
            y_pred = predict(x)
            loss = F.mean_squared_error(y, y_pred)

            l1.clear_grads()
            l2.clear_grads()
            loss.backward()

            for l in [l1, l2]:
                for p in l.params():
                    p.data -= lr * p.grad.data
            if i % 1000 == 0:
                print(loss)



if __name__ == '__main__':
    unittest.main()


