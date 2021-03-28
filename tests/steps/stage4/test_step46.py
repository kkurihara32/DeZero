import unittest

import numpy as np

from dezero import Variable
from dezero.models import MLP
import dezero.functions as F
from dezero import optimizers




class TestStep45(unittest.TestCase):
    def test_check(self):

        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        model = MLP((hidden_size, 1))
        optimizer = optimizers.SGD(lr)
        optimizer.setup(target=model)

        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y, y_pred)

            model.clear_grads()
            loss.backward()

            optimizer.update()

            if i % 1000 == 0:
                print(loss)


if __name__ == '__main__':
    unittest.main()


