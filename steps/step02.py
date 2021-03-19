from steps.step01 import Variable


class Function(object):
    def __call__(self, input: Variable):
        x = input.data
        y = x ** 2
        output = Variable(y)

        return output
