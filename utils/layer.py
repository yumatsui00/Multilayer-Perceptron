import numpy as np




def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    maximum = max(z)
    z = z - maximum
    total = sum(np.exp(z))
    return np.exp(z) / total


activation_functions = {
    "sigmoid": sigmoid,
    "softmax": softmax
}

class Layer():
    """Layer class"""
    def __init__(self, input, output, activation):
        self.input = input
        self.output = output
        if activation in activation_functions:
            self.activation = activation_functions[activation]
        else:
            raise AssertionError("Invalid argument for activation in Layer")

    