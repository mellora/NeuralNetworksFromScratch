import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None  # Added line to prevent warning.

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        pass


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


if __name__ == "__main__":
    layer_1 = LayerDense(2, 5)
    activation_1 = ActivationReLU()

    layer_1.forward(X)
    activation_1.forward(layer_1.output)

    print(activation_1.output)
