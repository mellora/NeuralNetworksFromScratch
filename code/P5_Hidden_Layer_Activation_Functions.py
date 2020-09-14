import numpy as np

x = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]


inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))

print(output)

# class LayerDense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))
#         self.output = None  # Added line to prevent warning.
#
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases
#
#
# if __name__ == "__main__":  # Added to code
#     layer_1 = LayerDense(4, 5)
#     layer_2 = LayerDense(5, 2)
#
#     layer_1.forward(x)
#     layer_2.forward(layer_1.output)
#     print(layer_2.output)
