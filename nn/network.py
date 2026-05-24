import numpy as np

from nn.activations import relu, softmax


class FeedForwardNN:

    def __init__(self, input_size, hidden_layers, output_size):

        self.weights = []
        self.biases = []

        layers = [input_size] + hidden_layers + [output_size]

        for i in range(len(layers) - 1):

            W = np.random.randn(layers[i], layers[i + 1]) * 0.01
            b = np.zeros((1, layers[i + 1]))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):

        self.activations = [X]

        A = X

        for i in range(len(self.weights) - 1):

            Z = np.dot(A, self.weights[i]) + self.biases[i]

            A = relu(Z)

            self.activations.append(A)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]

        output = softmax(Z)

        self.activations.append(output)

        return output
