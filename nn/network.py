import numpy as np

from nn.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    softmax
)


class FeedForwardNN:

    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        activation="relu",
        weight_init="random"
    ):

        self.weights = []
        self.biases = []

        self.activation_name = activation

        layers = [input_size] + hidden_layers + [output_size]

        for i in range(len(layers) - 1):

            if weight_init == "xavier":

                W = np.random.randn(
                    layers[i],
                    layers[i + 1]
                ) * np.sqrt(1 / layers[i])

            else:

                W = np.random.randn(
                    layers[i],
                    layers[i + 1]
                ) * 0.01

            b = np.zeros((1, layers[i + 1]))

            self.weights.append(W)
            self.biases.append(b)

    def activation(self, x):

        if self.activation_name == "relu":
            return relu(x)

        elif self.activation_name == "sigmoid":
            return sigmoid(x)

        elif self.activation_name == "tanh":
            return tanh(x)

    def activation_derivative(self, x):

        if self.activation_name == "relu":
            return relu_derivative(x)

        elif self.activation_name == "sigmoid":
            return sigmoid_derivative(x)

        elif self.activation_name == "tanh":
            return tanh_derivative(x)

    def forward(self, X):

        self.activations = [X]
        self.z_values = []

        A = X

        for i in range(len(self.weights) - 1):

            Z = np.dot(A, self.weights[i]) + self.biases[i]

            self.z_values.append(Z)

            A = self.activation(Z)

            self.activations.append(A)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]

        self.z_values.append(Z)

        output = softmax(Z)

        self.activations.append(output)

        return output

    def backward(self, X, y):

        m = X.shape[0]

        self.dW = [0] * len(self.weights)
        self.db = [0] * len(self.biases)

        dZ = self.activations[-1] - y

        self.dW[-1] = np.dot(
            self.activations[-2].T,
            dZ
        ) / m

        self.db[-1] = np.sum(
            dZ,
            axis=0,
            keepdims=True
        ) / m

        for i in reversed(range(len(self.weights) - 1)):

            dA = np.dot(
                dZ,
                self.weights[i + 1].T
            )

            dZ = dA * self.activation_derivative(
                self.z_values[i]
            )

            self.dW[i] = np.dot(
                self.activations[i].T,
                dZ
            ) / m

            self.db[i] = np.sum(
                dZ,
                axis=0,
                keepdims=True
            ) / m
