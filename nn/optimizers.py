import numpy as np


class Optimizer:

    def __init__(self, learning_rate=0.001):

        self.learning_rate = learning_rate

    def update(self, model):
        pass


class SGD(Optimizer):

    def update(self, model):

        for i in range(len(model.weights)):

            model.weights[i] -= (
                self.learning_rate * model.dW[i]
            )

            model.biases[i] -= (
                self.learning_rate * model.db[i]
            )


class Momentum(Optimizer):

    def __init__(self, learning_rate=0.001, beta=0.9):

        super().__init__(learning_rate)

        self.beta = beta

        self.vdw = None
        self.vdb = None

    def update(self, model):

        if self.vdw is None:

            self.vdw = [
                np.zeros_like(w)
                for w in model.weights
            ]

            self.vdb = [
                np.zeros_like(b)
                for b in model.biases
            ]

        for i in range(len(model.weights)):

            self.vdw[i] = (
                self.beta * self.vdw[i]
                + (1 - self.beta) * model.dW[i]
            )

            self.vdb[i] = (
                self.beta * self.vdb[i]
                + (1 - self.beta) * model.db[i]
            )

            model.weights[i] -= (
                self.learning_rate * self.vdw[i]
            )

            model.biases[i] -= (
                self.learning_rate * self.vdb[i]
            )
