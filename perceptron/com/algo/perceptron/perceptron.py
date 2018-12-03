import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, iteration=10):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = []
        self.errors = []

    def train(self, X, y):

        self.weights = np.zeros(1 + X.shape[1])
        delta = None

        for _ in range(self.iteration):
            delta = 0.0
            for xi, target in zip(X, y):
                error = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += error * xi
                self.weights[0] += error * self.learning_rate
                delta += int(error != 0.0)
            self.errors.append(delta)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

    def net_input(self, X):

        return np.dot(X, self.weights[1:]) + self.weights[0]
