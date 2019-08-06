import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


class Layer:

    '''
     n_input: number of input
     n_neuron: number of neurons in a layer
    '''
    def __init__(self, n_input, n_neuron, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neuron)
        self.activation = activation if activation is not None else "sigmoid"
        self.bias = bias if bias is not None else np.random.rand(n_neuron)
        self.layer_output = None
        self.error = None
        self.delta = None

    def activate(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        self.layer_output = self.__apply_activation_func(weighted_sum)
        return self.layer_output

    def __apply_activation_func(self, weighted_sum):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-weighted_sum))
        if self.activation == "tanh":
            return np.tanh(weighted_sum)

    def get_derivative(self, output):
        if self.activation == "sigmoid":
            return output * (1 - output)
        if self.activation == "tanh":
            return 1 - output ** 2


class NeuralNetwork:

    def __init__(self):
        self.__layers = []
        self.mse_list = []

    def add_layer(self, layer):
        self.__layers.append(layer)

    def __feed_forward(self, X):
        for layer in self.__layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        y = self.__feed_forward(X)
        if (y.ndim == 1):
            return np.argmax(y)
        return np.argmax(y, axis=1)

    def backprpogation(self, X, y, learning_rate):
        output = self.__feed_forward(X)
        index = len(self.__layers) - 1
        while index >= 0:
            layer = self.__layers[index]
            if (layer == self.__layers[-1]):
                '''
                Find the margin of error of the output layer (output) by taking the difference of the predicted
                 output and the actual output (y)
                '''
                layer.error = y - output
                '''
                Apply the derivative of our sigmoid activation function to the output layer error.
                '''
                layer.delta = layer.error * layer.get_derivative(output)
            else:
                previous_layer = self.__layers[index + 1]
                '''
                Use the delta output sum of the output layer error to figure out how much our (hidden) layer contributed 
                to the output error by performing a dot product with  weight matrix. .
                '''
                layer.error = np.dot(previous_layer.weights, previous_layer.delta)
                '''
                Calculate the delta output sum for the hidden layer by applying the derivative of 
                 sigmoid activation function 
                '''
                layer.delta = layer.error * layer.get_derivative(layer.layer_output)
            index -= 1

        # Updating weights
        for i in range(len(self.__layers)):
            layer = self.__layers[i]
            if (i == 0):
                # np.atleast_2d converts 1D array to 2D array
                input_to_use = np.atleast_2d(X).T
            else:
                input_to_use = np.atleast_2d(self.__layers[i - 1].layer_output).T
            layer.weights = layer.weights + layer.delta * input_to_use * learning_rate

    def train(self, X, y, epoch, learning_rate):

        for i in range(epoch):
            for j in range(len(X)):
                self.backprpogation(X[j], y[j], learning_rate)
                mse = np.mean(np.square(y - self.__feed_forward(X)))
            self.mse_list.append(mse)
            if (i % 10 == 0):
                print(f"Epoch {i}  MSE = {mse}")

    @staticmethod
    def accuracy(y_pred, y_true):
        return (y_pred == y_true).mean()


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    neural_network.add_layer(Layer(2, 3))
    neural_network.add_layer(Layer(3, 3))
    neural_network.add_layer(Layer(3, 2))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    errors = neural_network.train(X, y, 1000, 0.3)
    print(neural_network.predict(X))
    plt.plot(neural_network.mse_list)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
