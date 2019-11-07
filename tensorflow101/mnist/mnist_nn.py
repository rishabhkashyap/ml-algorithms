from mnist.util import get_normalized_data, y2indicator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MNISTNeuralNetwork:

    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.loss = None
        self.nlayer1 = 300
        self.nlayer2 = 100
        self.n_out_layer = 10
        self.cost_list = []

    def train(self, Xtrain, ytrain, Xtest, ytest):
        np.random.seed(100)
        n, d = Xtrain.shape
        Xtrain = Xtrain.astype("float")
        Xtest = Xtest.astype("float")
        ytest_ind = y2indicator(ytest)
        ytrain = y2indicator(ytrain)
        print(Xtrain.dtype)

        batch_sz = 500
        n_batches = n // batch_sz
        w1_values = np.random.rand(d, self.nlayer1) / np.sqrt(d)
        b1_values = np.zeros(self.nlayer1)
        w2_values = np.random.rand(self.nlayer1, self.nlayer2) / np.sqrt(self.nlayer1)
        b2_values = np.zeros(self.nlayer2)
        w3_values = np.random.rand(self.nlayer2, self.n_out_layer) / np.sqrt(self.nlayer2)
        b3_values = np.zeros(self.n_out_layer)
        X = tf.placeholder(dtype=tf.float64, shape=(None, d), name="X")
        y = tf.placeholder(dtype=tf.float64, shape=(None, self.n_out_layer), name="y")
        self.W1 = tf.Variable(w1_values)
        self.W2 = tf.Variable(w2_values)
        self.W3 = tf.Variable(w3_values)
        self.b1 = tf.Variable(b1_values)
        self.b2 = tf.Variable(b2_values)
        self.b3 = tf.Variable(b3_values)
        z1 = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        z2 = tf.nn.relu(tf.matmul(z1, self.W2) + self.b2)
        y_predict = tf.matmul(z2, self.W3) + self.b3

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00004, decay=0.99, momentum=0.9).minimize(cost)
        predict_op = tf.arg_max(y_predict, dimension=1)
        init = tf.global_variables_initializer()
        # training model
        with tf.Session() as session:
            session.run(init)
            for i in range(25):
                for j in range(n_batches):
                    start = j * batch_sz
                    end = start + batch_sz
                    Xbatch = Xtrain[start:end, ]
                    ybatch = ytrain[start:end, ]
                    session.run(optimizer, feed_dict={X: Xbatch, y: ybatch})
                    if (j % 50 == 0):
                        iteration_cost = session.run(cost, feed_dict={X: Xtest, y: ytest_ind})
                        prediction_iter = session.run(predict_op, feed_dict={X: Xtest})
                        err = self.error_rate(prediction_iter, ytest)
                        self.cost_list.append(iteration_cost)
                        print(f"iteration  =  {i}   batch  =  {j} cost  =  {iteration_cost}   error = {err}")
                        # below statement converts tensort into numpy array
                        # print(self.W1.eval())

    def predict(self, Xtest):
        n, d = Xtest.shape
        X = tf.placeholder(dtype=tf.float64, shape=(None, d), name="X")
        z1 = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        z2 = tf.nn.relu(tf.matmul(z1, self.W2) + self.b2)
        y_predict = tf.matmul(z2, self.W3) + self.b3

        return tf.arg_max(y_predict,dimension=1)

    def plot_cost(self):
        plt.plot(self.cost_list)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    def error_rate(self, p, t):
        return np.mean(p!=t)


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = get_normalized_data()
    # print(Xtrain.astype("float").dtype)
    mnist_nn = MNISTNeuralNetwork()
    mnist_nn.train(Xtrain, ytrain, Xtest, ytest)
    mnist_nn.plot_cost()
