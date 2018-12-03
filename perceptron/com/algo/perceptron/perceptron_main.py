import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from com.algo.perceptron.perceptron import Perceptron

if __name__ == "__main__":
    data_frame = pd.read_csv('/Users/rk185288/PycharmProjects/perceptron/iris.csv', header=None)
    if (data_frame.notnull):
        # select setosa and versicolor
        y = data_frame.iloc[:100, 4].values
        # Replace  setosa and versicolor with -1 and  1
        y = np.where(y == 'Iris-setosa', -1, 1)

        # extract sepal length and petal length
        X = data_frame.iloc[:100, [0, 2]].values
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='x', label='versicolor')

        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        ppn = Perceptron()
        ppn.train(X, y)

        plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Number of misclassifications')
        plt.tight_layout()
        plt.show()

        #Predict flower type

        # predict=np.dot([6,4],ppn.weights[1:])+ppn.weights[0]
        predict=ppn.predict([6,4])
        if(predict>0):
            print("Flower is versicolor {}".format(predict))
        else:
            print("Flower is setosa {}".format(predict))
