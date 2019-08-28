import numpy as np

import pandas as pd
import matplotlib.pyplot as pyplot


def visualize_data(x, y):
    pyplot.scatter(x, y, label='linear')
    fig = pyplot.gcf()
    pyplot.gca().grid(True)
    fig.suptitle("Data visualization")
    pyplot.xlabel('Population')
    pyplot.ylabel('Profit')
    pyplot.show()


def visualize_data_with_theta(x, y, theta):
    # x= np.matrix(x, np.float32).reshape(-1, 1)
    pyplot.scatter(x, y)
    axes = pyplot.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = theta[0, 0] + theta[0, 1] * x_vals  # the line equation
    pyplot.plot(x_vals, y_vals, '--')
    pyplot.xlabel('Population')
    pyplot.ylabel('Profit')
    pyplot.show()


def cost_function(x, y, theta):
    X = np.matrix(x, np.float32).reshape(-1, 1)  # -1 in reshape tells numpy size of array
    type(x)
    # creating array of ones
    ones = np.ones((X.shape[0], 1), np.float32)
    X = np.concatenate((ones, X), 1)
    # print(X)
    y = np.matrix(y, np.float32).reshape(-1, 1)
    # print(y)
    # print(theta)
    m = len(x)
    sqrt = np.square(X @ theta.T - y)
    cost = np.sum(sqrt) / (2 * m)

    return cost


def gradient_descent(x, y, theta):
    alpha = 0.01
    iteration = 1300

    # x = np.matrix(x, np.float32).reshape(-1, 1)
    x = np.ndarray(shape=(len(x),1),dtype=np.float, buffer=np.array(x))
    # y = np.matrix(y, np.float32).reshape(-1, 1)
    y=np.ndarray(shape=(len(y),1),dtype=np.float,buffer=np.array(y))
    m = len(x)

    # calculate summation of theta TX

    for i in range(iteration):
        h = theta[0, 0] + theta[0, 1] * x
        # print(h)

        cost_funct_val = cost_function(x, y, theta)
        # print(cost_funct_val)
        difference = h - y
        theta_0 = theta[0, 0] - (alpha / m) * np.sum(difference)
        theta_1 = theta[0, 1] - (alpha / m) * np.sum(np.multiply(difference, x))
        cost_funct_val_new = cost_function(x, y, np.matrix([theta_0, theta_1], np.float32))
        if (cost_funct_val_new < cost_funct_val):
            theta[0, :] = [theta_0, theta_1]

        # print('theta = {}'.format(theta))
        cost_funct_val_new = cost_function(x, y, theta)
        # print('New cost function = {}'.format(cost_funct_val))
    return theta


if __name__ == "__main__":
    data_set = pd.read_csv('/Users/rk185288/PycharmProjects/linear-regression-gradient-descent/dataset.csv')
    if (data_set.notnull and data_set['Population'].notnull and data_set['Profit'].notnull):
        population = data_set['Population'].tolist()
        profit = data_set['Profit'].tolist()
        visualize_data(population, profit)
        theta = np.matrix([-1, 2], np.float32)
        # cost_function(population,profit,theta)
        theta = gradient_descent(population, profit, theta)
        print('theta caluclated by gradient descent methond = {}'.format(theta.tolist()))

        

        predicted_profit = np.matrix([1, 3.5], np.float32) @ theta.T
        predicted_profit = predicted_profit * 10000
        print('Predicted profit for 35000 = {}'.format(predicted_profit[0, 0]))

        predicted_profit = np.matrix([1, 7], np.float32) @ theta.T
        predicted_profit = predicted_profit * 10000
        print('Predicted profit for 70000 = {}'.format(predicted_profit[0, 0]))

        visualize_data_with_theta(population, profit, theta)

    # print(population)
