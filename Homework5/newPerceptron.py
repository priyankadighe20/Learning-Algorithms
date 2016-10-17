# Homework 5
import numpy as np
import matplotlib.pyplot as plt
import math


def get_data_from_file(filename):
    file = open(filename)
    data = file.readlines()
    data = list(map(lambda s: s.strip('\n'), data))
    data = list(map(lambda s: s.strip(' ').split(' '), data))
    data = list(map(lambda s: (list(map(lambda y: int(y), s))), data))
    data = np.array(data)

    # Appending an extra column of ones
    x0 = np.ones(shape=(len(data), 1))
    data = np.c_[x0, data]

    return data


def get_shuffled_data(data):
    # np.random.shuffle(data)
    y = data[:, 3]
    x = np.array(data[:,[0,1,2]])

    return x, y


def k(x, z):
    # norm = -1*(pow(np.linalg.norm(x-z),2))/(2*1)
    # return math.exp(norm)

    return pow(1 + np.dot(x,z), 2)


def kernel_perceptron_dual(data):
    n = np.shape(data)[0]
    alpha = np.zeros(len(data))
    x, y = get_shuffled_data(data)
    ctr = 0
    while True:
        flag = False
        ctr += 1
        if ctr%100 == 0:
            print(ctr)
        for i in range(n):
            a_sum = 0
            for j in range(len(alpha)):
                a_sum += alpha[j] * y[j] * k(x[j], x[i])

            if a_sum * y[i] <= 0:
                alpha[i] += 1
                flag = True

        if flag == False:
            break

    print(alpha)
    return x, y, alpha


def classify_kernel_perceptron(x, y, alpha, x_test):
    y_classified = np.zeros(len(x_test))
    for i in range(len(x_test)):
        s = 0
        for j in range(len(alpha)):
            s += alpha[j] * y[j] * k(x[j], x_test[i])

        y_classified[i] = math.copysign(1,s)

    return y_classified


def draw_contour_kernel_perceptron(x_data, y_data, alpha, data):
    x1_neg, x2_neg, x1_pos, x2_pos = get_plot_data(data)

    testx = []
    for i in range(-2, 12):
        for j in range (-2, 12):
            testx.append([1,i, j])

    x_min, x_max = -2, 12
    y_min, y_max = -2, 12

    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))

    z = classify_kernel_perceptron(x_data, y_data, alpha, testx)
    print(z)

    z = z.reshape(xx.shape)

    plt.contourf(yy, xx, z, cmap=plt.cm.Paired)
    plt.scatter(x1_neg, x2_neg, c="red")
    plt.scatter(x1_pos, x2_pos, c="blue")

    plt.show()


def get_plot_data(data):
    col = np.shape(data)[1] - 1
    data_neg = data[data[:, 3] == -1, :]  # extract all rows with the last column -1
    data_pos = data[data[:, 3] == 1, :]  # extract all rows with the last column 1

    x1_neg = data_neg[:, 1]
    x2_neg = data_neg[:, 2]

    x1_pos = data_pos[:, 1]
    x2_pos = data_pos[:, 2]

    return x1_neg, x2_neg, x1_pos, x2_pos


def main():
    data = get_data_from_file("data1.txt")

    X, Y, ALPHA = kernel_perceptron_dual(data)
    draw_contour_kernel_perceptron(X, Y, ALPHA, data)


main()