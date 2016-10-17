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
    np.random.shuffle(data)
    y = data[:, 3]
    x = np.delete(data, 3, 1)

    return x, y


def voted_perceptron(data):
    c = [0]
    l = 0
    n = np.shape(data)[0]
    p = np.shape(data)[1] - 1
    w_list = []
    w_list.append(np.zeros(p))
    T = 10

    for t in range(T):
        x, y = get_shuffled_data(data)
        for i in range(n):
            if np.sign(y[i] * np.dot(w_list[l], x[i])) <= 0:
                w_list.append(w_list[l] + y[i] * x[i])
                c.append(1)
                l += 1
            else:
                c[l] = c[l] + 1

    print(l)

    return c, w_list


def classify_averaged_perceptron(w,  x_data):
    y_classified = np.zeros(len(x_data))
    for i in range(len(x_data)):
        if np.sign(np.dot(w, x_data[i])) <= 0.0:
            y_classified[i] = -1
        else:
            y_classified[i] = 1

    return y_classified


def plot_averaged_perceptron(w, data):
    x1_neg, x2_neg, x1_pos, x2_pos = get_plot_data(data)
    plt.scatter(x1_neg, x2_neg, c="red")
    plt.scatter(x1_pos, x2_pos, c="blue")

    xs1 = np.arange(-2, 12)
    xs2 = [(w[0] + w[1] * x1)/(-w[2]) for x1 in xs1]
    plt.plot(xs1, xs2)

    plt.show()

def classify_with_weighted_majority(c, w_list, x_data):
    y_classified = np.zeros(len(x_data))
    for i in range(len(x_data)):
        s = 0
        for j in range(len(c)):
            if np.sign(np.dot(w_list[j], x_data[i])) <= 0.0:
                s -= c[j]
            else:
                s += c[j]

        if np.sign(s) <= 0.0:
            y_classified[i] = -1
        else:
            y_classified[i] = 1

    return y_classified


def classify_kernel_perceptron(alpha, x_test):
    y_classified = np.zeros(len(x_test))
    print(x_test)
    for i in range(len(x_test)):
        s = 0
        for j in range(len(alpha)):
            s += alpha[j][1] * k(alpha[j][0], x_test[i])

        if np.sign(s) <= 0:
            y_classified[i] = -1
        else:
            y_classified[i] = 1

    return y_classified


def phi(x):
    px = []
    for i in range(len(x)):
        for j in range(i,len(x)):
            px.append(x[i]*x[j])

    return px


def voted_kernel_perceptron(data):
    c = [0]
    l = 0
    n = np.shape(data)[0]
    p = np.shape(data)[1] - 1
    w_list = []
    w_list.append(phi(np.zeros(p)))
    T = 10

    for t in range(T):
        x, y = get_shuffled_data(data)
        for i in range(n):
            if np.sign(y[i] * np.dot(w_list[l], phi(x[i]))) <= 0:
                w_list.append(w_list[l] + y[i] * phi(x[i]))
                c.append(1)
                l += 1
            else:
                c[l] = c[l] + 1
    print(l)

    return c, w_list


def kernel_perceptron_dual(data):
    n = np.shape(data)[0]
    alpha = []
    T = 1

    for t in range(T):
        x, y = get_shuffled_data(data)
        # t += 1
        # Length of alpha before the for loo
        alpha_len = len(alpha)
        for i in range(n):
            sum = 0
            for j in range(len(alpha)):
                sum += alpha[j][1] * k(alpha[j][0], x[i])

            if sum * y[i] <= 0:
                alpha.append((x[i], y[i]))
        # print(t)
        # if alpha_len == len(alpha):
        #     break
    print(alpha)
    return alpha


def k(x, z):
    # a = math.pow(np.linalg.norm(x-z), 2) / (2 * 1)
    # return np.exp(-a)
    return math.pow(1 + np.dot(x, z), 2)


def voted_perceptron_downsample(data, L):
    c = [0]
    l = 0
    n = np.shape(data)[0]
    p = np.shape(data)[1] - 1
    w_list = []
    w_list.append(np.zeros(p))
    T = 10

    for t in range(T):
        x, y = get_shuffled_data(data)
        for i in range(n):
            if np.sign(y[i] * np.dot(w_list[l], x[i])) <= 0:
                if l + 1 >= L:
                    min_index = np.argmin(c)
                    del c[min_index]
                    del w_list[min_index]
                    l -= 1

                w_list.append(w_list[l] + y[i] * x[i])
                c.append(1)
                l += 1
            else:
                c[l] = c[l] + 1

    print(l)

    return c, w_list


def averaged_perceptron_downsample(data):
    c = 0
    n = np.shape(data)[0]
    p = np.shape(data)[1] - 1
    w_final = w_update = np.zeros(p)
    T = 10

    for t in range(T):
        x, y = get_shuffled_data(data)
        for i in range(n):
            if np.sign(y[i] * np.dot(w_update, x[i])) <= 0:
                w_final += c * w_update
                w_update = w_update + (y[i] * x[i])
                c = 1
            else:
                c = c + 1

    return w_final


def get_plot_data(data):
    col = np.shape(data)[1] - 1
    data_neg = data[data[:, col] == -1, :]  # extract all rows with the last column -1
    data_pos = data[data[:, col] == 1, :]  # extract all rows with the last column 1

    x1_neg = data_neg[:, 1]
    x2_neg = data_neg[:, 2]

    x1_pos = data_pos[:, 1]
    x2_pos = data_pos[:, 2]

    return x1_neg, x2_neg, x1_pos, x2_pos


def draw_contour(c, w_list, data):
    x1_neg, x2_neg, x1_pos, x2_pos = get_plot_data(data)

    test_x = []
    for i in range(-2, 12):
        for j in range(-2, 12):
            test_x.append([1, i, j])

    test_x = np.array(test_x)

    x = y = np.arange(-2, 12)
    x1, x2 = np.meshgrid(x, y)
    z = classify_with_weighted_majority(c, w_list, test_x)
    z = z.reshape(x1.shape)

    plt.contourf(x1, x2, z)
    plt.scatter(x1_neg, x2_neg, c="red")
    plt.scatter(x1_pos, x2_pos, c="blue")

    plt.show()


def draw_contour_kernel_perceptron(alpha, data):
    x1_neg, x2_neg, x1_pos, x2_pos = get_plot_data(data)

    test_x = []
    for i in range(-2, 12):
        for j in range(-2, 12):
            test_x.append([1, i, j])

    test_x = np.array(test_x)

    x = y = np.arange(-2, 12)
    x1, x2 = np.meshgrid(x, y)
    z = classify_kernel_perceptron(alpha, test_x)
    print(z)
    z = z.reshape(x1.shape)

    plt.contourf(x1, x2, z)
    plt.scatter(x1_neg, x2_neg, c="red")
    plt.scatter(x1_pos, x2_pos, c="blue")

    plt.show()


def main():
    data = get_data_from_file("data2.txt")

    # C, W_LIST = voted_perceptron(data)
    # C, W_LIST = voted_perceptron_downsample(data, 2)
    # draw_contour(C, W_LIST, data)

    # W = averaged_perceptron_downsample(data)
    # plot_averaged_perceptron(W, data)

    ALPHA = kernel_perceptron_dual(data)
    draw_contour_kernel_perceptron(ALPHA, data)


main()