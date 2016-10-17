# Gradient Descent
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as sc

def grad_descent(x, y, m):
    n = len(x)
    p = len(x[0])
    w = np.zeros(shape=(p+1, 1))

    # Appending an extra column of ones
    x0 = np.ones(shape=(n, 1))
    x = np.c_[x0, x]

    for t in range(m):
        delta = calculate_likelihood_gradient(x, y, w)
        eta = calculate_eta(x, y, w, delta)
        # print(np.linalg.norm(delta-old_delta))
        if np.linalg.norm(delta) < 0.0001:
            print("Reached optimal value", t)
            break

        w = w - (eta * delta)
        print(w)
    return w


# def likelihood_function(x, y ,w):
#     a = sc.expit(-1*y*np.dot(x, w))
#     a = np.log(1+a)
#
#     return np.sum(a)


def likelihood_function(x, y, w):
        sum = 0
        for i in range(len(x)):
            sum += np.logaddexp(0,-y[i]*np.dot(x[i], w))
        print(np.shape(np.dot(x[i], w)))
        return sum


def calculate_eta(x,y,w,gr):
    alpha = 0.2
    beta = 0.5
    eta = 1
    while likelihood_function(x, y, w-eta*gr) > likelihood_function(x,y,w) - alpha*eta*math.pow(np.linalg.norm(gr),2):
        eta = beta * eta
    return eta

# def calculate_likelihood_gradient(x, y, w):
#     a = sc.expit(-1*y * np.dot(x, w))
#     x = x*a
#
#     grad = -1 * np.dot(np.transpose(y), x)
#
#     return np.transpose(grad)


def calculate_likelihood_gradient(x, y, w):
    grad = np.zeros(x.shape[1])
    for i in range(x.shape[0]):
        grad = grad + y[i] * x[i] * sigmoid(-1 * y[i] * np.dot(x[i], w))
    return -1 * grad

def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))

def main1():
    X = np.array([[2, 1], [1, 20], [1, 5], [4, 1], [1, 40], [3, 30]])
    Y = np.array([-1, -1, -1, 1, 1, 1]).reshape((len(X), 1))
    M = 10

    W = grad_descent(X, Y, M)
    print(W/np.linalg.norm(W))

    # x0 = np.ones(shape=(len(X), 1))
    # X = np.c_[x0, X]
    # print(calculate_likelihood_gradient(X, Y, W))


def generate_dataset(m, s, n):
    x1, y1 = np.random.multivariate_normal(m, s, n).T
    x1 = np.array(x1).reshape((len(x1), 1))
    y1 = np.array(y1).reshape((len(y1), 1))

    return np.c_[x1, y1]



def main2():
    mu1 = np.array([7,4])
    sig1 = np.array([[9,1.75],[1.75,4]])
    x1, y1 = np.random.multivariate_normal(mu1, sig1, 50).T
    x1 = np.array(x1).reshape((len(x1), 1))
    y1 = np.array(y1).reshape((len(y1), 1))

    X1 = np.c_[x1, y1]
    Y1 = np.array(50*[-1])

    mu2 = np.array([0,0])
    sig2 = np.array([[1,-0.75],[-0.75,9]])
    x2, y2 = np.random.multivariate_normal(mu2, sig2, 50).T
    x2 = np.array(x2).reshape((len(x2), 1))
    y2 = np.array(y2).reshape((len(y2), 1))

    X2 = np.c_[x2, y2]
    Y2 = np.array(50*[1])

    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    M = 302
    W = grad_descent(X, Y, M)
    print(W)

    w = W / W[2]
    x = np.arange(-10, 10, 1)
    plt.plot(x, ((-1)*w[1]*x) - w[0])


    # plotting graph
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'o')
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bi-variate Gaussian Distribution Classification")
    plt.show()


main1()



