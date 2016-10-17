# Homework7

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def load_cites():
    file = open("cities.txt")
    cities = list(map(lambda s: s.strip('\n'), file.readlines()))

    return np.array(cities)


def load_distances():
    file = open("distances.txt")
    data = list(map(lambda s: s.strip('\n'), file.readlines()))
    data = list(map(lambda s: s.split(','), data))
    data = list(map(lambda s: (list(map(lambda y: float(y), s))), data))

    return np.array(data)


def calculate_h(n):
    ones = np.ones(shape=(n, n))
    i = np.identity(n)
    h = i - (1/n) * ones

    return h


def calculate_gram_matrix(d):
    h = calculate_h(len(d))
    b = (-0.5) * np.dot(h, np.dot(d, h))

    return b


def plot_cities(cities, points):
    plt.scatter(-points[:,0], -points[:, 1], marker="o", label=cities)
    for i, txt in enumerate(cities):
        plt.annotate(txt, (-points[i, 0], -points[i, 1]))

    plt.title("Plot of US cities on 2D grid")
    plt.show()


def main():
    CITIES = load_cites()
    D = load_distances()
    B = calculate_gram_matrix(D ** 2)
    N = len(D)

    U, EIG, V = np.linalg.svd(B)
    EIG_SQM = np.identity(N) * np.sqrt(EIG)

    Y = np.dot(U, EIG_SQM)
    Y = Y[:, [0, 1]]
    print(B)

    plot_cities(CITIES, Y)


def main2():
    M = np.array([[1,2,3], [4,5,6]])
    U, SING, V = np.linalg.svd(M, full_matrices=0)
    SING_M = np.identity(len(M)) * SING

    W = np.dot(SING_M, V)

    # print(np.shape(U[:, 0]), np.shape(W[0]))
    M2 = np.dot(U[:, 0].reshape(2,1), np.transpose(W[0]).reshape(1,3))
    print(U[:, 0], W[0])


main2()




